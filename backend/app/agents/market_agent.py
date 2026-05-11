"""
MarketAgent — 行情分析 Agent。

职责：
1. 通过 AKShare 获取行情数据
2. 调用 linkbridge-finance 计算风险收益指标（Beta/Sharpe/波动率/最大回撤）
3. 调用 LLM 解读指标并生成分析报告
"""

import json
import os
from typing import Optional

import pandas as pd

from linkbridge_core.agent import BaseAgent
from linkbridge_core.llm_hub import BaseLLMClient
from linkbridge_core.message import TaskRequest
from linkbridge_core.tool import Tool, ToolRegistry
from linkbridge_finance.risk import (
    annualized_return,
    annualized_volatility,
    beta,
    alpha,
    sharpe_ratio,
    max_drawdown,
    value_at_risk_historical,
)
from linkbridge_finance.technical import sma, ema, rsi, macd as calc_macd

from app.data.akshare_provider import AKShareProvider
from app.data.fallback_provider import AutoDataProvider


MARKET_AGENT_SYSTEM_PROMPT = """你是一位专业的证券市场分析师，擅长解读技术指标和风险指标。

## 你的能力
- 解读股票的价格走势、波动特征和风险水平
- 解释 Sharpe 比率、Beta、Alpha、VaR 等风险收益指标的含义
- 分析 MACD、RSI、均线等关键技术指标信号
- 给出客观、量化的市场分析

## 重要规则
1. 所有量化数据来自本地金融计算引擎，你必须直接引用提供给你的具体数值
2. 用通俗易懂的语言解释专业指标
3. 不做买卖建议，只提供分析参考
4. 回复结构：先概述行情 → 风险收益指标分析 → 技术面分析 → 总结
5. **绝对禁止**：不得使用"请在此处补充"、"根据实际数据填入"、"XX数据待补充"等占位词；不得输出报告模板格式；所有输出的数字必须来自提供给您的实际数据

## 回复格式
使用 Markdown 格式，关键指标用表格呈现。
"""

SYSTEM_PROMPT = MARKET_AGENT_SYSTEM_PROMPT


class MarketAgent(BaseAgent):
    name = "market_agent"
    description = "行情分析 Agent — 获取行情数据并计算风险收益与技术指标"

    def __init__(
        self,
        llm_client: BaseLLMClient,
        data_provider: Optional[AutoDataProvider] = None,
    ):
        super().__init__(llm_client, ToolRegistry())
        self.data_provider = data_provider or AutoDataProvider(primary=AKShareProvider())
        self._register_tools()

    def _register_tools(self):
        self.tools.register(
            Tool(
                name="get_stock_history",
                description="获取股票历史日线数据",
                func=self._fetch_stock_history,
                parameters=[
                    {"name": "symbol", "type": "string", "description": "股票代码（如 000001）", "required": True},
                    {"name": "days", "type": "integer", "description": "获取天数（默认 120）", "required": False},
                ],
            )
        )

    async def _fetch_stock_history(self, symbol: str, days: int = 120):
        import datetime
        end = datetime.date.today().strftime("%Y%m%d")
        start = (datetime.date.today() - datetime.timedelta(days=days)).strftime("%Y%m%d")
        df = await self.data_provider.get_stock_history(symbol, start, end)
        if df.empty:
            return {"error": f"未获取到 {symbol} 的数据"}
        return df.tail(days).to_dict(orient="records")

    async def _compute_indicators(self, symbol: str, days: int = 120) -> dict:
        """核心：拉取数据 → 计算金融指标"""
        import datetime
        end = datetime.date.today().strftime("%Y%m%d")
        start = (datetime.date.today() - datetime.timedelta(days=days + 30)).strftime("%Y%m%d")
        df = await self.data_provider.get_stock_history(symbol, start, end)
        if df.empty:
            return {"error": f"无行情数据: {symbol}"}

        close = pd.to_numeric(df["收盘"], errors="coerce") if "收盘" in df.columns else pd.to_numeric(df.iloc[:, 4], errors="coerce")
        returns = close.pct_change().dropna()
        data_source = df["_source"].iloc[0] if "_source" in df.columns else "live"

        # 以沪深300为基准（近似：用 000300）
        try:
            bench_start = (datetime.date.today() - datetime.timedelta(days=days + 30)).strftime("%Y%m%d")
            bench_df = await self.data_provider.get_stock_history("000300", bench_start, end)
            bench_close = pd.to_numeric(bench_df["收盘"], errors="coerce") if "收盘" in bench_df.columns else pd.to_numeric(bench_df.iloc[:, 4], errors="coerce")
            bench_returns = bench_close.pct_change().dropna()
        except Exception:
            bench_returns = None

        recent_ret = returns.tail(days)
        ann_ret = annualized_return(recent_ret) if len(recent_ret) > 20 else None
        ann_vol = annualized_volatility(recent_ret) if len(recent_ret) > 5 else None
        sr = sharpe_ratio(recent_ret) if len(recent_ret) > 20 else None
        mdd = max_drawdown(recent_ret) if len(recent_ret) > 5 else None
        var95 = value_at_risk_historical(recent_ret, 0.95) if len(recent_ret) > 20 else None

        b = beta(recent_ret, bench_returns) if bench_returns is not None and len(recent_ret) > 20 else None
        a = alpha(recent_ret, bench_returns) if bench_returns is not None and len(recent_ret) > 20 else None

        rsi_val = float(rsi(close).iloc[-1]) if len(close) > 14 else None
        macd_df = calc_macd(close)
        macd_dif = float(macd_df["dif"].iloc[-1]) if len(macd_df) > 0 else None
        macd_signal = float(macd_df["dea"].iloc[-1]) if len(macd_df) > 0 else None

        ma5 = float(sma(close, 5).iloc[-1]) if len(close) >= 5 else None
        ma20 = float(sma(close, 20).iloc[-1]) if len(close) >= 20 else None
        latest_price = float(close.iloc[-1])

        return {
            "symbol": symbol,
            "data_source": data_source,
            "latest_price": round(latest_price, 2),
            "change_1d": round(float(returns.iloc[-1]) * 100, 2) if len(returns) > 0 else None,
            "indicators": {
                "annualized_return": round(ann_ret * 100, 2) if ann_ret is not None else None,
                "annualized_volatility": round(ann_vol * 100, 2) if ann_vol is not None else None,
                "sharpe_ratio": round(sr, 2) if sr is not None else None,
                "max_drawdown": round(mdd * 100, 2) if mdd is not None else None,
                "value_at_risk_95": round(var95 * 100, 2) if var95 is not None else None,
                "beta": round(b, 2) if b is not None else None,
                "alpha": round(a * 100, 2) if a is not None else None,
                "rsi_14": round(rsi_val, 1) if rsi_val is not None else None,
                "macd_dif": round(macd_dif, 2) if macd_dif is not None else None,
                "macd_dea": round(macd_signal, 2) if macd_signal is not None else None,
                "ma_5": round(ma5, 2) if ma5 is not None else None,
                "ma_20": round(ma20, 2) if ma20 is not None else None,
            },
        }

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    async def execute(self, task: TaskRequest, history: list[dict] = None) -> str:
        instruction = task.instruction
        ctx = task.context
        symbol = ctx.get("symbol", "")
        days = ctx.get("days", 120)

        if "股票" in instruction or symbol:
            stock_code = symbol or self._extract_symbol(instruction)
            if stock_code:
                indicators = await self._compute_indicators(stock_code, days)
                if "error" in indicators:
                    return f"获取数据失败: {indicators['error']}"

                source_note = ""
                if indicators.get("data_source") == "simulated":
                    source_note = "\n（注意：网络受限，以下数据为基于历史统计特征的模拟数据，仅供参考分析框架）"

                indicator_text = json.dumps(indicators, ensure_ascii=False, indent=2)
                prompt = f"""请分析以下股票的行情数据：{source_note}

{indicator_text}

用户问题：{instruction}

请基于以上数据给出专业的行情分析报告。"""
                response = await self.call_llm(prompt, history, temperature=0.3)
                return response.content

        response = await self.call_llm(instruction, history, temperature=0.5)
        return response.content

    async def execute_stream(self, task: TaskRequest, history: list[dict] = None):
        instruction = task.instruction
        ctx = task.context
        symbol = ctx.get("symbol", "")
        days = ctx.get("days", 120)

        if "股票" in instruction or symbol:
            stock_code = symbol or self._extract_symbol(instruction)
            if stock_code:
                yield f"🔄 正在获取 {stock_code} 行情数据...\n"
                indicators = await self._compute_indicators(stock_code, days)
                if "error" in indicators:
                    yield f"获取数据失败: {indicators['error']}"
                    return

                source_note = ""
                if indicators.get("data_source") == "simulated":
                    source_note = "（⚠️ 网络受限，当前使用模拟数据）"
                yield f"📊 行情数据获取完成{source_note}，正在计算金融指标...\n\n---\n\n"
                indicator_text = json.dumps(indicators, ensure_ascii=False, indent=2)
                prompt = f"""请分析以下股票的行情数据：

{indicator_text}

用户问题：{instruction}

请基于以上数据给出专业的行情分析报告。"""

                async for chunk in self.call_llm_stream(prompt, history, temperature=0.3):
                    yield chunk
                return

        async for chunk in self.call_llm_stream(instruction, history, temperature=0.5):
            yield chunk

    def _extract_symbol(self, text: str) -> Optional[str]:
        """从用户输入中提取 6 位股票代码"""
        import re
        match = re.search(r'\b(\d{6})\b', text)
        return match.group(1) if match else None
