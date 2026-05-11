"""
QuantAgent — 量化分析 Agent。

职责：
1. 多资产组合分析（有效前沿、最小方差、最大夏普、风险平价）
2. 因子暴露估计（市场 Beta、规模、价值等）
3. VaR/CVaR 多方法对比
4. 调用 LLM 解读量化结果
"""
import json
from typing import Optional

import numpy as np
import pandas as pd

from linkbridge_core.agent import BaseAgent
from linkbridge_core.llm_hub import BaseLLMClient
from linkbridge_core.message import TaskRequest
from linkbridge_finance.risk import (
    sharpe_ratio, beta, alpha, max_drawdown,
    value_at_risk_historical, value_at_risk_parametric, value_at_risk_monte_carlo,
    cvar_historical, sortino_ratio, calmar_ratio,
)
from linkbridge_finance.portfolio import (
    covariance_matrix, efficient_frontier,
    minimum_variance_portfolio, max_sharpe_portfolio,
    diversification_ratio, portfolio_volatility,
)

from app.data.akshare_provider import AKShareProvider
from app.data.fallback_provider import AutoDataProvider


SYSTEM_PROMPT = """你是一位量化投资策略师，擅长运用数学和统计方法分析市场。

## 你的能力
- 解读风险收益指标（Sharpe、Sortino、Calmar、信息比率）
- 分析投资组合优化结果（有效前沿、最小方差、最大夏普）
- 解读 VaR/CVaR 在不同方法下的含义和差异
- 评估因子暴露和风格归因

## 重要规则
1. 所有量化指标由本地计算引擎精确计算，你必须直接引用提供的具体数值，不要编造数据
2. 区分样本内和样本外分析，注意过拟合风险
3. 量化指标需要结合市场环境解读，孤立的数据可能误导
4. 不做买卖建议，只提供量化分析参考
5. 回复结构：风险收益概况 → 组合优化分析 → VaR 分析 → 综合量化评估
6. **绝对禁止**：不得使用"请补充"、"根据实际填入"等占位词；所有输出数字必须来自提供的实际数据

## 回复格式
使用 Markdown 格式，关键量化指标用表格呈现。
"""


class QuantAgent(BaseAgent):
    name = "quant_agent"
    description = "量化分析 Agent — 组合优化、因子分析、多方法 VaR/CVaR"

    def __init__(
        self,
        llm_client: BaseLLMClient,
        data_provider: Optional[AutoDataProvider] = None,
    ):
        from linkbridge_core.tool import ToolRegistry
        super().__init__(llm_client, ToolRegistry())
        self.data_provider = data_provider or AutoDataProvider(primary=AKShareProvider())

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    async def _compute_quant_metrics(self, symbol: str, days: int = 252) -> dict:
        """计算全量量化指标"""
        import datetime
        end = datetime.date.today().strftime("%Y%m%d")
        start = (datetime.date.today() - datetime.timedelta(days=days + 30)).strftime("%Y%m%d")
        df = await self.data_provider.get_stock_history(symbol, start, end)
        if df.empty:
            return {"error": f"无行情数据: {symbol}"}

        close = pd.to_numeric(df["收盘"], errors="coerce")
        returns = close.pct_change().dropna()
        data_source = df["_source"].iloc[0] if "_source" in df.columns else "live"

        # 沪深300 基准
        try:
            bench_df = await self.data_provider.get_stock_history("000300", start, end)
            bench_close = pd.to_numeric(bench_df["收盘"], errors="coerce")
            bench_returns = bench_close.pct_change().dropna()
        except Exception:
            bench_returns = None

        # 核心风险指标
        ann_vol = float(returns.std() * np.sqrt(252)) if len(returns) > 5 else None
        sr = sharpe_ratio(returns) if len(returns) > 20 else None
        sortino = sortino_ratio(returns) if len(returns) > 20 else None
        mdd = max_drawdown(returns) if len(returns) > 5 else None
        calmar = calmar_ratio(returns) if len(returns) > 20 and mdd and mdd != 0 else None

        b = beta(returns, bench_returns) if bench_returns is not None and len(returns) > 20 else None
        a = alpha(returns, bench_returns) if bench_returns is not None and len(returns) > 20 else None

        # 三方法 VaR
        var_hist = value_at_risk_historical(returns) if len(returns) > 20 else None
        var_param = value_at_risk_parametric(returns) if len(returns) > 5 else None
        var_mc = value_at_risk_monte_carlo(returns) if len(returns) > 5 else None
        cvar = cvar_historical(returns) if len(returns) > 20 else None

        # 简单组合优化（与沪深300做两资产组合）
        port_analysis = None
        if bench_returns is not None and len(returns) > 20:
            try:
                aligned = pd.concat([returns.tail(252), bench_returns.tail(252)], axis=1).dropna()
                if len(aligned) > 60:
                    ann_rets = aligned.mean() * 252
                    cov = aligned.cov() * 252
                    ann_ret_arr = ann_rets.values
                    cov_arr = cov.values

                    min_var = minimum_variance_portfolio(cov_arr)
                    max_sr = max_sharpe_portfolio(ann_ret_arr, cov_arr)
                    port_analysis = {
                        "stock_weight_min_var": round(min_var["weights"][0] * 100, 1),
                        "min_var_volatility": round(min_var["volatility"] * 100, 2),
                        "stock_weight_max_sr": round(max_sr["weights"][0] * 100, 1),
                        "max_sr_return": round(max_sr["return"] * 100, 2),
                        "max_sr_volatility": round(max_sr["volatility"] * 100, 2),
                        "max_sharpe_value": round(max_sr["sharpe"], 2),
                    }
            except Exception:
                pass

        return {
            "symbol": symbol,
            "data_source": data_source,
            "risk_metrics": {
                "annualized_volatility_pct": round(ann_vol * 100, 2) if ann_vol else None,
                "sharpe_ratio": round(sr, 2) if sr else None,
                "sortino_ratio": round(sortino, 2) if sortino else None,
                "calmar_ratio": round(calmar, 2) if calmar else None,
                "max_drawdown_pct": round(mdd * 100, 2) if mdd else None,
                "beta": round(b, 2) if b is not None else None,
                "alpha_pct": round(a * 100, 2) if a is not None else None,
            },
            "var_analysis": {
                "var_historical_pct": round(var_hist * 100, 2) if var_hist else None,
                "var_parametric_pct": round(var_param * 100, 2) if var_param else None,
                "var_monte_carlo_pct": round(var_mc * 100, 2) if var_mc else None,
                "cvar_historical_pct": round(cvar * 100, 2) if cvar else None,
            },
            "portfolio_optimization": port_analysis,
        }

    def _extract_symbol(self, text: str) -> Optional[str]:
        import re
        match = re.search(r'\b(\d{6})\b', text)
        return match.group(1) if match else None

    async def execute(self, task: TaskRequest, history: list[dict] = None) -> str:
        instruction = task.instruction
        ctx = task.context
        symbol = ctx.get("symbol", "")

        stock_code = symbol or self._extract_symbol(instruction)
        if stock_code:
            quant_data = await self._compute_quant_metrics(stock_code)
            if "error" in quant_data:
                return f"获取数据失败: {quant_data['error']}"

            source_note = ""
            if quant_data.get("data_source") == "simulated":
                source_note = "\n（⚠️ 网络受限，以下数据为模拟数据，仅供分析方法参考）"

            data_text = json.dumps(quant_data, ensure_ascii=False, indent=2)
            prompt = f"""请对以下股票进行量化分析：{source_note}

{data_text}

用户问题：{instruction}

请基于以上数据给出专业的量化分析报告。"""
            response = await self.call_llm(prompt, history, temperature=0.3)
            return response.content

        response = await self.call_llm(instruction, history, temperature=0.5)
        return response.content

    async def execute_stream(self, task: TaskRequest, history: list[dict] = None):
        instruction = task.instruction
        ctx = task.context
        symbol = ctx.get("symbol", "")

        stock_code = symbol or self._extract_symbol(instruction)
        if stock_code:
            yield f"🔢 正在计算 {stock_code} 量化风险指标...\n"
            quant_data = await self._compute_quant_metrics(stock_code)
            if "error" in quant_data:
                yield f"获取数据失败: {quant_data['error']}"
                return

            source_note = ""
            if quant_data.get("data_source") == "simulated":
                source_note = "（⚠️ 网络受限，当前使用模拟数据）"
            yield f"📐 量化指标计算完成{source_note}，正在分析...\n\n---\n\n"
            data_text = json.dumps(quant_data, ensure_ascii=False, indent=2)
            prompt = f"""请对以下股票进行量化分析：

{data_text}

用户问题：{instruction}

请基于以上数据给出专业的量化分析报告。"""
            async for chunk in self.call_llm_stream(prompt, history, temperature=0.3):
                yield chunk
            return

        async for chunk in self.call_llm_stream(instruction, history, temperature=0.5):
            yield chunk
