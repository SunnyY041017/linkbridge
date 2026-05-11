"""
TechnAgent — 技术面分析 Agent。

职责：
1. 获取历史 K 线数据
2. 调用 linkbridge-finance 计算全量技术指标（MACD/RSI/KDJ/布林带/均线/ATR/OBV/CCI）
3. 调用 LLM 解读技术信号
"""
import json
from typing import Optional

import pandas as pd

from linkbridge_core.agent import BaseAgent
from linkbridge_core.llm_hub import BaseLLMClient
from linkbridge_core.message import TaskRequest
from linkbridge_finance.technical import (
    sma, ema, macd, rsi, kdj, bollinger_bands, atr, obv,
    ma_cross_signal, support_resistance, williams_r, cci,
)

from app.data.akshare_provider import AKShareProvider
from app.data.fallback_provider import AutoDataProvider


SYSTEM_PROMPT = """你是一位专业的技术分析师，精通多时间周期图表分析。

## 你的能力
- 解读 MACD 金叉/死叉信号和柱状图变化
- 分析 RSI 超买超卖区域和背离
- 评估布林带宽度（波动率）和价格在带内的位置
- 识别均线排列（多头/空头排列）和关键价位
- 解读 KDJ、ATR、OBV、CCI、Williams %R 等辅助指标

## 重要规则
1. 所有量化指标来自本地计算引擎，你必须直接引用提供的具体数值，不要编造数值
2. 技术分析应结合多个指标相互验证，单一指标容易产生假信号
3. 区分趋势市场和震荡市场，不同市场环境下指标可靠性不同
4. 不做买卖建议，只提供技术面参考
5. 回复结构：趋势判断 → 指标信号汇总 → 关键支撑/阻力 → 综合评估
6. **绝对禁止**：不得使用"请补充"、"根据实际填入"等占位词；所有输出数字必须来自提供的实际数据

## 回复格式
使用 Markdown 格式，关键信号用表格呈现。
"""


class TechnAgent(BaseAgent):
    name = "techn_agent"
    description = "技术面分析 Agent — MACD/RSI/KDJ/布林带/均线系统综合研判"

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

    async def _compute_technicals(self, symbol: str, days: int = 180) -> dict:
        """拉取 K 线数据并计算全部技术指标"""
        import datetime
        end = datetime.date.today().strftime("%Y%m%d")
        start = (datetime.date.today() - datetime.timedelta(days=days + 60)).strftime("%Y%m%d")
        df = await self.data_provider.get_stock_history(symbol, start, end)
        if df.empty:
            return {"error": f"无行情数据: {symbol}"}

        close = pd.to_numeric(df["收盘"], errors="coerce")
        high = pd.to_numeric(df["最高"], errors="coerce")
        low = pd.to_numeric(df["最低"], errors="coerce")
        volume = pd.to_numeric(df["成交量"], errors="coerce")
        data_source = df["_source"].iloc[0] if "_source" in df.columns else "live"

        latest_price = float(close.iloc[-1])

        # MACD
        macd_df = macd(close)
        macd_dif = float(macd_df["dif"].iloc[-1]) if len(macd_df) > 0 else None
        macd_dea = float(macd_df["dea"].iloc[-1]) if len(macd_df) > 0 else None
        macd_hist = float(macd_df["macd_hist"].iloc[-1]) if len(macd_df) > 0 else None

        # RSI
        rsi_val = float(rsi(close).iloc[-1]) if len(close) >= 14 else None
        rsi_series = rsi(close) if len(close) >= 14 else None

        # KDJ
        k_val, d_val, j_val = None, None, None
        if len(close) >= 9:
            kdj_df = kdj(high, low, close)
            if not kdj_df.empty:
                k_val = float(kdj_df["k"].iloc[-1])
                d_val = float(kdj_df["d"].iloc[-1])
                j_val = float(kdj_df["j"].iloc[-1])

        # 布林带
        bb = bollinger_bands(close, period=20, std_dev=2) if len(close) >= 20 else pd.DataFrame()
        bb_upper = float(bb["upper"].iloc[-1]) if not bb.empty else None
        bb_lower = float(bb["lower"].iloc[-1]) if not bb.empty else None
        bb_mid = float(bb["middle"].iloc[-1]) if not bb.empty else None
        bb_width = round((bb_upper - bb_lower) / bb_mid * 100, 2) if bb_mid and bb_mid > 0 else None

        # 均线系统
        ma5 = float(sma(close, 5).iloc[-1]) if len(close) >= 5 else None
        ma10 = float(sma(close, 10).iloc[-1]) if len(close) >= 10 else None
        ma20 = float(sma(close, 20).iloc[-1]) if len(close) >= 20 else None
        ma60 = float(sma(close, 60).iloc[-1]) if len(close) >= 60 else None

        # 均线排列
        ma_arrangement = "多头排列" if (ma5 and ma10 and ma20 and ma5 > ma10 > ma20) else (
            "空头排列" if (ma5 and ma10 and ma20 and ma5 < ma10 < ma20) else "交叉/震荡"
        )

        # 均线交叉信号
        cross = ma_cross_signal(close, fast_period=5, slow_period=20) if len(close) >= 20 else pd.Series(dtype=int)
        latest_cross = int(cross.iloc[-1]) if len(cross) > 0 else 0
        cross_signal = {1: "金叉↑", -1: "死叉↓", 0: "无信号"}.get(latest_cross, "无信号")

        # ATR — 14 期平均真实波幅 / 收盘价
        atr_series = atr(high, low, close, period=14) if len(close) >= 14 else None
        atr_pct = round(float(atr_series.iloc[-1]) / latest_price * 100, 2) if atr_series is not None else None

        # OBV 趋势
        obv_series = obv(close, volume) if len(close) >= 20 else None
        obv_trend = None
        if obv_series is not None and len(obv_series) >= 20:
            obv_sma = sma(obv_series, 20)
            obv_trend = "上升" if obv_series.iloc[-1] > obv_sma.iloc[-1] else "下降"

        # Williams %R
        wr = williams_r(high, low, close, period=14) if len(close) >= 14 else None
        wr_val = float(wr.iloc[-1]) if wr is not None else None

        # CCI
        cci_val = float(cci(high, low, close, period=20).iloc[-1]) if len(close) >= 20 else None

        # 支撑阻力位
        sr_levels = {}
        if len(close) >= 20:
            try:
                sr_df = support_resistance(high, low, close)
                if not sr_df.empty:
                    sr_levels = {
                        "support": round(float(sr_df["support"].iloc[-1]), 2),
                        "resistance": round(float(sr_df["resistance"].iloc[-1]), 2),
                        "pivot": round(float(sr_df["pivot"].iloc[-1]), 2),
                    }
            except Exception:
                sr_levels = {}

        return {
            "symbol": symbol,
            "latest_price": latest_price,
            "data_source": data_source,
            "trend": {
                "ma_arrangement": ma_arrangement,
                "ma5": round(ma5, 2) if ma5 else None,
                "ma10": round(ma10, 2) if ma10 else None,
                "ma20": round(ma20, 2) if ma20 else None,
                "ma60": round(ma60, 2) if ma60 else None,
                "ma_cross_signal": cross_signal,
            },
            "momentum": {
                "rsi_14": round(rsi_val, 1) if rsi_val else None,
                "macd_dif": round(macd_dif, 3) if macd_dif else None,
                "macd_dea": round(macd_dea, 3) if macd_dea else None,
                "macd_histogram": round(macd_hist, 3) if macd_hist else None,
                "kdj_k": round(k_val, 1) if k_val else None,
                "kdj_d": round(d_val, 1) if d_val else None,
                "kdj_j": round(j_val, 1) if j_val else None,
                "williams_r": round(wr_val, 1) if wr_val else None,
                "cci": round(cci_val, 1) if cci_val else None,
            },
            "volatility": {
                "bollinger_upper": round(bb_upper, 2) if bb_upper else None,
                "bollinger_mid": round(bb_mid, 2) if bb_mid else None,
                "bollinger_lower": round(bb_lower, 2) if bb_lower else None,
                "bollinger_width_pct": bb_width,
                "atr_pct": atr_pct,
            },
            "volume": {
                "obv_trend": obv_trend,
            },
            "key_levels": sr_levels,
        }

    def _extract_symbol(self, text: str) -> Optional[str]:
        import re
        match = re.search(r'\b(\d{6})\b', text)
        return match.group(1) if match else None

    async def execute(self, task: TaskRequest, history: list[dict] = None) -> str:
        instruction = task.instruction
        ctx = task.context
        symbol = ctx.get("symbol", "")
        days = ctx.get("days", 180)

        stock_code = symbol or self._extract_symbol(instruction)
        if stock_code:
            tech_data = await self._compute_technicals(stock_code, days)
            if "error" in tech_data:
                return f"获取数据失败: {tech_data['error']}"

            source_note = ""
            if tech_data.get("data_source") == "simulated":
                source_note = "\n（⚠️ 网络受限，以下数据为模拟数据，技术形态仅供参考分析方法）"

            data_text = json.dumps(tech_data, ensure_ascii=False, indent=2)
            prompt = f"""请对以下股票进行技术面分析：{source_note}

{data_text}

用户问题：{instruction}

请基于以上数据给出专业的技术分析报告。"""
            response = await self.call_llm(prompt, history, temperature=0.3)
            return response.content

        response = await self.call_llm(instruction, history, temperature=0.5)
        return response.content

    async def execute_stream(self, task: TaskRequest, history: list[dict] = None):
        instruction = task.instruction
        ctx = task.context
        symbol = ctx.get("symbol", "")
        days = ctx.get("days", 180)

        stock_code = symbol or self._extract_symbol(instruction)
        if stock_code:
            yield f"📈 正在获取 {stock_code} K线数据...\n"
            tech_data = await self._compute_technicals(stock_code, days)
            if "error" in tech_data:
                yield f"获取数据失败: {tech_data['error']}"
                return

            source_note = ""
            if tech_data.get("data_source") == "simulated":
                source_note = "（⚠️ 网络受限，当前使用模拟数据）"
            yield f"📉 技术指标计算完成{source_note}，正在分析...\n\n---\n\n"
            data_text = json.dumps(tech_data, ensure_ascii=False, indent=2)
            prompt = f"""请对以下股票进行技术面分析：

{data_text}

用户问题：{instruction}

请基于以上数据给出专业的技术分析报告。"""
            async for chunk in self.call_llm_stream(prompt, history, temperature=0.3):
                yield chunk
            return

        async for chunk in self.call_llm_stream(instruction, history, temperature=0.5):
            yield chunk
