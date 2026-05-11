"""
RiskAgent — 风险管理 Agent。

职责：
1. 多维度风险评估（利率风险、市场风险、流动性风险）
2. 压力测试场景分析
3. 债券利率风险指标（久期、凸性、DV01）
4. 调用 LLM 解读风险指标
"""
import json
from typing import Optional

import numpy as np
import pandas as pd

from linkbridge_core.agent import BaseAgent
from linkbridge_core.llm_hub import BaseLLMClient
from linkbridge_core.message import TaskRequest
from linkbridge_finance.risk import (
    max_drawdown, max_drawdown_duration, value_at_risk_historical,
    cvar_historical, beta, downside_risk, ulcer_index,
)
from linkbridge_finance.rates import (
    macaulay_duration, modified_duration, convexity,
    bond_price, price_impact_duration,
)

from app.data.akshare_provider import AKShareProvider
from app.data.fallback_provider import AutoDataProvider


SYSTEM_PROMPT = """你是一位首席风控官（CRO），精通各类金融风险的识别、度量和缓释策略。

## 你的能力
- 解读久期和凸性对债券组合的利率风险影响
- 分析多种 VaR 方法（历史/参数/蒙特卡洛）的适用场景和局限性
- 设计压力测试场景并评估极端市场下的潜在损失
- 评估下行风险（Ulcer Index、最大回撤及持续时间）
- 识别集中度风险和流动性风险

## 重要规则
1. 所有风险指标由本地计算引擎精确计算，你必须直接引用提供的具体数值，不要编造数据
2. 风险分析需结合置信度和时间窗口，不同参数下结论可能不同
3. 压力测试是基于"如果...会怎样"的假设性分析，不代表预测
4. 不做买卖建议，只提供风险参考
5. 回复结构：风险总览 → 利率风险 → 市场风险 → 尾部风险（VaR/CVaR）→ 压力测试
6. **绝对禁止**：不得使用"请补充"、"根据实际填入"等占位词；所有输出数字必须来自提供的实际数据

## 回复格式
使用 Markdown 格式，风险指标用表格呈现，标注风险等级（低/中/高）。
"""


class RiskAgent(BaseAgent):
    name = "risk_agent"
    description = "风险管理 Agent — 利率/市场/尾部风险度量、压力测试"

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

    async def _compute_risk_profile(self, symbol: str) -> dict:
        """计算完整的风险画像"""
        import datetime
        end = datetime.date.today().strftime("%Y%m%d")
        start = (datetime.date.today() - datetime.timedelta(days=280)).strftime("%Y%m%d")
        df = await self.data_provider.get_stock_history(symbol, start, end)
        if df.empty:
            return {"error": f"无行情数据: {symbol}"}

        close = pd.to_numeric(df["收盘"], errors="coerce")
        returns = close.pct_change().dropna()
        data_source = df["_source"].iloc[0] if "_source" in df.columns else "live"

        # 基准
        try:
            bench_df = await self.data_provider.get_stock_history("000300", start, end)
            bench_close = pd.to_numeric(bench_df["收盘"], errors="coerce")
            bench_returns = bench_close.pct_change().dropna()
        except Exception:
            bench_returns = None

        b = beta(returns, bench_returns) if bench_returns is not None and len(returns) > 20 else None
        mdd = max_drawdown(returns) if len(returns) > 5 else None
        mdd_dur = max_drawdown_duration(returns) if len(returns) > 5 else 0
        var95 = value_at_risk_historical(returns, 0.95) if len(returns) > 20 else None
        var99 = value_at_risk_historical(returns, 0.99) if len(returns) > 50 else None
        cvar95 = cvar_historical(returns, 0.95) if len(returns) > 20 else None
        downside = downside_risk(returns) if len(returns) > 5 else None
        ui = ulcer_index(returns) if len(returns) > 20 else None

        # 债券利率风险（假设组合中包含国债）
        bond_risk = self._compute_bond_risk()

        # 压力测试场景
        stress = self._stress_test(returns)

        return {
            "symbol": symbol,
            "data_source": data_source,
            "market_risk": {
                "beta": round(b, 2) if b is not None else None,
                "max_drawdown_pct": round(mdd * 100, 2) if mdd else None,
                "max_drawdown_days": mdd_dur,
                "downside_risk_pct": round(downside * 100, 2) if downside else None,
                "ulcer_index_pct": round(ui * 100, 2) if ui else None,
            },
            "tail_risk": {
                "var_95_pct": round(var95 * 100, 2) if var95 else None,
                "var_99_pct": round(var99 * 100, 2) if var99 else None,
                "cvar_95_pct": round(cvar95 * 100, 2) if cvar95 else None,
            },
            "bond_risk": bond_risk,
            "stress_test": stress,
        }

    def _compute_bond_risk(self) -> dict:
        """计算债券利率风险指标（示例国债组合）"""
        # 10年期国债示例：面值100，票息3%，10期年付，YTM 3.5%
        face_value = 100.0
        coupon_rate = 0.03
        periods = 10
        ytm = 0.035

        cash_flows = [coupon_rate * face_value] * (periods - 1) + [coupon_rate * face_value + face_value]
        times = list(range(1, periods + 1))

        dur = macaulay_duration(cash_flows, times, ytm)
        mod_dur = modified_duration(cash_flows, times, ytm)
        conv = convexity(cash_flows, times, ytm)
        price = bond_price(face_value, coupon_rate, periods, ytm)

        # DV01：收益率上升 1bp 的价格变化
        price_up = bond_price(face_value, coupon_rate, periods, ytm + 0.0001)
        dv01 = abs(price_up - price)

        # 收益率 ±50bp 对价格的影响
        price_up50 = bond_price(face_value, coupon_rate, periods, ytm + 0.005)
        price_down50 = bond_price(face_value, coupon_rate, periods, ytm - 0.005)

        return {
            "example_bond": "10年期国债（面值100，票息3%，YTM 3.5%）",
            "macaulay_duration": round(dur, 2) if dur else None,
            "modified_duration": round(mod_dur, 2) if mod_dur else None,
            "convexity": round(conv, 2) if conv else None,
            "dv01": round(dv01, 4),
            "current_price": round(price, 2) if price else None,
            "rate_hike_50bp_impact_pct": round((price_up50 / price - 1) * 100, 2) if price else None,
            "rate_cut_50bp_impact_pct": round((price_down50 / price - 1) * 100, 2) if price else None,
        }

    def _stress_test(self, returns: pd.Series) -> list[dict]:
        """压力测试场景分析"""
        if len(returns) < 20:
            return []

        scenarios = [
            {"name": "2008 金融危机", "multiplier": 1.0, "vol_mult": 3.0, "mu_shift": -0.03},
            {"name": "2015 股灾", "multiplier": 0.8, "vol_mult": 2.5, "mu_shift": -0.02},
            {"name": "2020 疫情冲击", "multiplier": 0.6, "vol_mult": 2.0, "mu_shift": -0.01},
            {"name": "温和下跌", "multiplier": 0.3, "vol_mult": 1.5, "mu_shift": -0.005},
        ]

        mu = returns.mean()
        sigma = returns.std()
        results = []

        for s in scenarios:
            # 用 t-Copula 或简单正态模拟压力情景
            shock_returns = np.random.normal(mu + s["mu_shift"] / 252, sigma * s["vol_mult"], min(252, len(returns)))
            shock_cum = np.cumprod(1 + shock_returns)
            shock_max = np.maximum.accumulate(shock_cum)
            shock_mdd = float(np.min(shock_cum / shock_max - 1))
            shock_var = float(-np.percentile(shock_returns, 5))

            results.append({
                "scenario": s["name"],
                "estimated_max_drawdown_pct": round(shock_mdd * 100, 2),
                "daily_var_95_pct": round(shock_var * 100, 2),
            })

        return results

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
            risk_data = await self._compute_risk_profile(stock_code)
            if "error" in risk_data:
                return f"获取数据失败: {risk_data['error']}"

            source_note = ""
            if risk_data.get("data_source") == "simulated":
                source_note = "\n（⚠️ 网络受限，以下数据为模拟数据，仅供分析方法参考）"

            data_text = json.dumps(risk_data, ensure_ascii=False, indent=2)
            prompt = f"""请对以下股票进行全面的风险分析：{source_note}

{data_text}

用户问题：{instruction}

请基于以上数据给出专业的风险管理报告。"""
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
            yield f"🛡️ 正在评估 {stock_code} 风险画像...\n"
            risk_data = await self._compute_risk_profile(stock_code)
            if "error" in risk_data:
                yield f"获取数据失败: {risk_data['error']}"
                return

            source_note = ""
            if risk_data.get("data_source") == "simulated":
                source_note = "（⚠️ 网络受限，当前使用模拟数据）"
            yield f"⚠️ 风险评估完成{source_note}，包含压力测试...\n\n---\n\n"
            data_text = json.dumps(risk_data, ensure_ascii=False, indent=2)
            prompt = f"""请对以下股票进行全面的风险分析：

{data_text}

用户问题：{instruction}

请基于以上数据给出专业的风险管理报告。"""
            async for chunk in self.call_llm_stream(prompt, history, temperature=0.3):
                yield chunk
            return

        async for chunk in self.call_llm_stream(instruction, history, temperature=0.5):
            yield chunk
