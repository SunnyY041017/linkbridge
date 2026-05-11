"""
FundaAgent — 基本面分析 Agent。

职责：
1. 获取财务数据（PE/PB/ROE/营收增长等）
2. 调用 linkbridge-finance 计算 DCF 估值、Gordon 增长模型
3. 调用 LLM 解读基本面指标
"""
import json
from typing import Optional

from linkbridge_core.agent import BaseAgent
from linkbridge_core.llm_hub import BaseLLMClient
from linkbridge_core.message import TaskRequest
from linkbridge_finance.valuation import (
    pe_ratio, pb_ratio, peg_ratio, ps_ratio,
    dcf_valuation, gordon_growth_model, ev_to_ebitda,
    net_net_working_capital, roe, roa,
    free_cash_flow_yield, dividend_yield, earning_yield,
)

from app.data.akshare_provider import AKShareProvider
from app.data.fallback_provider import AutoDataProvider


SYSTEM_PROMPT = """你是一位资深价值投资分析师，专长于基本面分析和估值建模。

## 你的能力
- 解读 PE、PB、PS、PEG 等估值倍数及其分位数含义
- 分析 ROE、ROA 等盈利质量指标
- 解释 DCF（现金流折现）估值模型结果
- 评估自由现金流和股息质量
- 给出客观的基本面质量评估

## 重要规则
1. 所有量化数据来自本地金融计算引擎，你必须直接引用提供的具体数值，不要编造数据
2. 估值倍数需结合行业和历史对比才有意义
3. 注意区分一次性损益和经常性损益
4. 不做买卖建议，只提供分析参考
5. 回复结构：估值面概览 → 盈利质量 → 估值模型 → 综合判断
6. **绝对禁止**：不得使用"请补充"、"根据实际填入"等占位词；所有输出数字必须来自提供的实际数据

## 回复格式
使用 Markdown 格式，关键指标用表格呈现。
"""


class FundaAgent(BaseAgent):
    name = "funda_agent"
    description = "基本面分析 Agent — 估值分析、盈利质量评估、DCF 估值"

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

    async def _compute_fundamentals(self, symbol: str) -> dict:
        """获取基本面数据并计算估值指标"""
        fin_data = await self.data_provider.get_financial_data(symbol)
        info = await self.data_provider.get_stock_info(symbol)

        price = float(info.get("最新价", 100))
        eps = fin_data.diluted_eps
        bvps = float(info.get("每股净资产", price / fin_data.pb)) if fin_data.pb and fin_data.pb > 0 else price / 2
        sps = price / fin_data.revenue_yoy * 10 if fin_data.revenue_yoy else price / 0.5
        growth_rate = max(fin_data.profit_yoy / 100, 0.02) if fin_data.profit_yoy else 0.05

        return {
            "symbol": symbol,
            "latest_price": price,
            "data_source": fin_data.symbol + "_source" if hasattr(fin_data, '_source') else (
                info.get("_source", "live")
            ),
            "valuation": {
                "pe": round(pe_ratio(price, eps), 2) if eps and eps > 0 else None,
                "pb": round(pb_ratio(price, bvps), 2) if bvps and bvps > 0 else None,
                "ps": round(ps_ratio(price, sps), 2) if sps and sps > 0 else None,
                "peg": round(peg_ratio(pe_ratio(price, eps), growth_rate * 100), 2)
                       if eps > 0 and growth_rate > 0 else None,
                "ev_ebitda": None,
            },
            "profitability": {
                "roe": round(fin_data.roe, 2) if fin_data.roe else None,
                "roa": round(fin_data.roa, 2) if fin_data.roa else None,
                "revenue_growth_yoy": round(fin_data.revenue_yoy, 2) if fin_data.revenue_yoy else None,
                "profit_growth_yoy": round(fin_data.profit_yoy, 2) if fin_data.profit_yoy else None,
                "eps": round(eps, 2) if eps else None,
            },
            "dcf": self._compute_dcf(eps, growth_rate, price),
            "market": {
                "total_market_cap": fin_data.total_market_cap,
                "diluted_eps": fin_data.diluted_eps,
            },
        }

    def _compute_dcf(self, eps: float, growth: float, price: float) -> dict:
        """计算 DCF 估值和 Gordon 增长模型"""
        fcf_current = eps * 0.75 if eps and eps > 0 else 1.0
        # 构建 5 年预测期 FCF（年增长减速）
        fcf_forecast = [fcf_current * (1 + growth) ** i for i in range(1, 6)]
        dcf = dcf_valuation(
            free_cash_flows=fcf_forecast,
            terminal_growth_rate=0.025,
            discount_rate=0.10,
            shares_outstanding=1.0,
        )
        ggm = gordon_growth_model(
            dividend=fcf_current * 0.3,
            required_return=0.10,
            growth_rate=0.025,
        ) if fcf_current > 0 else None
        fair_price = dcf.get("fair_price", 0) if dcf else 0
        return {
            "dcf_fair_price": round(fair_price, 2),
            "ggm_value": round(ggm, 2) if ggm else None,
            "upside_pct": round((fair_price / price - 1) * 100, 2) if fair_price and fair_price > 0 else None,
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
            fundamentals = await self._compute_fundamentals(stock_code)
            source_note = ""
            if fundamentals.get("data_source") == "simulated":
                source_note = "\n（⚠️ 网络受限，以下数据为基于历史统计特征的模拟数据，仅供参考分析框架）"

            data_text = json.dumps(fundamentals, ensure_ascii=False, indent=2)
            prompt = f"""请对以下股票进行基本面分析：{source_note}

{data_text}

用户问题：{instruction}

请基于以上数据给出专业的价值投资分析报告。"""
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
            yield f"📊 正在获取 {stock_code} 基本面数据...\n"
            fundamentals = await self._compute_fundamentals(stock_code)
            source_note = ""
            if fundamentals.get("data_source") == "simulated":
                source_note = "（⚠️ 网络受限，当前使用模拟数据）"
            yield f"📈 财务数据获取完成{source_note}，正在估值分析...\n\n---\n\n"
            data_text = json.dumps(fundamentals, ensure_ascii=False, indent=2)
            prompt = f"""请对以下股票进行基本面分析：

{data_text}

用户问题：{instruction}

请基于以上数据给出专业的价值投资分析报告。"""
            async for chunk in self.call_llm_stream(prompt, history, temperature=0.3):
                yield chunk
            return

        async for chunk in self.call_llm_stream(instruction, history, temperature=0.5):
            yield chunk
