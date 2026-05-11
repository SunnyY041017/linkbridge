"""
SentiAgent — 舆情情绪分析 Agent。

职责：
1. 搜索/模拟新闻舆情数据
2. 调用 LLM 做情绪解读（正面/负面/中性）
3. 评估市场情绪对股价的潜在影响
"""
import json
from typing import Optional

from linkbridge_core.agent import BaseAgent
from linkbridge_core.llm_hub import BaseLLMClient
from linkbridge_core.message import TaskRequest


SYSTEM_PROMPT = """你是一位资深财经媒体分析师，擅长从海量信息中捕捉市场情绪变化。

## 你的能力
- 从新闻标题和摘要中提取关键信息和情绪倾向
- 识别市场关注的核心话题和争议点
- 评估消息面的短期和中长期影响
- 判断市场情绪的极值点（过度乐观/过度恐慌）

## 重要规则
1. 新闻数据来源有限，分析基于可获得的信息，必须基于实际信息给出具体分析，不得使用占位文本
2. 区分实质性新闻和噪音信息
3. 市场情绪是滞后指标还是领先指标取决于具体情境
4. 不做买卖建议，只提供舆情参考
5. 回复结构：近期重要消息回顾 → 情绪指标评估 → 舆论焦点 → 综合研判
6. **绝对禁止**：不得使用"请补充"、"根据实际填入"等占位词；所有分析必须基于具体信息

## 回复格式
使用 Markdown 格式，情绪信号用表格呈现。
"""


class SentiAgent(BaseAgent):
    name = "senti_agent"
    description = "舆情情绪分析 Agent — 新闻情绪解读、舆论焦点分析、市场心理评估"

    def __init__(
        self,
        llm_client: BaseLLMClient,
    ):
        from linkbridge_core.tool import ToolRegistry
        super().__init__(llm_client, ToolRegistry())

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def _extract_symbol(self, text: str) -> Optional[str]:
        import re
        match = re.search(r'\b(\d{6})\b', text)
        return match.group(1) if match else None

    async def execute(self, task: TaskRequest, history: list[dict] = None) -> str:
        instruction = task.instruction
        ctx = task.context
        symbol = ctx.get("symbol", "")

        stock_code = symbol or self._extract_symbol(instruction)
        # SentiAgent 将已知股票信息隐式传递给 LLM，让它基于训练数据的知识分析
        if stock_code:
            prompt = f"""请分析股票代码 {stock_code} 的市场舆情和投资者情绪。

用户问题：{instruction}

请注意：
- 本系统当前无实时新闻抓取能力
- 请基于你对该公司的公开信息认知，分析当前可能的市场情绪和舆论焦点
- 说明分析的限制性和时效性
- 给出情绪分析框架和关键关注点
- 关键：明确指出你的分析基于知识截止日期前的信息，不是实时数据"""
            response = await self.call_llm(prompt, history, temperature=0.4)
            return response.content

        prompt = f"""请分析以下投资市场的舆情和投资者情绪：

{instruction}

请基于可获得的信息，给出当前市场情绪的评估框架和分析维度。"""
        response = await self.call_llm(instruction, history, temperature=0.5)
        return response.content

    async def execute_stream(self, task: TaskRequest, history: list[dict] = None):
        instruction = task.instruction
        ctx = task.context
        symbol = ctx.get("symbol", "")

        stock_code = symbol or self._extract_symbol(instruction)
        if stock_code:
            yield f"📰 正在分析 {stock_code} 的舆情情绪...\n"
            yield "（⚠️ 当前无实时新闻数据源，基于公开信息进行框架性分析）\n\n---\n\n"
            prompt = f"""请分析股票代码 {stock_code} 的市场舆情和投资者情绪。

用户问题：{instruction}

请注意：
- 本系统当前无实时新闻抓取能力
- 请基于你对该公司的公开信息认知，分析当前可能的市场情绪和舆论焦点
- 说明分析的限制性和时效性
- 给出情绪分析框架和关键关注点
- 关键：明确指出你的分析基于知识截止日期前的信息，不是实时数据"""
            async for chunk in self.call_llm_stream(prompt, history, temperature=0.4):
                yield chunk
            return

        async for chunk in self.call_llm_stream(instruction, history, temperature=0.5):
            yield chunk
