"""
Orchestrator — Multi-Agent 编排器。

核心职责：
1. 意图识别 → 决定需要哪些 Agent
2. 任务分解 → 将用户查询拆解为子任务
3. 并行调度 → asyncio.gather 并行执行独立子任务
4. 结果聚合 → 合成各 Agent 输出为统一报告
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from linkbridge_core.agent import BaseAgent
from linkbridge_core.llm_hub import BaseLLMClient, ChatMessage
from linkbridge_core.message import TaskRequest, TaskResult, AgentStatus


class TaskDependency(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


@dataclass
class SubTask:
    agent_name: str
    instruction: str
    context: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)


@dataclass
class OrchestrationPlan:
    original_query: str
    sub_tasks: list[SubTask]
    reasoning: str = ""


ORCHESTRATOR_PROMPT = """你是一个 Multi-Agent 系统的任务编排器。你的职责是将用户的投资研究问题分解为子任务，分配给合适的专业 Agent。

## 可用的 Agent
{agent_descriptions}

## 输出格式
严格按照 JSON 格式输出，不要包含其他内容：
```json
{{
  "reasoning": "任务分解思路（一句话）",
  "sub_tasks": [
    {{
      "agent_name": "agent_name_here",
      "instruction": "给这个 Agent 的具体指令",
      "depends_on": []
    }}
  ]
}}
```

## 规则
1. 独立子任务之间 depends_on 留空数组
2. 有依赖的子任务在 depends_on 中注明依赖哪个 Agent 的输出
3. 优先并行化：能并行的任务不要串行
4. 每个子任务的 instruction 要具体、可执行
5. 至少分配 2 个 Agent，最多 5 个
"""

SYNTHESIS_PROMPT = """你是一位资深投资研究总监。以下是多位专业分析师对同一问题的独立分析，请综合各方观点形成一份完整的投资研究报告。

## 用户原始问题
{original_query}

## 各分析师报告
{agent_reports}

## 报告格式
使用 Markdown 格式，包含以下部分：
1. **研究摘要** — 2-3 句话核心结论
2. **多维度分析** — 整合各方观点，按逻辑组织
3. **关键指标汇总** — 表格形式
4. **风险提示** — 需要关注的风险点
5. **免责声明** — 本报告由 AI 生成，不构成投资建议
"""


class Orchestrator:
    """
    Multi-Agent 编排器。

    用法:
        orchestrator = Orchestrator(llm, agents={"market": market_agent, ...})
        result = await orchestrator.run("分析宁德时代")
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        agents: dict[str, BaseAgent],
        plan_with_llm: bool = True,
        agent_timeout: float = 60.0,
    ):
        self.llm = llm_client
        self.agents = agents
        self.plan_with_llm = plan_with_llm
        self.agent_timeout = agent_timeout

    def _build_agent_descriptions(self) -> str:
        lines = []
        for name, agent in self.agents.items():
            desc = getattr(agent, "description", "未知 Agent") if agent is not None else "未知 Agent"
            lines.append(f"- **{name}** ({desc})")
        return "\n".join(lines)

    async def plan(self, query: str) -> OrchestrationPlan:
        """使用 LLM 将用户查询分解为子任务"""
        if not self.plan_with_llm:
            return self._default_plan(query)

        prompt = ORCHESTRATOR_PROMPT.format(
            agent_descriptions=self._build_agent_descriptions()
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content=f"请将以下投资研究问题分解为子任务：\n{query}",
            ),
        ]
        response = await self.llm.chat(messages, temperature=0.1)
        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            data = json.loads(content)
            sub_tasks = [
                SubTask(
                    agent_name=t["agent_name"],
                    instruction=t["instruction"],
                    depends_on=t.get("depends_on", []),
                )
                for t in data.get("sub_tasks", [])
            ]
            return OrchestrationPlan(
                original_query=query,
                sub_tasks=sub_tasks,
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, IndexError):
            # LLM 输出格式异常时退回默认计划
            return self._default_plan(query)

    def _default_plan(self, query: str) -> OrchestrationPlan:
        """当 LLM 规划不可用时的兜底计划：分配给所有 Agent"""
        sub_tasks = [
            SubTask(agent_name=name, instruction=query)
            for name in self.agents
        ]
        return OrchestrationPlan(
            original_query=query,
            sub_tasks=sub_tasks,
            reasoning="默认策略：全 Agent 并行分析",
        )

    def _resolve_dependencies(self, plan: OrchestrationPlan) -> list[list[SubTask]]:
        """解析依赖关系，按层级分组（每组内可并行）"""
        results = {}
        remaining = list(plan.sub_tasks)
        groups = []

        while remaining:
            group = []
            still_waiting = []
            for task in remaining:
                if all(dep in results for dep in task.depends_on):
                    group.append(task)
                else:
                    still_waiting.append(task)

            if not group:
                # 存在循环依赖或无法解析的依赖，剩余的全部并行执行
                group = still_waiting
                still_waiting = []

            groups.append(group)
            remaining = still_waiting
            # 标记这些任务已在之前的结果中（防止死循环）
            for t in group:
                results[t.agent_name] = None

        return groups

    async def _execute_task(self, task: SubTask) -> TaskResult:
        """执行单个子任务"""
        agent = self.agents.get(task.agent_name)
        if agent is None:
            return TaskResult(
                task_id="unknown",
                agent_name=task.agent_name,
                content="",
                status=AgentStatus.ERROR,
                error=f"Agent '{task.agent_name}' 未注册",
            )

        task_req = TaskRequest(
            task_id=f"sub_{task.agent_name}_{int(time.time())}",
            agent_name=task.agent_name,
            instruction=task.instruction,
            context=task.context,
        )

        try:
            return await asyncio.wait_for(
                agent.run(task_req), timeout=self.agent_timeout
            )
        except asyncio.TimeoutError:
            return TaskResult(
                task_id=task_req.task_id,
                agent_name=task.agent_name,
                content="",
                status=AgentStatus.ERROR,
                error=f"Agent '{task.agent_name}' 执行超时",
            )

    async def execute_plan(self, plan: OrchestrationPlan) -> dict[str, TaskResult]:
        """按依赖关系分层执行所有子任务"""
        groups = self._resolve_dependencies(plan)
        all_results: dict[str, TaskResult] = {}

        for group in groups:
            tasks = [self._execute_task(t) for t in group]
            results = await asyncio.gather(*tasks)
            for task, result in zip(group, results):
                all_results[task.agent_name] = result
                # 将前一层结果注入后层子任务的 context
                for t in plan.sub_tasks:
                    if task.agent_name in t.depends_on:
                        if result.status == AgentStatus.DONE:
                            t.context[f"dep_{task.agent_name}"] = result.content

        return all_results

    async def synthesize(self, query: str, results: dict[str, TaskResult]) -> str:
        """汇总所有 Agent 结果，生成最终报告"""
        reports = []
        for name, result in results.items():
            if result.status == AgentStatus.DONE:
                reports.append(f"### {name}\n{result.content}")
            else:
                reports.append(f"### {name}\n[错误] {result.error}")

        agent_reports = "\n\n".join(reports)
        prompt = SYNTHESIS_PROMPT.format(
            original_query=query, agent_reports=agent_reports
        )

        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content="请综合以上分析生成最终研究报告。"),
        ]
        response = await self.llm.chat(messages, temperature=0.3)
        return response.content

    async def run(self, query: str, synthesize: bool = True) -> dict:
        """
        一键运行：规划 → 执行 → 合成。

        Returns:
            {
                "plan": OrchestrationPlan,
                "results": {agent_name: TaskResult},
                "synthesis": str | None,
            }
        """
        plan = await self.plan(query)
        results = await self.execute_plan(plan)
        synthesis = await self.synthesize(query, results) if synthesize else None
        return {
            "plan": plan,
            "results": results,
            "synthesis": synthesis,
        }
