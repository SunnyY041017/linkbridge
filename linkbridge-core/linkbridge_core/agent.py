"""BaseAgent — 所有 Agent 的抽象基类"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from linkbridge_core.llm_hub import BaseLLMClient, ChatMessage, LLMResponse
from linkbridge_core.message import AgentStatus, TaskRequest, TaskResult
from linkbridge_core.tool import Tool, ToolRegistry


class BaseAgent(ABC):
    """
    Agent 抽象基类——定义 Agent 生命周期。

    生命周期: init → plan → execute → respond

    子类需实现：
    - system_prompt: 系统提示词
    - execute():     核心执行逻辑
    """

    name: str = "base_agent"
    description: str = "基础 Agent"

    def __init__(
        self,
        llm_client: BaseLLMClient,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        self.llm = llm_client
        self.tools = tool_registry or ToolRegistry()
        self._status = AgentStatus.IDLE

    @property
    def status(self) -> AgentStatus:
        return self._status

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        ...

    def _build_messages(self, user_message: str, history: list[dict] = None) -> list[ChatMessage]:
        msgs = [ChatMessage(role="system", content=self.system_prompt)]
        if history:
            for h in history:
                msgs.append(ChatMessage(role=h.get("role", "user"), content=h.get("content", "")))
        msgs.append(ChatMessage(role="user", content=user_message))
        return msgs

    async def run(self, task: TaskRequest, history: list[dict] = None) -> TaskResult:
        """执行一个任务并返回结果"""
        self._status = AgentStatus.THINKING
        try:
            result_content = await self.execute(task, history)
            self._status = AgentStatus.DONE
            return TaskResult(
                task_id=task.task_id,
                agent_name=self.name,
                content=result_content,
                status=AgentStatus.DONE,
            )
        except Exception as e:
            self._status = AgentStatus.ERROR
            return TaskResult(
                task_id=task.task_id,
                agent_name=self.name,
                content="",
                status=AgentStatus.ERROR,
                error=str(e),
            )

    async def run_stream(
        self, task: TaskRequest, history: list[dict] = None
    ) -> AsyncIterator[str]:
        """流式执行任务"""
        self._status = AgentStatus.EXECUTING
        async for chunk in self.execute_stream(task, history):
            yield chunk
        self._status = AgentStatus.DONE

    @abstractmethod
    async def execute(self, task: TaskRequest, history: list[dict] = None) -> str:
        """核心执行逻辑——子类必须实现"""
        ...

    async def execute_stream(
        self, task: TaskRequest, history: list[dict] = None
    ) -> AsyncIterator[str]:
        """流式执行——默认调用非流式，子类可覆盖"""
        result = await self.execute(task, history)
        yield result

    async def call_llm(
        self,
        user_message: str,
        history: list[dict] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """快捷 LLM 调用"""
        msgs = self._build_messages(user_message, history)
        return await self.llm.chat(msgs, temperature=temperature)

    async def call_llm_stream(
        self,
        user_message: str,
        history: list[dict] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """快捷 LLM 流式调用"""
        msgs = self._build_messages(user_message, history)
        async for chunk in self.llm.chat_stream(msgs, temperature=temperature):
            yield chunk
