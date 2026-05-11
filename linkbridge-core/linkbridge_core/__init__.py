"""
灵桥 Agent 编排框架 (LinkBridge Core)

核心模块：
- agent.py:   BaseAgent 抽象基类，Agent 生命周期管理
- orchestrator.py: Multi-Agent 编排器，任务分解与调度
- llm_hub.py: 多模型路由层，统一 LLM 调用接口
- tool.py:    Tool 抽象，Agent 工具注册与执行
- message.py: Agent 间消息传递协议
"""

from linkbridge_core.agent import BaseAgent
from linkbridge_core.orchestrator import Orchestrator, OrchestrationPlan, SubTask, TaskDependency
from linkbridge_core.llm_hub import BaseLLMClient, LLMHub, ChatMessage, LLMResponse, CachedLLMClient
from linkbridge_core.tool import Tool, ToolRegistry
from linkbridge_core.message import TaskRequest, TaskResult, AgentStatus, AgentMessage, MessageRole
from linkbridge_core.cache import LLMCache

__all__ = [
    "BaseAgent",
    "Orchestrator",
    "OrchestrationPlan",
    "SubTask",
    "TaskDependency",
    "BaseLLMClient",
    "LLMHub",
    "ChatMessage",
    "LLMResponse",
    "CachedLLMClient",
    "Tool",
    "ToolRegistry",
    "TaskRequest",
    "TaskResult",
    "AgentStatus",
    "AgentMessage",
    "MessageRole",
    "LLMCache",
]
