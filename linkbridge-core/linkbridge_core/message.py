"""Agent 间消息传递协议"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    AGENT = "agent"
    TOOL = "tool"


class AgentStatus(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Agent 间的标准消息格式"""
    id: str
    role: MessageRole
    content: str
    sender: str  # Agent name
    receiver: str = "orchestrator"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskRequest:
    """编排器发给 Agent 的任务"""
    task_id: str
    agent_name: str
    instruction: str
    context: dict[str, Any] = field(default_factory=dict)
    tools: list[str] = field(default_factory=list)


@dataclass
class TaskResult:
    """Agent 返回给编排器的结果"""
    task_id: str
    agent_name: str
    content: str
    status: AgentStatus = AgentStatus.DONE
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
