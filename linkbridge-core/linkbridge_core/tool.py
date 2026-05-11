"""Tool 抽象与注册中心"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ToolParameter:
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True


@dataclass
class Tool:
    """Agent 可调用的工具"""
    name: str
    description: str
    func: Callable
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_openai_schema(self) -> dict:
        """转换为 OpenAI Function Calling 格式"""
        properties = {}
        required = []
        for p in self.parameters:
            properties[p.name] = {"type": p.type, "description": p.description}
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    async def execute(self, **kwargs) -> Any:
        """执行工具"""
        import asyncio

        if asyncio.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        return self.func(**kwargs)


class ToolRegistry:
    """全局工具注册中心"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_many(self, names: list[str]) -> list[Tool]:
        return [t for n in names if (t := self._tools.get(n))]

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def to_openai_schemas(self, names: list[str]) -> list[dict]:
        return [t.to_openai_schema() for n in names if (t := self._tools.get(n))]
