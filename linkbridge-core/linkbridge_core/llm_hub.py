"""
LLM Hub — 多模型路由层。

统一 LLM 调用接口，支持：
- DeepSeek V4 (主力)
- Qwen3 (Tongyi)
- GLM-4 (Zhipu)
- 本地 Ollama 模型

所有 OpenAI-compatible 的 API 使用通用适配器。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI


class ModelProvider(str, Enum):
    DEEPSEEK = "deepseek"
    TONGYI = "tongyi"
    ZHIPU = "zhipu"
    OLLAMA = "ollama"


@dataclass
class LLMConfig:
    provider: ModelProvider
    model: str
    api_key: str
    base_url: str
    temperature: float = 0.7
    max_tokens: int = 4096
    extra_headers: dict = field(default_factory=dict)


@dataclass
class ChatMessage:
    role: str  # system / user / assistant
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    finish_reason: str = ""


class BaseLLMClient(ABC):
    """LLM 客户端抽象基类"""

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        ...


class OpenAICompatibleClient(BaseLLMClient):
    """OpenAI-compatible API 通用适配器 (DeepSeek / Qwen / GLM 等)"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            default_headers=config.extra_headers,
        )

    def _build_messages(self, messages: list[ChatMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def chat(
        self,
        messages: list[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=self._build_messages(messages),
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            finish_reason=choice.finish_reason or "",
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self.config.model,
            messages=self._build_messages(messages),
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class LLMHub:
    """
    LLM Hub — 多模型管理中心。

    功能：
    - 模型注册与发现
    - 智能路由（按任务类型选择最优模型）
    - Fallback 链
    - 统一调用接口
    """

    def __init__(self):
        self._clients: dict[str, BaseLLMClient] = {}
        self._default: Optional[str] = None

    def register(self, name: str, client: BaseLLMClient, default: bool = False):
        self._clients[name] = client
        if default or self._default is None:
            self._default = name

    def get(self, name: Optional[str] = None) -> BaseLLMClient:
        client = self._clients.get(name or self._default or "")
        if client is None:
            raise ValueError(f"未找到 LLM 客户端: {name}")
        return client

    @property
    def default_client(self) -> BaseLLMClient:
        return self.get(self._default)

    @property
    def available_models(self) -> list[str]:
        return list(self._clients.keys())

    @classmethod
    def from_configs(cls, configs: list[tuple[str, LLMConfig, bool]]) -> "LLMHub":
        """从配置列表批量创建 Hub"""
        hub = cls()
        for name, config, is_default in configs:
            hub.register(name, OpenAICompatibleClient(config), default=is_default)
        return hub


# ===== 预设配置工厂 =====

def create_deepseek_client(api_key: str, model: str = "deepseek-chat") -> OpenAICompatibleClient:
    """DeepSeek V4 客户端 (API 兼容 OpenAI)"""
    return OpenAICompatibleClient(
        LLMConfig(
            provider=ModelProvider.DEEPSEEK,
            model=model,
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0.3,
            max_tokens=4096,
        )
    )


def create_tongyi_client(api_key: str) -> OpenAICompatibleClient:
    """Tongyi Qwen3 客户端"""
    return OpenAICompatibleClient(
        LLMConfig(
            provider=ModelProvider.TONGYI,
            model="qwen-plus",
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.3,
            max_tokens=4096,
        )
    )


class CachedLLMClient(BaseLLMClient):
    """LLM 客户端缓存装饰器 — 透明地为任何 LLM 客户端添加响应缓存"""

    def __init__(self, client: BaseLLMClient, cache=None):
        self._client = client
        from linkbridge_core.cache import LLMCache
        self._cache = cache or LLMCache(max_size=500)

    async def chat(
        self,
        messages: list[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        temp = temperature or getattr(self._client, "config", None)
        temp_val = temp.temperature if hasattr(temp, "temperature") else (temperature or 0.3)
        model = getattr(self._client, "config", None)
        model_name = model.model if hasattr(model, "model") else "unknown"

        raw_msgs = [{"role": m.role, "content": m.content} for m in messages]
        cached = self._cache.get(raw_msgs, model_name, temp_val)
        if cached:
            return LLMResponse(content=cached, model=model_name)

        response = await self._client.chat(messages, temperature, max_tokens)
        self._cache.set(raw_msgs, model_name, temp_val, response.content)
        return response

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        # 流式输出不缓存（需累积完整响应才能缓存）
        async for chunk in self._client.chat_stream(messages, temperature, max_tokens):
            yield chunk

    @property
    def cache_stats(self) -> dict:
        return self._cache.stats


def create_zhipu_client(api_key: str) -> OpenAICompatibleClient:
    """Zhipu GLM-4 客户端"""
    return OpenAICompatibleClient(
        LLMConfig(
            provider=ModelProvider.ZHIPU,
            model="glm-4-flash",
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4",
            temperature=0.3,
            max_tokens=4096,
        )
    )
