"""LLM Hub 初始化 — 从配置文件创建 LLM 客户端，默认启用响应缓存"""
import os

from linkbridge_core.llm_hub import (
    LLMHub, CachedLLMClient,
    create_deepseek_client,
    create_tongyi_client,
    create_zhipu_client,
)
from linkbridge_core.cache import LLMCache

from app.config import settings


_hub: LLMHub | None = None
_cache: LLMCache | None = None


def get_llm_cache() -> LLMCache:
    global _cache
    if _cache is not None:
        return _cache
    cache_path = os.path.join(os.path.dirname(__file__), "..", "..", "llm_cache.json")
    _cache = LLMCache(max_size=500, persist_path=cache_path)
    return _cache


def get_llm_hub() -> LLMHub:
    global _hub
    if _hub is not None:
        return _hub

    _hub = LLMHub()
    cache = get_llm_cache()

    if settings.deepseek_api_key:
        client = create_deepseek_client(settings.deepseek_api_key)
        _hub.register(
            "deepseek",
            CachedLLMClient(client, cache=cache),
            default=True,
        )

    if settings.tongyi_api_key:
        _hub.register(
            "tongyi",
            create_tongyi_client(settings.tongyi_api_key),
        )

    if settings.zhipu_api_key:
        _hub.register(
            "zhipu",
            create_zhipu_client(settings.zhipu_api_key),
        )

    if not _hub.available_models:
        raise RuntimeError(
            "至少配置一个 LLM API Key（DEEPSEEK_API_KEY / TONGYI_API_KEY / ZHIPU_API_KEY）"
        )

    return _hub
