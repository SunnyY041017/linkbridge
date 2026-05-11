"""
LLM 响应缓存层 — LRU 内存缓存 + JSON 文件持久化。

设计理念：
- 相同问题 + 相同模型参数 → 缓存命中，不消耗 API 额度
- 股票数据日级变化快，缓存 key 含日期 → 同一天内重复查询命中
- 降级时缓存仍可用，不影响服务
"""

import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional


def _make_cache_key(
    messages: list[dict],
    model: str,
    temperature: float,
    date_prefix: str = "",
) -> str:
    """基于消息内容、模型和参数生成唯一缓存键"""
    content = json.dumps(
        {"msgs": messages, "model": model, "temp": temperature},
        ensure_ascii=False,
        sort_keys=True,
    )
    h = hashlib.sha256(content.encode()).hexdigest()[:16]
    if date_prefix:
        return f"{date_prefix}:{h}"
    return h


class LLMCache:
    """
    LRU 缓存 + JSON 文件持久化。

    用法:
        cache = LLMCache(max_size=500, persist_path="llm_cache.json")
        cached = cache.get(messages, model, temp)
        if cached:
            return cached
        response = await llm.chat(messages)
        cache.set(messages, model, temp, response)
    """

    def __init__(self, max_size: int = 500, persist_path: Optional[str] = None):
        self.max_size = max_size
        self._store: OrderedDict[str, dict] = OrderedDict()
        self.persist_path = Path(persist_path) if persist_path else None
        self.hits = 0
        self.misses = 0

        if self.persist_path and self.persist_path.exists():
            self._load()

    def _load(self):
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 只加载最近 max_size 条
            items = list(data.items())[-self.max_size:]
            self._store = OrderedDict(items)
        except (json.JSONDecodeError, OSError):
            self._store = OrderedDict()

    def _save(self):
        if not self.persist_path:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(dict(self._store), f, ensure_ascii=False, indent=2)
        except OSError:
            pass  # 缓存写入失败不影响主流程

    def _date_prefix(self) -> str:
        from datetime import date
        return date.today().isoformat()

    def get(
        self,
        messages: list[dict],
        model: str = "deepseek-chat",
        temperature: float = 0.3,
    ) -> Optional[str]:
        """获取缓存的 LLM 响应。返回 None 表示未命中。"""
        key = _make_cache_key(messages, model, temperature, self._date_prefix())
        if key in self._store:
            self._store.move_to_end(key)
            self.hits += 1
            return self._store[key]["response"]
        self.misses += 1
        return None

    def set(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        response: str,
    ):
        """缓存 LLM 响应"""
        key = _make_cache_key(messages, model, temperature, self._date_prefix())
        self._store[key] = {
            "response": response,
            "model": model,
            "cached_at": time.time(),
        }
        # LRU 淘汰
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)
        # 每 10 次写入持久化一次（减少 I/O）
        if len(self._store) % 10 == 0:
            self._save()

    def flush(self):
        """强制持久化"""
        self._save()

    def clear(self):
        """清空缓存"""
        self._store.clear()
        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink(missing_ok=True)

    @property
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._store),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total * 100, 1) if total > 0 else 0,
        }
