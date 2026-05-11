"""
安全中间件 — 请求限流、输入过滤、速率保护。
"""
import re
import time
import hashlib
from collections import defaultdict
from typing import Callable

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


# ===== 输入过滤 =====

# 检测潜在的注入模式（SQL 注入、命令注入、XSS）
_INJECTION_PATTERNS = [
    re.compile(r"(?i)(\bSELECT\b.*\bFROM\b|\bDROP\b\s+\bTABLE\b|\bINSERT\b\s+\bINTO\b|\bDELETE\b\s+\bFROM\b|\bUPDATE\b.*\bSET\b)"),
    re.compile(r"(?i)(<script|javascript:|on\w+\s*=)", re.IGNORECASE),
    re.compile(r"(?i)(\bexec\b\s*\(|\bsystem\b\s*\(|\bexec\b\s*\(|\bpassthru\b)", re.IGNORECASE),
    re.compile(r"(?i)(\.\./|\.\.\\)"),
    re.compile(r"(?i)(;--|\buname\b|\bid\b\s*=\s*\d+\s+or\s+1)", re.IGNORECASE),
]

MAX_INPUT_LENGTH = 4000  # 最大输入长度


def sanitize_input(text: str) -> tuple[str, bool]:
    """
    输入清理和注入检测。

    Returns:
        (sanitized_text, is_safe)
    """
    if not text:
        return text, True

    # 长度限制
    if len(text) > MAX_INPUT_LENGTH:
        text = text[:MAX_INPUT_LENGTH]

    # 注入检测
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return "", False

    # 移除不可打印字符（保留中文、英文、数字、标点）
    cleaned = re.sub(r'[^一-鿿　-〿＀-￯a-zA-Z0-9\s\.\,\;\:\!\?\-\+\=\%\(\)\[\]\{\}\@\#\$\&\*\"\'\~\`\|\/\\\_\<\>\^]', '', text)

    return cleaned.strip(), True


# ===== 限流 =====

class RateLimiter:
    """
    简单的滑动窗口限流器。

    默认：每个 IP 每分钟最多 30 次请求。
    """

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _clean_old(self, key: str, now: float):
        cutoff = now - self.window
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        self._clean_old(key, now)
        if len(self._requests[key]) >= self.max_requests:
            return False
        self._requests[key].append(now)
        return True

    def remaining(self, key: str) -> int:
        self._clean_old(key, time.time())
        return max(0, self.max_requests - len(self._requests[key]))

    def reset_time(self, key: str) -> float:
        now = time.time()
        self._clean_old(key, now)
        if not self._requests[key]:
            return 0
        return self.window - (now - min(self._requests[key]))


_limiter = RateLimiter(max_requests=30, window_seconds=60)


def get_client_key(request: Request) -> str:
    """基于 IP + User-Agent 生成客户端标识"""
    ip = request.client.host if request.client else "unknown"
    ua = request.headers.get("user-agent", "")
    raw = f"{ip}:{ua[:50]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


class SecurityMiddleware(BaseHTTPMiddleware):
    """FastAPI 安全中间件"""

    async def dispatch(self, request: Request, call_next: Callable):
        # 只对 API 路由限流
        if request.url.path.startswith("/api/"):
            client_key = get_client_key(request)
            if not _limiter.is_allowed(client_key):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "请求过于频繁，请稍后再试", "retry_after": round(_limiter.reset_time(client_key))},
                )

        response = await call_next(request)
        return response


def get_rate_limiter() -> RateLimiter:
    return _limiter
