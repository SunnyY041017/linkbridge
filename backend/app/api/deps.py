"""FastAPI 依赖注入"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import decode_access_token
from app.data.akshare_provider import AKShareProvider
from app.data.fallback_provider import AutoDataProvider
from app.agents.llm_setup import get_llm_hub

security_scheme = HTTPBearer(auto_error=False)

_data_provider: AutoDataProvider | None = None


def get_data_provider() -> AutoDataProvider:
    """获取数据源 — 优先 AKShare 真实数据，网络不可用时自动降级为模拟数据"""
    global _data_provider
    if _data_provider is None:
        _data_provider = AutoDataProvider(primary=AKShareProvider())
    return _data_provider


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme),
) -> dict | None:
    """可选的用户认证——开发阶段允许未认证访问"""
    if credentials is None:
        return None
    payload = decode_access_token(credentials.credentials)
    if payload is None:
        return None
    return payload
