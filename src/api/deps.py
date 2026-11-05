import secrets

from fastapi import Header, HTTPException, Request, status

from src import settings
from src.embedding import EmbeddingProvider


def get_provider(request: Request) -> EmbeddingProvider:
    return request.app.state.provider


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """設定 API_KEY 時要求 X-API-Key header；未設定則不驗證（僅限本機 / 內網）。"""
    if settings.API_KEY and not (x_api_key and secrets.compare_digest(x_api_key, settings.API_KEY)):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, 'Invalid API key')
