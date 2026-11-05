"""FastAPI 進入點：uvicorn src.main:app"""
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlalchemy import select

from src import settings
from src.api import documents, health, search
from src.api.deps import require_api_key
from src.db import SessionLocal, engine
from src.embedding import build_provider
from src.models import Embedding

logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def _check_model(provider) -> None:
    try:
        async with SessionLocal() as s:
            models = list((await s.execute(select(Embedding.model_name).distinct())).scalars())
    except Exception:  # noqa: BLE001 - DB 尚未就緒時由 /health 回報
        return
    other = [m for m in models if m != provider.model_name]
    if other:
        logger.warning('資料庫中有其他模型的向量 %s，目前設定為 %s；請執行 scripts/reembed.py',
                       other, provider.model_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.provider = build_provider()
    await _check_model(app.state.provider)
    yield
    await app.state.provider.aclose()
    await engine.dispose()


app = FastAPI(title='Vector API', version='0.1.0', lifespan=lifespan)
api_deps = [Depends(require_api_key)]
app.include_router(documents.router, prefix='/api/v1', dependencies=api_deps)
app.include_router(search.router, prefix='/api/v1', dependencies=api_deps)
app.include_router(health.router, prefix='/api/v1')
