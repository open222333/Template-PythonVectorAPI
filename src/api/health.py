from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_provider
from src.db import get_session
from src.embedding import EmbeddingProvider
from src.models import Embedding

router = APIRouter(tags=['health'])

REQUIRED_INDEXES = ('ix_embeddings_hnsw', 'ix_chunks_content_trgm')


@router.get('/health')
async def health(response: Response, deep: bool = False, session: AsyncSession = Depends(get_session),
                 provider: EmbeddingProvider = Depends(get_provider)):
    """deep=true 時額外測試 Embedding provider 連線。"""
    checks: dict = {}
    problems: list[str] = []
    try:
        await session.execute(text('SELECT 1'))
        checks['database'] = 'ok'
        ext = dict((await session.execute(text(
            "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'pg_trgm')"))).all())
        checks['extensions'] = ext
        for name in ('vector', 'pg_trgm'):
            if name not in ext:
                problems.append(f'extension_missing:{name}')
        idx = set((await session.execute(text(
            'SELECT indexname FROM pg_indexes WHERE indexname = ANY(:names)'),
            {'names': list(REQUIRED_INDEXES)})).scalars())
        checks['indexes'] = sorted(idx)
        problems += [f'index_missing:{i}' for i in REQUIRED_INDEXES if i not in idx]
        models = list((await session.execute(select(Embedding.model_name).distinct())).scalars())
        checks['embedding_models_in_db'] = models
        if any(m != provider.model_name for m in models):
            problems.append('embedding_model_mismatch')
    except Exception as e:  # noqa: BLE001
        checks['database'] = f'error: {type(e).__name__}'
        problems.append('database_unavailable')

    checks['embedding'] = {'provider': provider.name, 'model': provider.model_name, 'dim': provider.dim}
    if deep:
        ok = await provider.ping()
        checks['embedding']['reachable'] = ok
        if not ok:
            problems.append('embedding_unreachable')

    if 'database_unavailable' in problems:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {'status': 'ok' if not problems else 'degraded', 'problems': problems, 'checks': checks}
