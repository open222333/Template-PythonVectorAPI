from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src import settings
from src.api.deps import get_provider
from src.db import get_session
from src.embedding import EmbeddingError, EmbeddingProvider
from src.schemas import SearchHit, SearchIn, SearchOut
from src.search import Filters, search

router = APIRouter(tags=['search'])


@router.post('/search', response_model=SearchOut)
async def search_endpoint(body: SearchIn, session: AsyncSession = Depends(get_session),
                          provider: EmbeddingProvider = Depends(get_provider)):
    mode = body.mode or settings.SEARCH_DEFAULT_MODE
    f = body.filters
    filters = Filters(workspace_id=f.workspace_id, sources=f.source, metadata=f.metadata,
                      document_ids=f.document_ids)
    try:
        async with session.begin():     # SET LOCAL 只在交易內有效
            hits = await search(session, provider, body.query, mode, filters, body.limit)
    except EmbeddingError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(e)) from e
    return SearchOut(mode=mode, model=provider.model_name,
                     results=[SearchHit(**h.__dict__) for h in hits])
