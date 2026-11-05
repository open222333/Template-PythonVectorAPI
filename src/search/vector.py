from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src import settings
from src.models import Chunk, Document, Embedding
from src.search.base import Filters, Hit, pgvector_version


async def vector_search(session: AsyncSession, query_vec: list[float], model_name: str,
                        filters: Filters, limit: int) -> list[Hit]:
    """餘弦距離排序（HNSW 索引）。similarity = 1 - distance。"""
    await session.execute(text(f'SET LOCAL hnsw.ef_search = {int(settings.HNSW_EF_SEARCH)}'))
    if settings.HNSW_ITERATIVE_SCAN and await pgvector_version(session) >= (0, 8):
        # 加上 WHERE 篩選後結果不足 LIMIT 時，繼續往下掃描
        await session.execute(text('SET LOCAL hnsw.iterative_scan = relaxed_order'))

    distance = Embedding.embedding.cosine_distance(query_vec).label('distance')
    stmt = (
        select(Chunk.id, Chunk.document_id, Document.title, Chunk.content, distance)
        .join(Embedding, Embedding.chunk_id == Chunk.id)
        .join(Document, Document.id == Chunk.document_id)
        .where(Embedding.model_name == model_name)
        .order_by(distance)
        .limit(limit)
    )
    rows = (await session.execute(filters.apply(stmt))).all()
    return [Hit(r.id, r.document_id, r.title, r.content, 1 - float(r.distance),
                {'vector_similarity': round(1 - float(r.distance), 4)}) for r in rows]
