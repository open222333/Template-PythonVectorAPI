"""Hybrid Search：向量與全文各取前 N 名，以 RRF（Reciprocal Rank Fusion）合併。

RRF score = Σ 1 / (k + rank)，k 預設 60；只看排名不看原始分數，兩種分數尺度不同也能合併。
"""
from sqlalchemy.ext.asyncio import AsyncSession

from src import settings
from src.embedding import EmbeddingProvider
from src.search.base import Filters, Hit
from src.search.fulltext import fulltext_search
from src.search.vector import vector_search

MODES = ('vector', 'fulltext', 'hybrid')


def rrf_merge(result_lists: dict[str, list[Hit]], k: int, limit: int) -> list[Hit]:
    merged: dict[str, Hit] = {}
    for name, hits in result_lists.items():
        for rank, h in enumerate(hits, start=1):
            m = merged.setdefault(h.chunk_id, Hit(h.chunk_id, h.document_id, h.title, h.content, 0.0, {}))
            m.score += 1.0 / (k + rank)
            m.scores[f'{name}_rank'] = rank
            m.scores.update(h.scores)
    out = sorted(merged.values(), key=lambda h: h.score, reverse=True)[:limit]
    for h in out:
        h.score = round(h.score, 6)
    return out


async def search(session: AsyncSession, provider: EmbeddingProvider, query: str,
                 mode: str = 'hybrid', filters: Filters | None = None, limit: int = 5) -> list[Hit]:
    if mode not in MODES:
        raise ValueError(f'mode 必須是 {", ".join(MODES)}')
    filters = filters or Filters()
    if mode == 'fulltext':
        return await fulltext_search(session, query, filters, limit)
    qvec = await provider.embed_one(query)
    if mode == 'vector':
        return await vector_search(session, qvec, provider.model_name, filters, limit)
    n = max(limit, settings.SEARCH_CANDIDATES)
    lists = {
        'vector': await vector_search(session, qvec, provider.model_name, filters, n),
        'fulltext': await fulltext_search(session, query, filters, n),
    }
    return rrf_merge(lists, settings.SEARCH_RRF_K, limit)
