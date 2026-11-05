"""全文搜尋：pg_trgm（不需分詞，中英文皆可）。

- 每個詞（空白分隔）做 ILIKE 子字串比對 → 精確找到型號、版本號、專有名詞
- 加上 word_similarity 做模糊比對（拼字差異）
- 中文斷詞品質要求高時，可改用 zhparser / pg_jieba 的 tsvector
"""
from sqlalchemy import case, func, literal, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import Chunk, Document
from src.search.base import Filters, Hit


def _escape_like(term: str) -> str:
    return term.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')


async def fulltext_search(session: AsyncSession, query: str, filters: Filters, limit: int) -> list[Hit]:
    terms = [t for t in query.split() if t]
    if not terms:
        return []
    matches = [Chunk.content.ilike(f'%{_escape_like(t)}%', escape='\\') for t in terms]
    term_score = sum((case((m, 1.0), else_=0.0) for m in matches), literal(0.0)) / len(terms)
    fuzzy = func.word_similarity(query, Chunk.content)
    score = (term_score + fuzzy).label('score')
    stmt = (
        select(Chunk.id, Chunk.document_id, Document.title, Chunk.content, score,
               term_score.label('term_score'), fuzzy.label('fuzzy'))
        .join(Document, Document.id == Chunk.document_id)
        .where(or_(*matches, literal(query).op('<%')(Chunk.content)))
        .order_by(score.desc())
        .limit(limit)
    )
    rows = (await session.execute(filters.apply(stmt))).all()
    return [Hit(r.id, r.document_id, r.title, r.content, float(r.score),
                {'term_match': round(float(r.term_score), 4), 'trigram': round(float(r.fuzzy), 4)})
            for r in rows]
