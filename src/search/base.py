from dataclasses import dataclass, field

from sqlalchemy import Select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import Chunk, Document


@dataclass
class Filters:
    """權限 / 條件篩選，直接進 SQL WHERE（不在應用層事後過濾）。"""
    workspace_id: str = 'default'
    sources: list[str] | None = None
    metadata: dict | None = None            # JSONB 包含（@>）
    document_ids: list[str] | None = None

    def apply(self, stmt: Select) -> Select:
        stmt = stmt.where(Document.workspace_id == self.workspace_id)
        if self.sources:
            stmt = stmt.where(Document.source.in_(self.sources))
        if self.metadata:
            stmt = stmt.where(Document.metadata_.contains(self.metadata))
        if self.document_ids:
            stmt = stmt.where(Chunk.document_id.in_(self.document_ids))
        return stmt


@dataclass
class Hit:
    chunk_id: str
    document_id: str
    title: str
    content: str
    score: float
    scores: dict = field(default_factory=dict)


_pgvector_version: tuple[int, ...] | None = None


async def pgvector_version(session: AsyncSession) -> tuple[int, ...]:
    global _pgvector_version
    if _pgvector_version is None:
        v = (await session.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))).scalar()
        _pgvector_version = tuple(int(x) for x in (v or '0').split('.') if x.isdigit())
    return _pgvector_version
