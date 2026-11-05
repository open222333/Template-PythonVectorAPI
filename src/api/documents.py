from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.deps import get_provider
from src.db import get_session
from src.embedding import EmbeddingError, EmbeddingProvider
from src.ingest.pipeline import ingest_document
from src.models import Chunk, Document
from src.schemas import ChunkOut, DocumentIn, DocumentList, DocumentOut

router = APIRouter(prefix='/documents', tags=['documents'])


def _out(doc: Document, chunk_count: int, with_chunks: bool = False) -> DocumentOut:
    return DocumentOut(
        id=doc.id, title=doc.title, source=doc.source, workspace_id=doc.workspace_id,
        metadata=doc.metadata_, chunk_count=chunk_count, created_at=doc.created_at,
        chunks=[ChunkOut(id=c.id, seq=c.seq, content=c.content) for c in doc.chunks] if with_chunks else None,
    )


@router.post('', response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def create_document(body: DocumentIn, session: AsyncSession = Depends(get_session),
                          provider: EmbeddingProvider = Depends(get_provider)):
    try:
        doc = await ingest_document(session, provider, content=body.content, title=body.title,
                                    source=body.source, workspace_id=body.workspace_id, metadata=body.metadata)
    except EmbeddingError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return _out(doc, len(doc.chunks))


@router.get('', response_model=DocumentList)
async def list_documents(workspace_id: str = 'default', source: str | None = None,
                         limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0),
                         session: AsyncSession = Depends(get_session)):
    cond = [Document.workspace_id == workspace_id] + ([Document.source == source] if source else [])
    total = (await session.execute(select(func.count()).select_from(Document).where(*cond))).scalar_one()
    counts = select(Chunk.document_id, func.count().label('n')).group_by(Chunk.document_id).subquery()
    rows = (await session.execute(
        select(Document, func.coalesce(counts.c.n, 0))
        .outerjoin(counts, counts.c.document_id == Document.id)
        .where(*cond).order_by(Document.created_at.desc()).limit(limit).offset(offset)
    )).all()
    return DocumentList(items=[_out(d, n) for d, n in rows], total=total)


@router.get('/{doc_id}', response_model=DocumentOut)
async def get_document(doc_id: str, session: AsyncSession = Depends(get_session)):
    doc = (await session.execute(
        select(Document).options(selectinload(Document.chunks)).where(Document.id == doc_id)
    )).scalar_one_or_none()
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, 'Document not found')
    return _out(doc, len(doc.chunks), with_chunks=True)


@router.delete('/{doc_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: str, session: AsyncSession = Depends(get_session)):
    """chunks 與 embeddings 由 ON DELETE CASCADE 在同一交易刪除。"""
    result = await session.execute(delete(Document).where(Document.id == doc_id))
    await session.commit()
    if result.rowcount == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, 'Document not found')
