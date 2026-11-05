"""資料模型：documents 1─* chunks 1─* embeddings（每個模型一筆）。"""
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from src import settings


def new_id(prefix: str) -> str:
    return f'{prefix}_{uuid.uuid4().hex[:16]}'


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = 'documents'

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id('doc'))
    workspace_id: Mapped[str] = mapped_column(String(64), index=True, default='default')
    title: Mapped[str] = mapped_column(String(500), default='')
    source: Mapped[str] = mapped_column(String(100), index=True, default='manual')
    metadata_: Mapped[dict] = mapped_column('metadata', JSONB, default=dict)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),
                                                 onupdate=func.now())

    chunks: Mapped[list['Chunk']] = relationship(back_populates='document', cascade='all, delete-orphan',
                                                 passive_deletes=True, order_by='Chunk.seq')


class Chunk(Base):
    __tablename__ = 'chunks'

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id('chk'))
    document_id: Mapped[str] = mapped_column(ForeignKey('documents.id', ondelete='CASCADE'), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates='chunks')
    embeddings: Mapped[list['Embedding']] = relationship(cascade='all, delete-orphan', passive_deletes=True)


class Embedding(Base):
    __tablename__ = 'embeddings'

    chunk_id: Mapped[str] = mapped_column(ForeignKey('chunks.id', ondelete='CASCADE'), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(200), primary_key=True)
    dim: Mapped[int] = mapped_column(Integer)
    embedding = mapped_column(Vector(settings.EMBEDDING_DIM))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
