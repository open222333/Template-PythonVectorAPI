from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    content: str = Field(min_length=1, description='文件全文，會自動切塊並產生向量')
    title: str = ''
    source: str = 'manual'
    workspace_id: str = 'default'
    metadata: dict = Field(default_factory=dict)


class ChunkOut(BaseModel):
    id: str
    seq: int
    content: str


class DocumentOut(BaseModel):
    id: str
    title: str
    source: str
    workspace_id: str
    metadata: dict
    chunk_count: int
    created_at: datetime | None = None
    chunks: list[ChunkOut] | None = None


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int


class SearchFilters(BaseModel):
    workspace_id: str = 'default'
    source: list[str] | None = None
    metadata: dict | None = None
    document_ids: list[str] | None = None


class SearchIn(BaseModel):
    query: str = Field(min_length=1)
    mode: Literal['vector', 'fulltext', 'hybrid'] | None = None
    filters: SearchFilters = Field(default_factory=SearchFilters)
    limit: int = Field(5, ge=1, le=100)


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    content: str
    score: float
    scores: dict


class SearchOut(BaseModel):
    mode: str
    model: str
    results: list[SearchHit]
