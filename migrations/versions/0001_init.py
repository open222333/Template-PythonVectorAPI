"""init：extension、documents / chunks / embeddings、HNSW 與 trigram 索引

向量維度取自建立時的 EMBEDDING_DIM；之後要改維度請新增 migration。

Revision ID: 0001
Revises:
"""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

from src import settings

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')

    op.create_table(
        'documents',
        sa.Column('id', sa.String(32), primary_key=True),
        sa.Column('workspace_id', sa.String(64), nullable=False, server_default='default'),
        sa.Column('title', sa.String(500), nullable=False, server_default=''),
        sa.Column('source', sa.String(100), nullable=False, server_default='manual'),
        sa.Column('metadata', JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column('is_demo', sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_documents_workspace_id', 'documents', ['workspace_id'])
    op.create_index('ix_documents_source', 'documents', ['source'])
    op.create_index('ix_documents_metadata', 'documents', ['metadata'], postgresql_using='gin')

    op.create_table(
        'chunks',
        sa.Column('id', sa.String(32), primary_key=True),
        sa.Column('document_id', sa.String(32), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('seq', sa.Integer, nullable=False),
        sa.Column('content', sa.Text, nullable=False),
    )
    op.create_index('ix_chunks_document_id', 'chunks', ['document_id'])
    op.execute('CREATE INDEX ix_chunks_content_trgm ON chunks USING gin (content gin_trgm_ops)')

    op.create_table(
        'embeddings',
        sa.Column('chunk_id', sa.String(32), sa.ForeignKey('chunks.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('model_name', sa.String(200), primary_key=True),
        sa.Column('dim', sa.Integer, nullable=False),
        sa.Column('embedding', Vector(settings.EMBEDDING_DIM), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.execute('CREATE INDEX ix_embeddings_hnsw ON embeddings USING hnsw (embedding vector_cosine_ops) '
               'WITH (m = 16, ef_construction = 64)')


def downgrade() -> None:
    op.drop_table('embeddings')
    op.drop_table('chunks')
    op.drop_table('documents')
