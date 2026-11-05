"""新增文件：切塊 → 產生向量 → 寫入（同一交易，失敗時全部 rollback）。"""
from sqlalchemy.ext.asyncio import AsyncSession

from src import settings
from src.embedding import EmbeddingProvider
from src.ingest.chunker import chunk_text
from src.models import Chunk, Document, Embedding


async def ingest_document(session: AsyncSession, provider: EmbeddingProvider, *, content: str,
                          title: str = '', source: str = 'manual', workspace_id: str = 'default',
                          metadata: dict | None = None, is_demo: bool = False) -> Document:
    texts = chunk_text(content, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    if not texts:
        raise ValueError('內容為空')
    vectors = await provider.embed(texts)       # 先產生向量，避免寫到一半失敗

    doc = Document(title=title, source=source, workspace_id=workspace_id,
                   metadata_=metadata or {}, is_demo=is_demo)
    for seq, (text, vec) in enumerate(zip(texts, vectors)):
        chunk = Chunk(seq=seq, content=text)
        chunk.embeddings.append(Embedding(model_name=provider.model_name, dim=provider.dim, embedding=vec))
        doc.chunks.append(chunk)
    session.add(doc)
    await session.commit()
    return doc
