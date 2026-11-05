"""以目前設定的模型重新產生向量。

python -m scripts.reembed                  # 補齊缺少目前模型向量的 chunk
python -m scripts.reembed --purge-old      # 刪除其他模型的向量（確認新向量完成後再執行）
"""
import argparse
import asyncio
import logging

from sqlalchemy import delete, func, select

from src import settings
from src.db import SessionLocal, engine
from src.embedding import build_provider
from src.models import Chunk, Embedding

logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('reembed')


async def reembed(batch_size: int) -> int:
    provider = build_provider()
    done = 0
    try:
        while True:
            async with SessionLocal() as s:
                has_current = select(Embedding.chunk_id).where(Embedding.model_name == provider.model_name)
                chunks = (await s.execute(
                    select(Chunk).where(Chunk.id.not_in(has_current)).order_by(Chunk.id).limit(batch_size)
                )).scalars().all()
                if not chunks:
                    return done
                vectors = await provider.embed([c.content for c in chunks])
                s.add_all(Embedding(chunk_id=c.id, model_name=provider.model_name, dim=provider.dim, embedding=v)
                          for c, v in zip(chunks, vectors))
                await s.commit()
                done += len(chunks)
                logger.info('已處理 %d 個 chunk', done)
    finally:
        await provider.aclose()


async def purge_old() -> int:
    model = build_provider().model_name
    async with SessionLocal() as s:
        missing = (await s.execute(
            select(func.count()).select_from(Chunk).where(Chunk.id.not_in(
                select(Embedding.chunk_id).where(Embedding.model_name == model)))
        )).scalar_one()
        if missing:
            raise SystemExit(f'仍有 {missing} 個 chunk 缺少 {model} 的向量，請先執行 reembed')
        r = await s.execute(delete(Embedding).where(Embedding.model_name != model))
        await s.commit()
        return r.rowcount


async def main() -> None:
    parser = argparse.ArgumentParser(prog='python -m scripts.reembed')
    parser.add_argument('--batch-size', type=int, default=settings.EMBEDDING_BATCH_SIZE)
    parser.add_argument('--purge-old', action='store_true')
    args = parser.parse_args()
    try:
        if args.purge_old:
            logger.info('已刪除 %d 筆舊模型向量', await purge_old())
        else:
            logger.info('完成，共 %d 個 chunk', await reembed(args.batch_size))
    finally:
        await engine.dispose()


if __name__ == '__main__':
    asyncio.run(main())
