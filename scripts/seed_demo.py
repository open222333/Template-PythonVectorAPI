"""寫入 Demo 資料（is_demo = true）。

python -m scripts.seed_demo            # 寫入
python -m scripts.seed_demo --purge    # 清除所有 Demo 資料
"""
import argparse
import asyncio

from sqlalchemy import delete

from src.db import SessionLocal, engine
from src.embedding import build_provider
from src.ingest.pipeline import ingest_document
from src.models import Document

DEMO = [
    ('回答偏好', 'preferences', '使用者偏好使用繁體中文回答，技術名詞保留英文。回答以條列為主，避免冗長。'),
    ('開發環境', 'technical', '主要開發環境為 macOS，使用 Docker Compose 部署服務。資料庫使用 PostgreSQL 17 與 pgvector 0.8。'),
    ('專案：MEMORA', 'projects', 'MEMORA 是跨 AI 共用記憶平台，讓 ChatGPT、Claude、Gemini 在授權下讀寫同一份長期記憶。'),
    ('Backup policy', 'knowledge', 'Database backups run nightly at 03:17 UTC. Keep 7 local copies and 30 remote copies in object storage.'),
]


async def main() -> None:
    parser = argparse.ArgumentParser(prog='python -m scripts.seed_demo')
    parser.add_argument('--purge', action='store_true')
    args = parser.parse_args()
    try:
        async with SessionLocal() as s:
            if args.purge:
                r = await s.execute(delete(Document).where(Document.is_demo.is_(True)))
                await s.commit()
                print(f'已刪除 {r.rowcount} 筆 Demo 文件')
                return
            provider = build_provider()
            try:
                for title, source, content in DEMO:
                    doc = await ingest_document(s, provider, content=content, title=title, source=source,
                                                metadata={'demo': True}, is_demo=True)
                    print(f'{doc.id}  {title}')
            finally:
                await provider.aclose()
    finally:
        await engine.dispose()


if __name__ == '__main__':
    asyncio.run(main())
