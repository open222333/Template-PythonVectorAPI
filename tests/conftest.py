"""整合測試需要 PostgreSQL + pgvector。

TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/vectorapi_test pytest
未設定時只跑單元測試；測試會清空該資料庫的 public schema。
"""
import os

TEST_DB = os.environ.get('TEST_DATABASE_URL', '')
os.environ['EMBEDDING_PROVIDER'] = 'hash'
os.environ['EMBEDDING_DIM'] = '64'
os.environ.setdefault('CHUNK_SIZE', '120')
os.environ.setdefault('CHUNK_OVERLAP', '20')
if TEST_DB:
    os.environ['DATABASE_URL'] = TEST_DB

import pytest  # noqa: E402


@pytest.fixture(scope='session')
async def migrated():
    if not TEST_DB:
        pytest.skip('未設定 TEST_DATABASE_URL，略過資料庫整合測試')
    import asyncio

    from alembic import command
    from alembic.config import Config
    from sqlalchemy import text

    from src.db import engine
    async with engine.begin() as conn:
        await conn.execute(text('DROP SCHEMA public CASCADE'))
        await conn.execute(text('CREATE SCHEMA public'))
    cfg = Config(os.path.join(os.path.dirname(__file__), '..', 'alembic.ini'))
    await asyncio.to_thread(command.upgrade, cfg, 'head')
    yield


@pytest.fixture
async def client(migrated):
    import httpx
    from sqlalchemy import text

    from src.db import engine
    from src.embedding import build_provider
    from src.main import app
    async with engine.begin() as conn:
        await conn.execute(text('TRUNCATE documents CASCADE'))
    app.state.provider = build_provider()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as c:
        yield c
