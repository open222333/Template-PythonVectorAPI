import httpx
import pytest

from src.embedding.base import EmbeddingError
from src.embedding.hash import HashProvider
from src.embedding.ollama import OllamaProvider
from src.embedding.openai_compat import OpenAICompatProvider
from src.ingest.chunker import chunk_text
from src.search.base import Hit
from src.search.hybrid import rrf_merge


def test_chunker_respects_size_and_overlap():
    text = '\n\n'.join(f'第{i}段。' + '內容' * 30 for i in range(5))
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1 and all(len(c) <= 100 for c in chunks)
    assert chunk_text('短文', 100, 20) == ['短文']
    assert chunk_text('  \n\n ', 100) == []
    assert all(len(c) <= 50 for c in chunk_text('x' * 500, 50, 10))


def test_rrf_merge_prefers_items_in_both_lists():
    def h(i):
        return Hit(i, 'd', '', i, 0.0, {})
    merged = rrf_merge({'vector': [h('a'), h('b')], 'fulltext': [h('b'), h('c')]}, k=60, limit=3)
    assert [m.chunk_id for m in merged] == ['b', 'a', 'c']
    assert merged[0].scores == {'vector_rank': 2, 'fulltext_rank': 1}


async def test_hash_provider_dim_and_similarity():
    p = HashProvider(dim=64)
    a, b, c = await p.embed(['pgvector 語意搜尋', 'pgvector 搜尋', 'completely different'])
    dot = lambda x, y: sum(i * j for i, j in zip(x, y))  # noqa: E731
    assert len(a) == 64 and dot(a, b) > dot(a, c)


async def test_ollama_and_openai_providers():
    def handler(request: httpx.Request):
        if request.url.path == '/api/embed':
            return httpx.Response(200, json={'embeddings': [[0.1] * 4, [0.2] * 4]})
        return httpx.Response(200, json={'data': [{'index': 1, 'embedding': [2.0] * 4},
                                                  {'index': 0, 'embedding': [1.0] * 4}]})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    o = OllamaProvider(base_url='http://o', client=client, model_name='m', dim=4)
    assert (await o.embed(['a', 'b']))[1] == [0.2] * 4
    oa = OpenAICompatProvider(base_url='http://x/v1', client=client, model_name='m', dim=4)
    assert (await oa.embed(['a', 'b']))[0] == [1.0] * 4        # 依 index 排序
    bad = OllamaProvider(base_url='http://o', client=client, model_name='m', dim=8)
    with pytest.raises(EmbeddingError):
        await bad.embed(['a'])
