DOCS = [
    {'title': '偏好', 'source': 'preferences', 'content': '使用者偏好使用繁體中文回答，技術名詞保留英文。'},
    {'title': '環境', 'source': 'technical', 'content': '資料庫使用 PostgreSQL 17 與 pgvector 0.8，部署用 Docker Compose。'},
    {'title': 'GPU', 'source': 'technical', 'content': 'Training server uses RTX 4090 and RTX 3090 GPUs.',
     'metadata': {'team': 'ml'}},
    {'title': '其他', 'source': 'notes', 'content': '週末去爬山，天氣很好。', 'workspace_id': 'other'},
]


async def _seed(client):
    ids = []
    for d in DOCS:
        r = await client.post('/api/v1/documents', json=d)
        assert r.status_code == 201, r.text
        ids.append(r.json()['id'])
    return ids


async def test_document_crud(client):
    ids = await _seed(client)
    r = await client.get('/api/v1/documents')
    assert r.json()['total'] == 3                       # 'other' workspace 不列入
    r = await client.get(f'/api/v1/documents/{ids[0]}')
    assert r.json()['chunks'][0]['content'].startswith('使用者偏好')
    assert (await client.delete(f'/api/v1/documents/{ids[0]}')).status_code == 204
    assert (await client.get(f'/api/v1/documents/{ids[0]}')).status_code == 404
    assert (await client.delete(f'/api/v1/documents/{ids[0]}')).status_code == 404


async def test_long_document_chunked(client):
    content = '\n\n'.join(f'第 {i} 段：' + '向量搜尋與全文搜尋。' * 8 for i in range(6))
    r = await client.post('/api/v1/documents', json={'content': content})
    assert r.json()['chunk_count'] > 1


async def test_fulltext_exact_model_number(client):
    await _seed(client)
    r = await client.post('/api/v1/search', json={'query': '4090', 'mode': 'fulltext'})
    hits = r.json()['results']
    assert hits and hits[0]['title'] == 'GPU'
    r = await client.post('/api/v1/search', json={'query': '中文', 'mode': 'fulltext'})
    assert r.json()['results'][0]['title'] == '偏好'


async def test_vector_and_hybrid_with_filters(client):
    await _seed(client)
    r = await client.post('/api/v1/search', json={'query': 'pgvector PostgreSQL', 'mode': 'vector', 'limit': 3})
    assert r.json()['results'][0]['title'] == '環境'
    r = await client.post('/api/v1/search', json={'query': 'PostgreSQL pgvector', 'limit': 3})
    body = r.json()
    assert body['mode'] == 'hybrid' and body['results'][0]['title'] == '環境'
    assert {'vector_rank', 'fulltext_rank'} <= set(body['results'][0]['scores'])
    # 篩選：source、metadata、workspace 隔離
    r = await client.post('/api/v1/search', json={'query': 'RTX', 'filters': {'source': ['preferences']}})
    assert all(h['title'] == '偏好' for h in r.json()['results'])
    r = await client.post('/api/v1/search', json={'query': 'GPU', 'filters': {'metadata': {'team': 'ml'}}})
    assert [h['title'] for h in r.json()['results']] == ['GPU']
    r = await client.post('/api/v1/search', json={'query': '爬山', 'mode': 'fulltext'})
    assert r.json()['results'] == []
    r = await client.post('/api/v1/search', json={'query': '爬山', 'mode': 'fulltext',
                                                 'filters': {'workspace_id': 'other'}})
    assert r.json()['results'][0]['title'] == '其他'


async def test_health(client):
    await _seed(client)
    r = await client.get('/api/v1/health?deep=true')
    body = r.json()
    assert r.status_code == 200 and body['status'] == 'ok', body
    assert body['checks']['embedding_models_in_db'] == ['hash-ngram-v1']


async def test_api_key(client, monkeypatch):
    from src import settings
    monkeypatch.setattr(settings, 'API_KEY', 'k')
    assert (await client.get('/api/v1/documents')).status_code == 401
    assert (await client.get('/api/v1/documents', headers={'X-API-Key': 'k'})).status_code == 200
    assert (await client.get('/api/v1/health')).status_code == 200
