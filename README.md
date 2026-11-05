# Template-PythonVectorAPI

```
語意搜尋 API 起始模板
FastAPI + PostgreSQL/pgvector，內建 Embedding Provider 抽象層（OpenAI 相容 / Ollama 離線）、文件切塊匯入、向量 / 全文 / Hybrid 搜尋，Docker Compose 一鍵啟動
```

適用：RAG 知識庫、AI 記憶系統、文件 / 筆記語意搜尋。模板只負責「存進去、找回來」，生成回答交給使用模板的專案。

測試環境：Python 3.13 / PostgreSQL 16、17 / pgvector 0.6+（`hnsw.iterative_scan` 需 0.8+，舊版自動略過）

## 目錄

- [功能](#功能)
- [目錄結構](#目錄結構)
- [基於此模板建立新專案](#基於此模板建立新專案)
- [設定](#設定)
- [Docker](#docker)
- [本機執行](#本機執行)
- [資料模型](#資料模型)
- [API](#api)
- [搜尋模式](#搜尋模式)
- [Embedding Provider](#embedding-provider)
- [更換 Embedding 模型](#更換-embedding-模型)
- [擴充方式](#擴充方式)
- [注意事項](#注意事項)

## 功能

- **FastAPI**：async API、自動產生 OpenAPI（`/docs`）
- **PostgreSQL + pgvector**：一個 DB 同時存資料與向量，交易一致
- **Embedding Provider 抽象層**：Ollama（離線）/ OpenAI 相容 API / hash（測試用）
- **匯入流程**：文件 → 切塊（段落 → 句子 → 固定長度，含重疊）→ 批次產生向量 → 同一交易寫入
- **三種搜尋**：`vector`、`fulltext`（pg_trgm，中英文皆可）、`hybrid`（RRF 合併）
- **篩選在 SQL 層**：workspace、source、metadata（JSONB）、document_ids
- **模型版本檢查**：啟動與 `/health` 比對 DB 內向量模型與目前設定
- **Alembic migration**：建立 extension、資料表、HNSW / trigram 索引；容器啟動時自動執行
- **選用 API Key**：設定 `API_KEY` 後需帶 `X-API-Key`

## 目錄結構

```
Template-PythonVectorAPI/
├── conf/
│   └── config.ini.default        # 設定檔範本（複製為 config.ini）
├── docker/
│   ├── Dockerfile                # python:3.13-slim，非 root 執行
│   └── entrypoint.sh             # AUTO_MIGRATE=true 時先 alembic upgrade head
├── migrations/                   # Alembic（async）
│   └── versions/0001_init.py     # extension、資料表、索引
├── src/
│   ├── __init__.py               # 讀取設定（get_setting / get_int / get_bool）
│   ├── settings.py               # 設定集中讀取
│   ├── main.py                   # FastAPI app 進入點
│   ├── db.py                     # async engine / session
│   ├── models.py                 # documents、chunks、embeddings
│   ├── schemas.py                # Pydantic request / response
│   ├── api/                      # documents、search、health、deps（API Key）
│   ├── embedding/                # base、ollama、openai_compat、hash
│   ├── ingest/                   # chunker、pipeline
│   └── search/                   # vector、fulltext、hybrid（RRF）
├── scripts/
│   ├── reembed.py                # 更換模型時重新產生向量
│   └── seed_demo.py              # Demo 資料（is_demo = true）
├── tests/                        # 單元測試 + 整合測試（需 PostgreSQL）
├── .env.default                  # 環境變數範本（複製為 .env）
├── alembic.ini
├── docker-compose.yml            # api + db（+ 選用 ollama）
├── requirements.txt
├── requirements-test.txt
└── pytest.ini
```

## 基於此模板建立新專案

```bash
git clone <此模板 URL> your-new-project
cd your-new-project
rm -rf .git && git init          # 不保留模板歷史
cp conf/config.ini.default conf/config.ini
cp .env.default .env             # 修改 POSTGRES_PASSWORD
```

**先決定 Embedding 模型與維度**（`EMBEDDING_MODEL` / `EMBEDDING_DIM`），第一次 migration 會把維度寫進 `vector(n)`。

## 設定

讀取順序：**環境變數 > `conf/config.ini` > 預設值**。

| 變數 | 說明 | 預設 |
|---|---|---|
| `POSTGRES_HOST` / `POSTGRES_PORT` | 資料庫位址 | `db` / `5432` |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | 資料庫帳密（密碼只放 `.env`） | `postgres` / — / `vectorapi` |
| `DATABASE_URL` | 完整連線字串（優先於上面各欄位） | — |
| `EMBEDDING_PROVIDER` | `ollama` / `openai_compat` / `hash` | `ollama` |
| `EMBEDDING_BASE_URL` | Provider 端點 | `http://ollama:11434` |
| `EMBEDDING_API_KEY` | 外部 API 金鑰（只放 `.env`） | — |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | 模型名稱 / 向量維度 | `nomic-embed-text` / `768` |
| `EMBEDDING_BATCH_SIZE` | 每批送出的文字數 | `32` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 切塊字數 / 重疊字數 | `800` / `100` |
| `SEARCH_DEFAULT_MODE` | 預設搜尋模式 | `hybrid` |
| `SEARCH_CANDIDATES` / `SEARCH_RRF_K` | hybrid 候選數 / RRF k | `20` / `60` |
| `HNSW_EF_SEARCH` / `HNSW_ITERATIVE_SCAN` | HNSW 查詢參數 | `100` / `true` |
| `API_KEY` | 設定後需帶 `X-API-Key`（`/health` 除外） | — |
| `AUTO_MIGRATE` | 容器啟動時自動 migration | `true` |

- 預設 Provider 為 Ollama：**不設定外部服務時，資料不會傳出本機**

## Docker

```bash
docker compose up -d --build                         # api + db（啟動時自動 migration）
docker compose --profile ollama up -d                # 加上本地 Embedding 模型
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec api python -m scripts.seed_demo  # 選用：寫入 Demo 資料
```

- API 文件：<http://localhost:8000/docs>
- DB 使用 `pgvector/pgvector:pg17`，資料持久化於 `./data/postgres`
- 埠號只綁定 `127.0.0.1`；對外開放需自行加 TLS（反向代理）並設定 `API_KEY`

## 本機執行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-test.txt
docker compose up -d db                       # 只起資料庫
export POSTGRES_HOST=localhost POSTGRES_PASSWORD=change-me
alembic upgrade head
uvicorn src.main:app --reload                 # http://localhost:8000/docs
```

沒有 Embedding 模型時，可先用 `EMBEDDING_PROVIDER=hash EMBEDDING_DIM=64` 試跑（無語意能力，僅供開發）。

測試：

```bash
pytest                                        # 單元測試（整合測試自動略過）

# 整合測試：需一個可清空的測試資料庫（會 DROP SCHEMA public）
createdb -h localhost -U postgres vectorapi_test
TEST_DATABASE_URL=postgresql+asyncpg://postgres:change-me@localhost:5432/vectorapi_test pytest
```

## 資料模型

```
documents 1 ── * chunks 1 ── * embeddings（每個模型一筆）
```

| 資料表 | 主要欄位 | 說明 |
|---|---|---|
| `documents` | `id`、`workspace_id`、`title`、`source`、`metadata jsonb`、`is_demo`、`created_at`、`updated_at` | 原始文件 |
| `chunks` | `id`、`document_id`、`seq`、`content` | 切塊內容；`content` 有 trigram GIN 索引 |
| `embeddings` | `chunk_id` + `model_name`（PK）、`dim`、`embedding vector(n)`、`created_at` | 向量；HNSW（cosine）索引 |

- 內容與向量分表：換模型時新舊向量可並存，確認完成再刪舊的
- 刪除文件時，chunks 與 embeddings 以 `ON DELETE CASCADE` 在同一交易刪除

## API

```http
POST   /api/v1/documents          # 新增文件（自動切塊 + 產生向量）
GET    /api/v1/documents          # ?workspace_id=&source=&limit=&offset=
GET    /api/v1/documents/{id}     # 含 chunks
DELETE /api/v1/documents/{id}

POST   /api/v1/search
GET    /api/v1/health             # ?deep=true 額外測試 Embedding 連線
```

新增文件：

```bash
curl -X POST localhost:8000/api/v1/documents -H 'Content-Type: application/json' -d '{
  "title": "開發環境",
  "source": "technical",
  "content": "資料庫使用 PostgreSQL 17 與 pgvector 0.8，部署用 Docker Compose。",
  "metadata": {"project": "memora"}
}'
```

搜尋：

```json
{
  "query": "資料庫版本",
  "mode": "hybrid",
  "filters": { "workspace_id": "default", "source": ["technical"], "metadata": { "project": "memora" } },
  "limit": 5
}
```

回應附各來源分數與排名，方便調整：

```json
{
  "mode": "hybrid",
  "model": "nomic-embed-text",
  "results": [
    {
      "chunk_id": "chk_68aaabaef66842b4",
      "document_id": "doc_6d1d585f279b4305",
      "title": "開發環境",
      "content": "資料庫使用 PostgreSQL 17 與 pgvector 0.8，部署用 Docker Compose。",
      "score": 0.032522,
      "scores": { "vector_rank": 1, "vector_similarity": 0.81, "fulltext_rank": 1, "term_match": 1.0, "trigram": 0.25 }
    }
  ]
}
```

錯誤：Embedding 服務失敗 → `502`；內容為空 → `422`；API Key 錯誤 → `401`。

## 搜尋模式

| 模式 | 方式 | 適合 |
|---|---|---|
| `vector` | pgvector 餘弦距離 + HNSW | 意思相近、用詞不同 |
| `fulltext` | 每個詞（空白分隔）`ILIKE` 子字串比對 + pg_trgm `word_similarity` | 型號、版本號、專有名詞（例如 `4090` 不會被當成 `3090`） |
| `hybrid` | 兩者各取前 `SEARCH_CANDIDATES` 名，RRF 合併：`Σ 1 / (k + rank)` | 預設；兼顧語意與精確詞 |

- 中文沒有空白分詞：整段查詢視為一個詞做子字串比對；需要斷詞時可改用 zhparser / pg_jieba 的 `tsvector`
- 小於 3 個字的詞無法使用 trigram 索引（改為循序掃描），資料量大時留意效能
- pgvector 0.8+ 自動啟用 `hnsw.iterative_scan`，避免篩選條件嚴格時結果不足

## Embedding Provider

| Provider | 設定 | 備註 |
|---|---|---|
| `ollama` | `EMBEDDING_BASE_URL=http://ollama:11434` | 離線可用（`POST /api/embed`） |
| `openai_compat` | `EMBEDDING_BASE_URL=https://api.openai.com/v1`、`EMBEDDING_API_KEY` | OpenAI 及相容 API（`POST /embeddings`） |
| `hash` | — | 字元 n-gram 雜湊，**僅供測試 / 開發**，沒有語意能力 |

常見模型維度：`nomic-embed-text` 768、`bge-m3` 1024、`text-embedding-3-small` 1536、`text-embedding-3-large` 3072（超過 2000 維需改用 `halfvec`，見注意事項）。

新增 Provider：繼承 `src/embedding/base.py` 的 `EmbeddingProvider`，實作 `_embed_batch()`，加入 `build_provider()`：

```python
class MyProvider(EmbeddingProvider):
    name = 'my'

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...
```

回傳維度與 `EMBEDDING_DIM` 不符時會拋出 `EmbeddingError`，不會寫入錯誤資料。

## 更換 Embedding 模型

不同模型的向量不能混用。DB 內有其他模型的向量時，啟動 log 會警告，`/api/v1/health` 回報 `embedding_model_mismatch`。

```bash
# 1. 修改 .env / config.ini 的 EMBEDDING_MODEL（維度改變時另見下方）
# 2. 補齊新模型向量（舊向量保留，可隨時退回）
docker compose exec api python -m scripts.reembed
# 3. 確認後刪除舊模型向量（仍有 chunk 缺新向量時會拒絕執行）
docker compose exec api python -m scripts.reembed --purge-old
```

維度改變時，`vector(n)` 欄位無法同時存放兩種維度：新增 migration 修改欄位型別（並重建 HNSW 索引），先清空 `embeddings` 再執行 reembed。

## 擴充方式

| 需求 | 做法 |
|---|---|
| 大量匯入 | 加 Redis + RQ / Celery，將 `ingest/pipeline.py` 改為背景任務 |
| 新資料來源 | 在 `ingest/` 新增 loader（Markdown、PDF、CSV…），輸出純文字給 `ingest_document()` |
| 多租戶 / 權限 | 依登入身分決定 `Filters.workspace_id`，不要讓使用者自由指定 |
| RAG 問答 | 使用專案自行呼叫 LLM，以 `/search` 結果作為 context |
| MCP Server | 另起 process，呼叫本 API 或直接使用 `src.search.search()` |

## 注意事項

- 金鑰只放 `.env` / 環境變數，**不要**寫進 `*.default`
- 不在 log 中輸出完整文件或查詢內容
- `vector` 型別索引上限 2000 維；超過改用 `halfvec`（pgvector 0.7+，上限 4000 維），需同步修改 migration 與 `models.py`
- Demo 資料以 `is_demo` 標記，正式環境執行 `python -m scripts.seed_demo --purge` 清除
