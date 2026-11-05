"""集中讀取設定（環境變數 > conf/config.ini > 預設值）。"""
from urllib.parse import quote_plus

from src import get_bool, get_int, get_setting

POSTGRES_HOST = get_setting('DATABASE', 'POSTGRES_HOST', 'db')
POSTGRES_PORT = get_int('DATABASE', 'POSTGRES_PORT', 5432)
POSTGRES_USER = get_setting('DATABASE', 'POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = get_setting('DATABASE', 'POSTGRES_PASSWORD', '')
POSTGRES_DB = get_setting('DATABASE', 'POSTGRES_DB', 'vectorapi')

# 完整連線字串（優先於上面各欄位），例如測試用
DATABASE_URL = get_setting('DATABASE', 'DATABASE_URL', '') or (
    f'postgresql+asyncpg://{quote_plus(POSTGRES_USER)}:{quote_plus(POSTGRES_PASSWORD)}'
    f'@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'
)

EMBEDDING_PROVIDER = get_setting('EMBEDDING', 'EMBEDDING_PROVIDER', 'ollama')   # ollama / openai_compat / hash
EMBEDDING_BASE_URL = get_setting('EMBEDDING', 'EMBEDDING_BASE_URL', 'http://ollama:11434')
EMBEDDING_API_KEY = get_setting('EMBEDDING', 'EMBEDDING_API_KEY', '')
EMBEDDING_MODEL = get_setting('EMBEDDING', 'EMBEDDING_MODEL', 'nomic-embed-text')
EMBEDDING_DIM = get_int('EMBEDDING', 'EMBEDDING_DIM', 768)
EMBEDDING_BATCH_SIZE = get_int('EMBEDDING', 'EMBEDDING_BATCH_SIZE', 32)
EMBEDDING_TIMEOUT = get_int('EMBEDDING', 'EMBEDDING_TIMEOUT', 60)

CHUNK_SIZE = get_int('INGEST', 'CHUNK_SIZE', 800)
CHUNK_OVERLAP = get_int('INGEST', 'CHUNK_OVERLAP', 100)

SEARCH_DEFAULT_MODE = get_setting('SEARCH', 'SEARCH_DEFAULT_MODE', 'hybrid')
SEARCH_CANDIDATES = get_int('SEARCH', 'SEARCH_CANDIDATES', 20)       # hybrid 每種搜尋取前 N 名再合併
SEARCH_RRF_K = get_int('SEARCH', 'SEARCH_RRF_K', 60)
HNSW_EF_SEARCH = get_int('SEARCH', 'HNSW_EF_SEARCH', 100)
HNSW_ITERATIVE_SCAN = get_bool('SEARCH', 'HNSW_ITERATIVE_SCAN', True)  # pgvector 0.8+ 才會啟用

API_KEY = get_setting('API', 'API_KEY', '')     # 留空 = 不驗證（僅限本機 / 內網）
LOG_LEVEL = get_setting('LOG', 'LOG_LEVEL', 'INFO')
