import httpx

from src import settings
from src.embedding.base import EmbeddingError, EmbeddingProvider


class OpenAICompatProvider(EmbeddingProvider):
    """OpenAI 及相容 API（POST {base_url}/embeddings）。

    EMBEDDING_BASE_URL 例如 https://api.openai.com/v1；金鑰放環境變數 EMBEDDING_API_KEY。
    """

    name = 'openai_compat'

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 client: httpx.AsyncClient | None = None, **kw):
        super().__init__(**kw)
        self.base_url = (base_url or settings.EMBEDDING_BASE_URL).rstrip('/')
        key = api_key if api_key is not None else settings.EMBEDDING_API_KEY
        headers = {'Authorization': f'Bearer {key}'} if key else {}
        self.client = client or httpx.AsyncClient(timeout=settings.EMBEDDING_TIMEOUT, headers=headers)

    async def _embed_batch(self, texts):
        try:
            resp = await self.client.post(f'{self.base_url}/embeddings',
                                          json={'model': self.model_name, 'input': texts})
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise EmbeddingError(f'Embedding API 請求失敗：{e}') from e
        data = sorted(resp.json()['data'], key=lambda d: d['index'])
        return [d['embedding'] for d in data]

    async def aclose(self):
        await self.client.aclose()
