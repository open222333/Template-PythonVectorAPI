import httpx

from src import settings
from src.embedding.base import EmbeddingError, EmbeddingProvider


class OllamaProvider(EmbeddingProvider):
    """Ollama 本地模型（POST /api/embed）。資料不離開本機。"""

    name = 'ollama'

    def __init__(self, base_url: str | None = None, client: httpx.AsyncClient | None = None, **kw):
        super().__init__(**kw)
        self.base_url = (base_url or settings.EMBEDDING_BASE_URL).rstrip('/')
        self.client = client or httpx.AsyncClient(timeout=settings.EMBEDDING_TIMEOUT)

    async def _embed_batch(self, texts):
        try:
            resp = await self.client.post(f'{self.base_url}/api/embed',
                                          json={'model': self.model_name, 'input': texts})
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise EmbeddingError(f'Ollama 請求失敗：{e}') from e
        return resp.json()['embeddings']

    async def aclose(self):
        await self.client.aclose()
