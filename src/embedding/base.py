from abc import ABC, abstractmethod

from src import settings


class EmbeddingError(RuntimeError):
    pass


class EmbeddingProvider(ABC):
    """Embedding 介面。新增 Provider：繼承並實作 _embed_batch()，加入 build_provider()。"""

    name = 'base'

    def __init__(self, model_name: str | None = None, dim: int | None = None, batch_size: int | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.dim = dim or settings.EMBEDDING_DIM
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

    @abstractmethod
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors = await self._embed_batch(texts[i:i + self.batch_size])
            for v in vectors:
                if len(v) != self.dim:
                    raise EmbeddingError(
                        f'模型 {self.model_name} 回傳 {len(v)} 維，但 EMBEDDING_DIM={self.dim}；'
                        '請修正設定，或換模型後依 README「更換 Embedding 模型」處理')
            out.extend(vectors)
        return out

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]

    async def ping(self) -> bool:
        try:
            await self.embed_one('ping')
            return True
        except Exception:  # noqa: BLE001
            return False

    async def aclose(self) -> None:
        pass
