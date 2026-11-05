"""測試 / 開發用：以字元 n-gram 雜湊產生固定向量，不需模型與網路。

沒有真正的語意能力（只反映字面重疊），**不可用於正式環境**。
"""
import hashlib
import math

from src.embedding.base import EmbeddingProvider


class HashProvider(EmbeddingProvider):
    name = 'hash'

    def __init__(self, **kw):
        kw.setdefault('model_name', 'hash-ngram-v1')
        super().__init__(**kw)

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        t = text.lower()
        grams = [t[i:i + n] for n in (1, 2, 3) for i in range(max(0, len(t) - n + 1))] or ['']
        for g in grams:
            h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), 'big')
            v[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    async def _embed_batch(self, texts):
        return [self._vector(t) for t in texts]
