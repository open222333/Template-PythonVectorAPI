from src.embedding.base import EmbeddingError, EmbeddingProvider


def build_provider(kind: str | None = None) -> EmbeddingProvider:
    from src import settings
    kind = kind or settings.EMBEDDING_PROVIDER
    if kind == 'ollama':
        from src.embedding.ollama import OllamaProvider
        return OllamaProvider()
    if kind == 'openai_compat':
        from src.embedding.openai_compat import OpenAICompatProvider
        return OpenAICompatProvider()
    if kind == 'hash':
        from src.embedding.hash import HashProvider
        return HashProvider()
    raise ValueError(f'未知的 EMBEDDING_PROVIDER：{kind}（可用 ollama / openai_compat / hash）')


__all__ = ['EmbeddingProvider', 'EmbeddingError', 'build_provider']
