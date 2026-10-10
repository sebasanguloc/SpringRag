"""Embeddings: convierte texto en vectores numéricos con un modelo local.

El modelo corre en nuestra máquina (ONNX vía fastembed): el texto del corpus
no se envía a ningún servicio externo para vectorizarse.
"""
from functools import lru_cache

from fastembed import TextEmbedding

from springrag import config


@lru_cache(maxsize=1)
def _modelo() -> TextEmbedding:
    # lru_cache: el modelo se carga una sola vez y se reutiliza (cargarlo es lento)
    return TextEmbedding(model_name=config.EMBEDDING_MODEL)


def embed_documents(textos: list[str]) -> list[list[float]]:
    """Vectores para una lista de textos (se usa al construir el índice)."""
    return [v.tolist() for v in _modelo().embed(textos)]


def embed_query(texto: str) -> list[float]:
    """Vector para una sola pregunta (se usa al buscar)."""
    return embed_documents([texto])[0]
