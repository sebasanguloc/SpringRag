"""Recuperación: dada una pregunta, devuelve los chunks más parecidos.

Mecanismo: búsqueda por similitud (distancia coseno) de los `top_k` vectores
más cercanos al vector de la pregunta. Si ni siquiera el más cercano está
razonablemente cerca (umbral `max_distance`), se avisa de que la pregunta
parece estar fuera de la base de conocimiento.
"""
from dataclasses import dataclass
from functools import lru_cache

from springrag import config
from springrag.embeddings import embed_query
from springrag.vectorstore import get_collection


@dataclass
class Retrieved:
    text: str
    metadata: dict
    distance: float  # 0 = idéntico; más alto = menos parecido


@dataclass
class RetrievalResult:
    chunks: list[Retrieved]
    relevant: bool  # False => la pregunta parece estar fuera del corpus


@lru_cache(maxsize=4)
def _coleccion(name: str):
    return get_collection(name)  # se abre una vez por colección


def retrieve(
    pregunta: str,
    top_k: int = config.TOP_K,
    max_distance: float = config.MAX_DISTANCE,
    collection: str = config.COLLECTION_NAME,
) -> RetrievalResult:
    r = _coleccion(collection).query(
        query_embeddings=[embed_query(pregunta)],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    # Chroma devuelve listas de listas (una por pregunta); tenemos una sola pregunta
    chunks = [
        Retrieved(text=t, metadata=m, distance=d)
        for t, m, d in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
    ]
    relevant = bool(chunks) and chunks[0].distance <= max_distance
    return RetrievalResult(chunks=chunks, relevant=relevant)
