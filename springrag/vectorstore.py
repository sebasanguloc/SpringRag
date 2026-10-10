"""Base de datos vectorial (Chroma): guarda chunks, sus vectores y sus metadatos.

Chroma persiste en disco (data/chroma/), así que el índice se construye una
vez y luego solo se carga. Los metadatos (título, URL, sección, página)
son los que permiten citar la fuente de cada respuesta.
"""
import chromadb

from springrag import config
from springrag.embeddings import embed_documents
from springrag.ingest import Document

BATCH = 64  # cuántos chunks se vectorizan y guardan por tanda


def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection(name: str = config.COLLECTION_NAME):
    """Abre una colección ya construida (falla si no existe)."""
    return _client().get_collection(name)


def build_index(chunks: list[Document], name: str = config.COLLECTION_NAME):
    """Crea la colección desde cero, vectoriza los chunks y los guarda."""
    client = _client()
    try:
        client.delete_collection(name)  # reconstruir = empezar limpio
    except Exception:
        pass  # no existía todavía
    # "cosine": la distancia se mide por el ángulo entre vectores (lo habitual en texto)
    coleccion = client.create_collection(name, metadata={"hnsw:space": "cosine"})

    for i in range(0, len(chunks), BATCH):
        lote = chunks[i : i + BATCH]
        textos = [c.text for c in lote]
        coleccion.add(
            ids=[c.metadata["chunk_id"] for c in lote],
            documents=textos,
            embeddings=embed_documents(textos),
            metadatas=[c.metadata for c in lote],
        )
        print(f"  indexados {min(i + BATCH, len(chunks))}/{len(chunks)}", end="\r")
    print()
    return coleccion
