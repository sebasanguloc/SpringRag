"""Chunking: parte cada Document (sección/página) en fragmentos pequeños.

Estrategia en dos niveles:
1. La ingesta ya separó por secciones/páginas, así que un chunk nunca mezcla temas.
2. Aquí, si una sección es más larga que `chunk_size`, se corta en trozos con
   solapamiento, prefiriendo cortar en límites naturales (párrafo > línea > espacio).
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from springrag import config
from springrag.ingest import Document


def chunk_documents(
    docs: list[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # de "más natural" a "último recurso"
    )

    chunks: list[Document] = []
    for doc in docs:
        for i, trozo in enumerate(splitter.split_text(doc.text)):
            # Anteponer título y sección da contexto al chunk: un fragmento como
            # "server.port=9090" se entiende mejor sabiendo de qué documento viene.
            encabezado = f"{doc.metadata['source_title']} > {doc.metadata['section']}"
            chunks.append(
                Document(
                    text=f"{encabezado}\n{trozo}",
                    metadata={**doc.metadata, "chunk_in_section": i},
                )
            )

    # Identificador único y estable de cada chunk (lo usará la base vectorial)
    for n, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk-{n:05d}"
    return chunks
