"""Construye el índice vectorial completo: ingesta -> chunking -> embeddings -> Chroma.

Uso:  python scripts/build_index.py
      python scripts/build_index.py --chunk-size 1200 --overlap 200 --collection springrag_improved
"""
import argparse

from springrag import config
from springrag.chunking import chunk_documents
from springrag.ingest import load_file
from springrag.sources import SOURCES
from springrag.vectorstore import build_index

parser = argparse.ArgumentParser()
parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
parser.add_argument("--collection", default=config.COLLECTION_NAME)
args = parser.parse_args()

print("1) Ingesta")
docs = []
for src in SOURCES:
    docs += load_file(config.RAW_DIR / src.filename, src)
print(f"   {len(docs)} secciones")

print(f"2) Chunking (size={args.chunk_size}, overlap={args.overlap})")
chunks = chunk_documents(docs, args.chunk_size, args.overlap)
print(f"   {len(chunks)} chunks")

print(f"3) Embeddings + Chroma (colección '{args.collection}')")
coleccion = build_index(chunks, args.collection)
print(f"Listo: {coleccion.count()} chunks guardados en {config.CHROMA_DIR}")
