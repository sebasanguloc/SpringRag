"""Muestra estadísticas de los chunks y algunos ejemplos reales.

Uso:  python scripts/inspect_chunks.py
      python scripts/inspect_chunks.py 1200 200   (tamaño y solapamiento de prueba)
"""
import statistics
import sys

from springrag import config
from springrag.chunking import chunk_documents
from springrag.ingest import load_file
from springrag.sources import SOURCES

size = int(sys.argv[1]) if len(sys.argv) > 1 else config.CHUNK_SIZE
overlap = int(sys.argv[2]) if len(sys.argv) > 2 else config.CHUNK_OVERLAP

docs = []
for src in SOURCES:
    docs += load_file(config.RAW_DIR / src.filename, src)

chunks = chunk_documents(docs, size, overlap)
largos = [len(c.text) for c in chunks]

print(f"chunk_size={size}  overlap={overlap}")
print(f"secciones: {len(docs)}  ->  chunks: {len(chunks)}")
print(f"largo (caracteres): min={min(largos)}  mediana={int(statistics.median(largos))}  max={max(largos)}")
print(f"chunks de menos de 150 caracteres: {sum(l < 150 for l in largos)}")

# Dos chunks consecutivos de la misma sección: se ve el solapamiento
for c in chunks:
    if c.metadata["chunk_in_section"] == 1:
        i = chunks.index(c)
        for k in (i - 1, i):
            print("\n" + "=" * 70)
            print(chunks[k].metadata["chunk_id"], chunks[k].metadata["section"])
            print(chunks[k].text)
        break
