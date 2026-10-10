"""Muestra qué texto extrae la ingesta de cada documento del catálogo.

Uso:  python scripts/inspect_ingest.py            (resumen de todos)
      python scripts/inspect_ingest.py sb_profiles (detalle de uno)
"""
import sys

from springrag import config
from springrag.ingest import load_file
from springrag.sources import SOURCES

if len(sys.argv) > 1:
    src = next(s for s in SOURCES if s.id == sys.argv[1])
    docs = load_file(config.RAW_DIR / src.filename, src)
    for d in docs[:3]:
        print("=" * 70)
        print(d.metadata)
        print(d.text[:600])
else:
    print(f"{'documento':<24}{'secciones':>10}{'caracteres':>12}")
    for src in SOURCES:
        docs = load_file(config.RAW_DIR / src.filename, src)
        print(f"{src.id:<24}{len(docs):>10}{sum(len(d.text) for d in docs):>12}")
