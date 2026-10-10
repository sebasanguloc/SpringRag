"""Descarga los documentos del catálogo a data/raw/.

Uso:  python scripts/download_corpus.py
"""
import requests

from springrag import config
from springrag.sources import SOURCES

# Algunos sitios rechazan peticiones sin User-Agent
HEADERS = {"User-Agent": "Mozilla/5.0 (SpringRAG - proyecto academico)"}


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        destino = config.RAW_DIR / src.filename
        if destino.exists():
            print(f"[ya existe] {src.filename}")
            continue
        resp = requests.get(src.url, headers=HEADERS, timeout=60)
        resp.raise_for_status()  # lanza error si el servidor no responde 200
        destino.write_bytes(resp.content)
        print(f"[descargado] {src.filename} ({len(resp.content) // 1024} KB)")


if __name__ == "__main__":
    main()
