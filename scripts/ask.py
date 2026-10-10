"""Hace una pregunta al RAG completo desde la terminal.

Uso:  python scripts/ask.py "¿Cómo cambio el puerto del servidor?"
"""
import sys

from springrag.pipeline import answer

pregunta = " ".join(sys.argv[1:]) or "¿Cómo cambio el puerto del servidor?"
res = answer(pregunta)

print(f"\nPREGUNTA: {pregunta}\n")
print(res.text)
print("\nFUENTES:")
if not res.sources:
    print("  (ninguna: pregunta fuera del corpus, no se llamó al LLM)")
for s in res.sources:
    pagina = f", pág. {s['page']}" if s["page"] else ""
    print(f"  [{s['n']}] {s['title']} > {s['section']}{pagina}  (dist={s['distance']})")
    print(f"      {s['url']}")
