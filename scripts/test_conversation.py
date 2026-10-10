"""Simula una conversación con preguntas de seguimiento para probar la memoria.

Uso:  python scripts/test_conversation.py
"""
from springrag.pipeline import answer

PREGUNTAS = [
    "¿Cómo cambio el puerto del servidor en Spring Boot?",
    "¿y cómo lo hago en YAML?",
    "¿y si quiero que sea aleatorio?",
    "¿Cómo hago una receta de pizza?",
]

history: list[dict] = []
for p in PREGUNTAS:
    res = answer(p, history)
    print("=" * 70)
    print(f"USUARIO: {p}")
    if res.standalone_question != p:
        print(f"  (reescrita para buscar: {res.standalone_question})")
    print(f"\nASISTENTE:\n{res.text}")
    print("\n  Fuentes: " + (", ".join(f"[{s['n']}] {s['section']}" for s in res.sources) or "ninguna"))
    # Se guarda el turno en el historial: así la siguiente pregunta puede depender de este
    history += [{"role": "user", "content": p}, {"role": "assistant", "content": res.text}]
