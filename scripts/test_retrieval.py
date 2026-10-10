"""Prueba la recuperación con preguntas dentro y fuera del dominio.

Uso:  python scripts/test_retrieval.py
      python scripts/test_retrieval.py "tu pregunta aquí"   (detalle de una pregunta)
"""
import sys

from springrag.retriever import retrieve

EN_DOMINIO = [
    "¿Cómo cambio el puerto del servidor?",
    "¿Qué es un perfil de Spring?",
    "¿Cuál es el orden de prioridad de las fuentes de propiedades?",
    "¿Cómo configuro el nivel de log de un paquete?",
    "Mi aplicación dice Port 8080 was already in use",
    "¿Qué hace @Transactional?",
    "¿Qué es un scope de dependencia en Maven?",
    "¿Cuáles son las fases del ciclo de vida de Maven?",
    "¿Cómo expongo los endpoints de actuator?",
    "Error al configurar el datasource de la base de datos",
]
FUERA_DE_DOMINIO = [
    "receta de pizza",
    "¿Cómo configuro Django con PostgreSQL?",
    "¿Quién ganó el mundial de 2022?",
    "¿Cómo uso React hooks?",
    "¿Qué es Kubernetes?",
    "Explícame la fotosíntesis",
]


def detalle(pregunta: str) -> None:
    res = retrieve(pregunta)
    print(f"\nPregunta: {pregunta}   (relevante={res.relevant})")
    for i, c in enumerate(res.chunks, 1):
        print(f"  [{i}] dist={c.distance:.3f}  {c.metadata['source_title']} > {c.metadata['section']}")


if len(sys.argv) > 1:
    detalle(" ".join(sys.argv[1:]))
else:
    for titulo, preguntas in (("EN DOMINIO", EN_DOMINIO), ("FUERA DE DOMINIO", FUERA_DE_DOMINIO)):
        print(f"\n=== {titulo} ===")
        for p in preguntas:
            res = retrieve(p)
            mejor = res.chunks[0]
            print(f"best={mejor.distance:.3f}  relevante={res.relevant!s:<5}  {p}")
