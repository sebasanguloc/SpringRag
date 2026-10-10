"""Pipeline RAG de extremo a extremo con memoria de conversación.

pregunta (+ historial) -> reescribir -> recuperar -> prompt -> LLM -> respuesta + fuentes
"""
from dataclasses import dataclass, field

from springrag import config, llm, prompts
from springrag.retriever import retrieve


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)  # fragmentos que respaldan la respuesta
    relevant: bool = True  # False => la pregunta estaba fuera del corpus
    standalone_question: str = ""  # pregunta reescrita que se usó para buscar


def condense_question(pregunta: str, history: list[dict], english: bool = False) -> str:
    """Devuelve la consulta que se usará para BUSCAR.

    - Sin `english`: convierte un seguimiento en una pregunta completa (mismo idioma).
    - Con `english`: además la traduce a una consulta en inglés (una sola llamada al LLM).
    """
    recientes = history[-2 * config.HISTORY_TURNS :]
    if english:
        if recientes:  # seguimiento: resolver referencias y traducir
            mensaje = prompts.build_condense_message(recientes, pregunta)
            sistema = prompts.SEARCH_EN_FOLLOWUP_SYSTEM
        else:  # pregunta sola: traducción fiel (prompt evaluado con Ragas)
            mensaje = pregunta
            sistema = prompts.SEARCH_EN_SYSTEM
    elif not recientes:
        return pregunta  # primera pregunta: nada que resolver, no se gasta una llamada al LLM
    else:
        mensaje = prompts.build_condense_message(recientes, pregunta)
        sistema = prompts.CONDENSE_SYSTEM
    reescrita = llm.generate(sistema, [("user", mensaje)]).strip()
    return reescrita or pregunta  # si el modelo devuelve vacío, usamos la original


def answer(
    pregunta: str,
    history: list[dict] | None = None,
    top_k: int = config.TOP_K,
    collection: str = config.COLLECTION_NAME,
    search_in_english: bool = config.SEARCH_IN_ENGLISH,
) -> Answer:
    """`history` es una lista de {"role": "user" | "assistant", "content": str}."""
    history = history or []

    # 1) Reescribir la pregunta (resolver seguimientos y, opcionalmente, pasarla a inglés)
    standalone = condense_question(pregunta, history, english=search_in_english)

    # 2) Recuperación (con la pregunta ya completa)
    resultado = retrieve(standalone, top_k=top_k, collection=collection)

    # 3) Fuera del corpus: no se llama al LLM, así no hay riesgo de alucinar
    if not resultado.relevant:
        return Answer(prompts.NO_CORPUS_ANSWER, [], relevant=False, standalone_question=standalone)

    # 4) Prompt: Few-Shot + turnos recientes del diálogo + consulta actual con su contexto
    turnos = prompts.few_shot_messages()
    for m in history[-2 * config.HISTORY_TURNS :]:
        turnos.append(("user" if m["role"] == "user" else "model", m["content"]))
    contexto = prompts.format_context(resultado.chunks)
    turnos.append(("user", prompts.build_user_message(pregunta, contexto)))

    # 5) Generación
    texto = llm.generate(prompts.SYSTEM_PROMPT, turnos)

    # 6) Fuentes: mismo orden que los números [1], [2]... que cita el modelo
    fuentes = [
        {
            "n": i,
            "title": c.metadata["source_title"],
            "section": c.metadata["section"],
            "url": c.metadata["url"],
            "page": c.metadata.get("page"),
            "distance": round(c.distance, 3),
            "text": c.text,
        }
        for i, c in enumerate(resultado.chunks, start=1)
    ]
    return Answer(texto, fuentes, relevant=True, standalone_question=standalone)
