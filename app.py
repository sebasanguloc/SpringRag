"""Interfaz de chat de SpringRAG (Streamlit).

Ejecutar en local:  streamlit run app.py
"""
import os

import streamlit as st

# La API key llega por variables de entorno (.env en local) o por los
# "secrets" de la plataforma (Streamlit Cloud). Se copia al entorno ANTES de
# importar springrag, porque config.py lee el entorno al importarse.
try:
    for _k in ("GOOGLE_API_KEY", "LLM_MODEL"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # no hay archivo de secrets (uso local con .env)

from springrag import config  # noqa: E402
from springrag.embeddings import embed_query  # noqa: E402
from springrag.llm import CuotaAgotada  # noqa: E402
from springrag.pipeline import answer  # noqa: E402
from springrag.sources import SOURCES  # noqa: E402
from springrag.vectorstore import get_collection  # noqa: E402

st.set_page_config(page_title="SpringRAG", page_icon="🍃", layout="centered")

EJEMPLOS = [
    "¿Cómo cambio el puerto del servidor?",
    "Web server failed to start. Port 8080 was already in use",
    "¿Qué es un perfil de Spring y cómo lo activo?",
    "¿Cuáles son las fases del ciclo de vida de Maven?",
]


@st.cache_resource(show_spinner="Cargando modelo de embeddings e índice...")
def cargar_recursos() -> int:
    """Carga una sola vez el índice y el modelo (es lento); devuelve el nº de chunks."""
    embed_query("calentamiento")  # fuerza la carga del modelo de embeddings
    return get_collection().count()


def mostrar_fuentes(fuentes: list[dict]) -> None:
    with st.expander(f"📚 Fuentes ({len(fuentes)})"):
        for s in fuentes:
            pagina = f", pág. {s['page']}" if s.get("page") else ""
            st.markdown(f"**[{s['n']}]** [{s['title']} › {s['section']}{pagina}]({s['url']})")
            st.caption(f"distancia: {s['distance']}")
            st.code(s["text"], language="text", wrap_lines=True)


def mostrar_mensaje(m: dict) -> None:
    with st.chat_message(m["role"]):
        if m.get("standalone"):
            st.caption(f"🔎 Búsqueda: {m['standalone']}")
        st.markdown(m["content"])
        if m.get("fuera_de_alcance"):
            st.info("Esta pregunta parece estar fuera del alcance de la base de conocimiento.")
        if m.get("sources"):
            mostrar_fuentes(m["sources"])


# ---------- Barra lateral ----------
with st.sidebar:
    st.header("🍃 SpringRAG")
    st.write("Asistente experto en **Java, Maven y Spring Boot**. Responde solo con la documentación oficial indexada y cita las fuentes.")
    if st.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.subheader("Preguntas de ejemplo")
    for i, ej in enumerate(EJEMPLOS):
        if st.button(ej, key=f"ej{i}", use_container_width=True):
            st.session_state.pendiente = ej
    with st.expander("Documentos indexados"):
        for s in SOURCES:
            st.markdown(f"- [{s.title}]({s.url})")
    st.caption(
        f"top_k={config.TOP_K} · chunk={config.CHUNK_SIZE}/{config.CHUNK_OVERLAP} · "
        f"umbral={config.MAX_DISTANCE} · búsqueda en inglés={'sí' if config.SEARCH_IN_ENGLISH else 'no'} · "
        f"modelo={config.LLM_MODEL}"
    )

# ---------- Cuerpo ----------
st.title("SpringRAG")
st.caption("Pregunta sobre errores y configuración de Java, Maven y Spring Boot.")

if not config.GOOGLE_API_KEY:
    st.error("Falta la variable GOOGLE_API_KEY. Configúrala en `.env` (local) o en los secrets de la plataforma.")
    st.stop()

try:
    n_chunks = cargar_recursos()
except Exception:
    st.error("No se encontró el índice vectorial. Ejecuta primero `python scripts/build_index.py`.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    mostrar_mensaje(m)

pregunta = st.chat_input("Escribe tu pregunta o pega un error...") or st.session_state.pop("pendiente", None)

if pregunta:
    # El historial que ve el pipeline: solo rol y contenido, SIN el turno actual
    historial = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": pregunta})
    mostrar_mensaje(st.session_state.messages[-1])

    with st.chat_message("assistant"):
        try:
            with st.spinner("Buscando en la documentación..."):
                res = answer(pregunta, historial)
        except CuotaAgotada as e:
            st.error(str(e))
            st.session_state.messages.pop()
            st.stop()
        except Exception as e:  # p. ej. fallo de red o del servidor
            st.error(f"No pude generar la respuesta ({type(e).__name__}). Intenta de nuevo en unos segundos.")
            st.session_state.messages.pop()  # no dejar la pregunta sin respuesta en el historial
            st.stop()

    msg = {
        "role": "assistant",
        "content": res.text,
        "sources": res.sources,
        "fuera_de_alcance": not res.relevant,
        "standalone": res.standalone_question if res.standalone_question != pregunta else "",
    }
    st.session_state.messages.append(msg)
    st.rerun()  # redibuja todo desde el historial (una sola ruta de pintado)
