"""Configuración central del proyecto.

Todos los parámetros del pipeline viven aquí para poder cambiarlos
(y comparar evaluaciones) sin tocar el resto del código.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Carga las variables del archivo .env (si existe) al entorno del proceso
load_dotenv()

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

# Chunking (en caracteres). Se pueden sobreescribir con variables de entorno
# para probar otras configuraciones sin tocar el código.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# Embeddings (local, multilingüe: preguntas en español, documentos en inglés)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Base vectorial (Chroma persistente en disco; se versiona en el repo)
CHROMA_DIR = BASE_DIR / "data" / "chroma"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "springrag_baseline")

# Recuperación
TOP_K = int(os.getenv("TOP_K", "4"))  # cuántos chunks se recuperan por pregunta
# Si el chunk más cercano está más lejos que esto (distancia coseno), se
# considera que la pregunta NO está en la base de conocimiento.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.60"))

# Si es True, la pregunta se reescribe como consulta en inglés ANTES de buscar
# (los documentos están en inglés). La respuesta sigue siendo en español.
# Activado por defecto: fue la mejora validada con Ragas (iteración 1). La línea base usaba False.
SEARCH_IN_ENGLISH = os.getenv("SEARCH_IN_ENGLISH", "true").lower() in ("1", "true", "yes")

# Conversación: cuántos intercambios (pregunta + respuesta) recientes se recuerdan
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "3"))

# Modelo de lenguaje (se puede cambiar con la variable de entorno LLM_MODEL)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))  # bajo = respuestas más estables
