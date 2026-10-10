"""Cliente del LLM (Gemini). Un solo punto de acceso al modelo generativo."""
import re
import time
from functools import lru_cache

from google import genai
from google.genai import errors, types

from springrag import config


class CuotaAgotada(RuntimeError):
    """Se agotó la cuota DIARIA del plan gratuito: reintentar en segundos no sirve."""


# Separación mínima (segundos) entre llamadas. 0 en la app; la evaluación la sube para no
# superar el límite por minuto del plan gratuito (15 peticiones/minuto).
_min_interval = 0.0
_ultima_llamada = 0.0


def set_min_interval(segundos: float) -> None:
    global _min_interval
    _min_interval = segundos


def _espera_sugerida(e: Exception) -> float:
    """Lee 'Please retry in 8.1s' / retryDelay del error de Google; 0 si no aparece."""
    m = re.search(r"retry in ([0-9.]+)s|retryDelay'?: ?'([0-9.]+)s", str(e))
    return float(m.group(1) or m.group(2)) if m else 0.0


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    if not config.GOOGLE_API_KEY:
        raise RuntimeError("Falta GOOGLE_API_KEY (archivo .env o secrets de la plataforma).")
    return genai.Client(
        api_key=config.GOOGLE_API_KEY,
        # timeout: no quedarse colgado; attempts=1: los reintentos los controlamos nosotros abajo
        http_options=types.HttpOptions(
            timeout=60_000, retry_options=types.HttpRetryOptions(attempts=1)
        ),
    )


def generate(system: str, turnos: list[tuple[str, str]], intentos: int = 6) -> str:
    """Envía el system prompt + una lista de turnos [(rol, texto)] y devuelve el texto.

    rol es "user" o "model". Los Few-Shot son turnos previos de la conversación.
    """
    contents = [types.Content(role=rol, parts=[types.Part(text=txt)]) for rol, txt in turnos]
    cfg = types.GenerateContentConfig(
        system_instruction=system, temperature=config.LLM_TEMPERATURE
    )
    global _ultima_llamada
    for n in range(intentos):
        falta = _min_interval - (time.time() - _ultima_llamada)
        if falta > 0:
            time.sleep(falta)
        _ultima_llamada = time.time()
        try:
            resp = _client().models.generate_content(
                model=config.LLM_MODEL, contents=contents, config=cfg
            )
            return resp.text or ""
        except (errors.ServerError, errors.ClientError) as e:
            es_429 = getattr(e, "code", 0) == 429
            if es_429 and "PerDay" in str(e):
                # Cuota diaria agotada: esperar segundos no ayuda, avisar de inmediato
                raise CuotaAgotada(
                    f"Se agotó la cuota diaria gratuita del modelo {config.LLM_MODEL}. "
                    "Prueba otro modelo (variable LLM_MODEL) o espera a que se renueve."
                ) from e
            # 429 por minuto y errores 5xx son transitorios: esperar y reintentar
            if not (es_429 or isinstance(e, errors.ServerError)) or n == intentos - 1:
                raise
            time.sleep(max(_espera_sugerida(e) + 1, 5 * (n + 1)))
    return ""
