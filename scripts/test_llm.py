"""Prueba mínima: ¿funciona la API key y el modelo responde?"""
from google import genai

from springrag import config

if not config.GOOGLE_API_KEY:
    raise SystemExit("Falta GOOGLE_API_KEY. Crea el archivo .env (ver .env.example).")

client = genai.Client(api_key=config.GOOGLE_API_KEY)
respuesta = client.models.generate_content(
    model=config.LLM_MODEL,
    contents="Responde en una frase: ¿qué es Spring Boot?",
)
print(respuesta.text)
