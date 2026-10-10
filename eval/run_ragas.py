"""Evaluación del RAG con Ragas (faithfulness, answer_relevancy, context_precision, context_recall).

Flujo:
  1. Genera la respuesta del RAG para cada pregunta del dataset (se guarda en caché: reanudable).
  2. Mide el control de alucinaciones con las preguntas fuera del corpus (tasa de rechazo correcto).
  3. Evalúa con Ragas las preguntas del corpus usando un LLM como juez.

Uso:
  python eval/run_ragas.py --name baseline
  python eval/run_ragas.py --name baseline --only q01,q03     (prueba rápida con pocas preguntas)
  python eval/run_ragas.py --name baseline --skip-ragas       (solo genera respuestas)
"""
import argparse
import json
import sys
import types
import warnings
from pathlib import Path

# --- Compatibilidad: ragas 0.4.x importa un módulo de Vertex AI que langchain_community 0.4 eliminó.
# No usamos Vertex AI, así que se registra un módulo vacío para que el import no falle.
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault("langchain_community.chat_models.vertexai", _stub)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")  # aviso inofensivo de temperatura

import pandas as pd  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.cache import DiskCacheBackend  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig  # noqa: E402

from springrag import config, embeddings, llm  # noqa: E402
from springrag.llm import CuotaAgotada  # noqa: E402
from springrag.pipeline import answer  # noqa: E402
from springrag.retriever import retrieve  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DATASET = EVAL_DIR / "dataset.json"
RESULTS = EVAL_DIR / "results"
CACHE = EVAL_DIR / ".ragas_cache"

# Configuraciones evaluables. La iteración de mejora agrega aquí una entrada nueva.
CONFIGS = {
    "baseline": {"collection": "springrag_baseline", "top_k": 4},
    # Iteración 1: solo cambia el prompt de búsqueda (consulta en inglés). Mismo índice y mismo top_k.
    "iter1_english": {"collection": "springrag_baseline", "top_k": 4, "search_in_english": True},
}

REFUSAL = "No encuentro esta información en la base de conocimiento"
METRICS_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


class FastEmbedEmbeddings(Embeddings):
    """Adapta nuestros embeddings locales a la interfaz de LangChain que espera Ragas."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return embeddings.embed_query(text)


def generate_answers(items: list[dict], cfg: dict, path: Path) -> list[dict]:
    """Corre el RAG sobre cada pregunta. Guarda tras cada una para poder reanudar."""
    hechos = {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}
    for it in items:
        if it["id"] in hechos:
            continue
        print(f"  generando {it['id']}: {it['question'][:60]}")
        res = answer(
            it["question"],
            top_k=cfg["top_k"],
            collection=cfg["collection"],
            search_in_english=cfg.get("search_in_english", False),
        )
        # Contextos = lo que recuperó el buscador con la MISMA consulta que usó el pipeline
        # (aunque la respuesta haya sido un rechazo)
        chunks = retrieve(res.standalone_question, top_k=cfg["top_k"], collection=cfg["collection"]).chunks
        hechos[it["id"]] = {
            **it,
            "response": res.text,
            "contexts": [c.text for c in chunks],
            "relevant": res.relevant,
            "search_query": res.standalone_question,
            "refused": REFUSAL in res.text,
        }
        path.write_text(json.dumps(list(hechos.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    return [hechos[it["id"]] for it in items]


def hallucination_control(records: list[dict]) -> dict:
    fuera = [r for r in records if r["type"] == "out_of_corpus"]
    dentro = [r for r in records if r["type"] == "in_corpus"]
    return {
        "correct_refusal_rate": sum(r["refused"] for r in fuera) / len(fuera) if fuera else None,
        "n_out_of_corpus": len(fuera),
        "false_refusal_rate": sum(r["refused"] for r in dentro) / len(dentro) if dentro else None,
        "n_in_corpus": len(dentro),
    }


def run_ragas(records: list[dict], judge_model: str) -> pd.DataFrame:
    llm = ChatGoogleGenerativeAI(
        model=judge_model, google_api_key=config.GOOGLE_API_KEY, temperature=0, max_retries=1, timeout=90
    )
    cache = DiskCacheBackend(str(CACHE))  # cada llamada al juez queda guardada: reanudable
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["question"],
                "response": r["response"],
                "retrieved_contexts": r["contexts"],
                "reference": r["ground_truth"],
            }
            for r in records
        ]
    )
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            # strictness=1: Gemini no admite varias generaciones por llamada (error 400 con n>1)
            ResponseRelevancy(strictness=1),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=LangchainLLMWrapper(llm, cache=cache),
        embeddings=LangchainEmbeddingsWrapper(FastEmbedEmbeddings()),
        # max_workers=1: el plan gratuito permite 15 peticiones/minuto por modelo; con 2 en paralelo
        # se superaba en ráfagas y fallaban llamadas (celdas NaN)
        run_config=RunConfig(timeout=120, max_retries=8, max_wait=60, max_workers=1),
        raise_exceptions=False,  # un fallo puntual deja NaN en esa celda en vez de abortar todo
        show_progress=True,
    )
    df = result.to_pandas().rename(columns={"llm_context_precision_with_reference": "context_precision"})
    df.insert(0, "id", [r["id"] for r in records])
    return df[["id", "user_input", *METRICS_COLS]]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline", choices=CONFIGS)
    ap.add_argument("--only", help="ids separados por coma, p. ej. q01,q03")
    ap.add_argument("--skip-ragas", action="store_true")
    ap.add_argument("--judge-model", default=config.LLM_MODEL)
    args = ap.parse_args()

    cfg = CONFIGS[args.name]
    llm.set_min_interval(4.5)  # 15 peticiones/minuto en el plan gratuito => una cada 4 s como mínimo
    items = json.loads(DATASET.read_text(encoding="utf-8"))
    if args.only:
        quiero = set(args.only.split(","))
        items = [i for i in items if i["id"] in quiero]
    RESULTS.mkdir(exist_ok=True)
    print(f"Configuración '{args.name}': {cfg} | juez: {args.judge_model} | {len(items)} preguntas")

    print("1) Respuestas del RAG")
    try:
        records = generate_answers(items, cfg, RESULTS / f"{args.name}_answers.json")
    except CuotaAgotada as e:
        sys.exit(f"\nCuota agotada: {e}\nLo ya generado quedó guardado; vuelve a ejecutar luego para continuar.")

    print("2) Control de alucinaciones")
    control = hallucination_control(records)
    print(json.dumps(control, indent=2))
    if args.skip_ragas:
        return

    print("3) Ragas (juez LLM)")
    en_corpus = [r for r in records if r["type"] == "in_corpus"]
    df = run_ragas(en_corpus, args.judge_model)
    df.to_csv(RESULTS / f"{args.name}_ragas.csv", index=False, encoding="utf-8")

    resumen = {
        "config": {"name": args.name, **cfg, "chunk_size": config.CHUNK_SIZE, "judge": args.judge_model},
        "retrieval_queries": {r["id"]: r.get("search_query", r["question"]) for r in records},
        "metrics": {c: (None if df[c].isna().all() else round(float(df[c].mean()), 3)) for c in METRICS_COLS},
        "nan_cells": {c: int(df[c].isna().sum()) for c in METRICS_COLS},
        "hallucination_control": control,
    }
    (RESULTS / f"{args.name}_summary.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== RESUMEN ===")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
