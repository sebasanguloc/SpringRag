"""Compara dos evaluaciones Ragas (antes/después) y genera tabla + gráfico.

Uso:  python eval/compare.py baseline improved
Salida: eval/results/comparison.md  y  docs/figures/ragas-comparacion.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo guarda el archivo
import matplotlib.pyplot as plt  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS = EVAL_DIR / "results"
FIGURES = EVAL_DIR.parent / "docs" / "figures"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    antes_n, despues_n = sys.argv[1], sys.argv[2]
    a = json.loads((RESULTS / f"{antes_n}_summary.json").read_text(encoding="utf-8"))
    d = json.loads((RESULTS / f"{despues_n}_summary.json").read_text(encoding="utf-8"))

    filas = ["| Métrica | Antes | Después | Cambio |", "|---|---|---|---|"]
    for m in METRICS:
        x, y = a["metrics"][m], d["metrics"][m]
        filas.append(f"| {m} | {x:.3f} | {y:.3f} | {y - x:+.3f} |")
    for clave, nombre in (("correct_refusal_rate", "rechazo correcto (fuera del corpus)"),
                          ("false_refusal_rate", "falsos rechazos (en el corpus)")):
        x, y = a["hallucination_control"][clave], d["hallucination_control"][clave]
        filas.append(f"| {nombre} | {x:.2f} | {y:.2f} | {y - x:+.2f} |")
    tabla = "\n".join(filas)
    (RESULTS / "comparison.md").write_text(
        f"Antes: `{antes_n}` {a['config']}\n\nDespués: `{despues_n}` {d['config']}\n\n{tabla}\n", encoding="utf-8"
    )
    print(tabla)

    # Gráfico de barras agrupadas
    FIGURES.mkdir(parents=True, exist_ok=True)
    ancho = 0.38
    xs = range(len(METRICS))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    b1 = ax.bar([i - ancho / 2 for i in xs], [a["metrics"][m] for m in METRICS], ancho, label="Antes", color="#9AA5AD")
    b2 = ax.bar([i + ancho / 2 for i in xs], [d["metrics"][m] for m in METRICS], ancho, label="Después", color="#4E9292")
    for barras in (b1, b2):
        ax.bar_label(barras, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(METRICS, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Puntaje (0 a 1)")
    ax.set_title("Métricas Ragas: antes y después de la mejora")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "ragas-comparacion.png", dpi=200)
    print(f"\nGráfico guardado en {FIGURES / 'ragas-comparacion.png'}")


if __name__ == "__main__":
    main()
