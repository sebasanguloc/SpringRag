Antes: `baseline` {'name': 'baseline', 'collection': 'springrag_baseline', 'top_k': 4, 'chunk_size': 800, 'judge': 'gemini-3.5-flash-lite'}

Después: `iter1_english` {'name': 'iter1_english', 'collection': 'springrag_baseline', 'top_k': 4, 'search_in_english': True, 'chunk_size': 800, 'judge': 'gemini-3.5-flash-lite'}

| Métrica | Antes | Después | Cambio |
|---|---|---|---|
| faithfulness | 0.876 | 0.972 | +0.096 |
| answer_relevancy | 0.662 | 0.674 | +0.012 |
| context_precision | 0.520 | 0.711 | +0.191 |
| context_recall | 0.694 | 0.815 | +0.121 |
| rechazo correcto (fuera del corpus) | 1.00 | 1.00 | +0.00 |
| falsos rechazos (en el corpus) | 0.11 | 0.00 | -0.11 |
