"""
Valutazione del sistema.

    python -m scripts.run_eval

Produce results/evaluation.json con i quattro run confrontati e le metriche
per singola query, piu' l'accuratezza del tagger sui due livelli.
"""
from __future__ import annotations

import json

import pandas as pd

from src import config, conversations, evaluation
from src.indexing import VoiceTagIndex


def _fmt(summary: dict) -> str:
    return (f"P@5={summary['P@5']:.3f}  P@10={summary['P@10']:.3f}  "
            f"R@10={summary['R@10']:.3f}  MAP={summary['MAP']:.3f}")


def main():
    index = VoiceTagIndex.load()
    manifest = conversations.load_manifest()
    judgments = conversations.relevance_judgments(manifest)

    print(f"Collezione: {index.n_docs} conversazioni")
    print("Documenti rilevanti per tag:",
          {t: len(s) for t, s in judgments.items()})

    runs = {}

    runs["canoniche_ranked"] = evaluation.run_queries(
        index, judgments, evaluation.CANONICAL_QUERIES, engine="ranked")
    runs["canoniche_boolean"] = evaluation.run_queries(
        index, judgments, evaluation.CANONICAL_QUERIES, engine="boolean")
    runs["naturali_ranked"] = evaluation.run_queries(
        index, judgments, evaluation.NATURAL_QUERIES, engine="ranked")
    runs["naturali_senza_espansione"] = evaluation.run_without_expansion(
        index, judgments, evaluation.NATURAL_QUERIES)

    print("\n--- Confronto dei run " + "-" * 30)
    for name, run in runs.items():
        print(f"{name:32s} {_fmt(run['summary'])}")

    tagger_acc = evaluation.tagger_accuracy(manifest, index.docs)
    print("\n--- Tagger " + "-" * 40)
    print(f"accuratezza 8 classi   : {tagger_acc['accuratezza_8_classi']:.3f}")
    print(f"accuratezza 3 livelli  : {tagger_acc['accuratezza_3_livelli']:.3f}")
    print(f"turni valutati         : {tagger_acc['n_turni']}")

    out = {
        "collezione": index.stats(),
        "rilevanti_per_tag": {t: len(s) for t, s in judgments.items()},
        "tagger": tagger_acc,
        "runs": {k: v["summary"] for k, v in runs.items()},
        "dettaglio": {k: v["per_query"] for k, v in runs.items()},
    }
    path = config.RESULTS_DIR / "evaluation.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSalvato in {path}")

    df = pd.DataFrame([v["summary"] for v in runs.values()])
    df.insert(0, "run", list(runs))
    df.to_csv(config.RESULTS_DIR / "evaluation_summary.csv", index=False)
    return out


if __name__ == "__main__":
    main()
