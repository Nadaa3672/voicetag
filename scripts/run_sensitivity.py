"""
Analisi di sensibilita' e ablazioni.

    python -m scripts.run_sensitivity

Tre esperimenti, tutti eseguiti sull'indice gia' costruito (non richiedono di
rieseguire il tagging):

  1. soglia di document frequency - quanto il sistema dipende dal parametro che
     decide se un tag "compare" in un documento;
  2. potatura delle posting list - compromesso fra lunghezza dell'indice
     (efficienza) e MAP (efficacia);
  3. correzione per il prior del tagger - il classificatore sovrastima alcune
     classi; si verifica se dividere le probabilita' per la loro frequenza media
     sulla collezione, nello spirito dell'inverse user frequency, migliora il
     ranking o se l'idf assorbe gia' lo squilibrio.
"""
from __future__ import annotations

import json

import numpy as np

from src import config, conversations, evaluation, tagging
from src.indexing import VoiceTagIndex


def _documents(index: VoiceTagIndex) -> list[dict]:
    return [{k: (np.asarray(v) if k == "probs" else v) for k, v in d.items()}
            for d in index.docs.values()]


def _retag(documents: list[dict], mode: str, prior: np.ndarray) -> list[dict]:
    """
    L'indice conserva le probabilita' gia' calibrate. Per l'ablazione si torna
    alle probabilita' grezze moltiplicando di nuovo per il prior: la
    calibrazione e' una riscalatura per classe seguita da rinormalizzazione,
    quindi e' esattamente invertibile.
    """
    out = []
    for d in documents:
        p = d["probs"]
        if mode == "non_calibrato":
            p = p * prior
            p = p / p.sum(axis=1, keepdims=True)
        nd = dict(d)
        nd["probs"] = p.astype(np.float32)
        nd["tf"] = tagging.soft_tf(nd["probs"])
        nd["best_seg"] = {t: tagging.dominant_segment(nd["probs"], t)
                          for t in config.TAG_VOCAB}
        out.append(nd)
    return out


def main():
    index = VoiceTagIndex.load()
    manifest = conversations.load_manifest()
    judgments = conversations.relevance_judgments(manifest)
    documents = _documents(index)
    prior = (np.asarray(index.prior) if index.prior is not None
             else np.vstack([d["probs"] for d in documents]).mean(axis=0))

    report: dict = {}

    # --- 1. soglia di document frequency --------------------------------
    print("--- soglia di document frequency " + "-" * 24)
    rows = []
    for tau in (0.05, 0.10, 0.15, 0.20, 0.25):
        idx = VoiceTagIndex().build(documents, tag_sim=index.tag_sim, df_threshold=tau)
        s = evaluation.run_queries(idx, judgments,
                                   evaluation.CANONICAL_QUERIES)["summary"]
        rows.append({"soglia_df": tau, "P@5": s["P@5"], "R-prec": s["R-prec"],
                     "MAP": s["MAP"]})
        print(f"  tau={tau:.2f}  P@5={s['P@5']:.3f}  R-prec={s['R-prec']:.3f}  "
              f"MAP={s['MAP']:.3f}")
    report["soglia_df"] = rows

    # --- 2. potatura delle posting list ---------------------------------
    print("--- potatura delle posting list " + "-" * 25)
    rows = []
    for thr in (0.0, 0.02, 0.05, 0.08, 0.12):
        idx = VoiceTagIndex().build(documents, tag_sim=index.tag_sim,
                                    posting_threshold=thr)
        s = evaluation.run_queries(idx, judgments,
                                   evaluation.CANONICAL_QUERIES)["summary"]
        n_post = sum(len(v) for v in idx.postings.values())
        cand = float(np.mean([len(idx.candidates([t])) for t in config.EMOTIONS]))
        rows.append({"soglia_posting": thr, "posting": n_post,
                     "candidati_medi": cand, "P@5": s["P@5"],
                     "R-prec": s["R-prec"], "MAP": s["MAP"]})
        print(f"  soglia={thr:.2f}  posting={n_post:5d}  "
              f"candidati/tag={cand:6.1f}  MAP={s['MAP']:.3f}")
    report["potatura_posting"] = rows

    # --- 3. calibrazione sul prior --------------------------------------
    print("--- calibrazione sul prior del tagger " + "-" * 19)
    rows = []
    for mode in ("non_calibrato", "calibrato"):
        docs = _retag(documents, mode, prior)
        idx = VoiceTagIndex().build(docs, tag_sim=index.tag_sim)
        s = evaluation.run_queries(idx, judgments,
                                   evaluation.CANONICAL_QUERIES)["summary"]
        acc = evaluation.tagger_accuracy(manifest, idx.docs)
        rows.append({"modalita": mode, "P@5": s["P@5"], "R-prec": s["R-prec"],
                     "MAP": s["MAP"],
                     "acc_8": acc["accuratezza_8_classi"],
                     "acc_3": acc["accuratezza_3_livelli"]})
        print(f"  {mode:14s}  MAP={s['MAP']:.3f}  R-prec={s['R-prec']:.3f}  "
              f"acc8={acc['accuratezza_8_classi']:.3f}  "
              f"acc3={acc['accuratezza_3_livelli']:.3f}")
    report["calibrazione"] = rows
    report["prior_del_tagger"] = {e: round(float(p), 4)
                                  for e, p in zip(config.EMOTIONS, prior)}

    path = config.RESULTS_DIR / "sensitivity.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSalvato in {path}")
    return report


if __name__ == "__main__":
    main()
