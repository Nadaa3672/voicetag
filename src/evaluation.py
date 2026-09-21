"""
Valutazione del sistema con metriche di Information Retrieval.

I giudizi di rilevanza non vengono annotati a mano: la collezione e' costruita
concatenando clip di emozione nota, quindi per ogni conversazione si sa
esattamente quali stati emotivi contiene. La rilevanza e' quindi definita sul
contenuto vero della registrazione, non sull'uscita del classificatore: le
metriche misurano il sistema completo (tagger + indice + ranking), non il
solo tagger.

Tre confronti:
  A. ranking pesato tf-idf   vs   filtro booleano su etichetta unica
  B. query canoniche (nome del tag)   vs   query in linguaggio naturale
  C. tag fini (8)   vs   livelli di valenza (3)
"""
from __future__ import annotations

import numpy as np

from src import config, retrieval
from src.indexing import VoiceTagIndex

# ----------------------------------------------------------------------
# Insieme di query. Ogni query ha un tag obiettivo: i documenti rilevanti
# sono quelli che contengono davvero quello stato emotivo.
# ----------------------------------------------------------------------
CANONICAL_QUERIES = [(e, e) for e in config.EMOTIONS] + \
                    [(v, v) for v in config.VALENCE_LEVELS]

NATURAL_QUERIES = [
    ("cliente arrabbiato", "angry"),
    ("chiamate con urla e aggressivita", "angry"),
    ("cliente tranquillo e sereno", "calm"),
    ("conversazioni pacate", "calm"),
    ("cliente contento del servizio", "happy"),
    ("chiamate con esito soddisfacente", "happy"),
    ("cliente abbattuto e amareggiato", "sad"),
    ("clienti rassegnati", "sad"),
    ("cliente spaventato", "fearful"),
    ("chiamante in ansia per un guasto urgente", "fearful"),
    ("cliente infastidito e seccato", "disgust"),
    ("reclami con tono sdegnato", "disgust"),
    ("cliente stupito", "surprised"),
    ("reazioni di incredulita", "surprised"),
    ("chiamate informative ordinarie", "neutral"),
    ("conversazioni dal tono piatto", "neutral"),
    ("chiamate con esperienza negativa", "negativo"),
    ("clienti insoddisfatti", "negativo"),
    ("chiamate con esperienza positiva", "positivo"),
    ("conversazioni senza carica emotiva", "neutro"),
]


# ----------------------------------------------------------------------
# Metriche
# ----------------------------------------------------------------------
def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """
    Precision@k nella definizione standard: rilevanti fra i primi k diviso k.
    Il denominatore e' k e non il numero di documenti effettivamente restituiti,
    altrimenti un sistema che restituisce un solo documento rilevante otterrebbe
    P@5 = 1, premiando la scarsa copertura invece di penalizzarla.
    """
    if k <= 0:
        return 0.0
    return sum(1 for d in ranked[:k] if d in relevant) / k


def r_precision(ranked: list[str], relevant: set[str]) -> float:
    """
    Precision calcolata a k = |R|. Metrica robusta quando gli insiemi di
    documenti rilevanti hanno dimensioni molto diverse fra una query e l'altra,
    come accade qui (da 26 a 140 conversazioni).
    """
    r = len(relevant)
    if r == 0:
        return 0.0
    return sum(1 for d in ranked[:r] if d in relevant) / r


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant)


def f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def average_precision(ranked: list[str], relevant: set[str]) -> float:
    """Media delle precision calcolate in corrispondenza di ogni documento rilevante."""
    if not relevant:
        return 0.0
    hits, acc = 0, 0.0
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            hits += 1
            acc += hits / i
    return acc / len(relevant)


# ----------------------------------------------------------------------
# Esecuzione di un run
# ----------------------------------------------------------------------
def run_queries(index: VoiceTagIndex, judgments: dict[str, set],
                queries: list[tuple[str, str]], engine="ranked",
                k_list=(5, 10)) -> dict:
    """
    Esegue un insieme di query e restituisce le metriche medie piu' il
    dettaglio per singola query.
    """
    search_fn = retrieval.search if engine == "ranked" else retrieval.boolean_search
    per_query, rows = [], []

    for text, target in queries:
        relevant = judgments.get(target, set())
        out = search_fn(text, index, k=index.n_docs)
        ranked = [r["conv_id"] for r in out["results"]]

        row = {"query": text, "target": target,
               "n_relevant": len(relevant), "n_retrieved": len(ranked)}
        for k in k_list:
            row[f"P@{k}"] = precision_at_k(ranked, relevant, k)
            row[f"R@{k}"] = recall_at_k(ranked, relevant, k)
            row[f"F1@{k}"] = f1(row[f"P@{k}"], row[f"R@{k}"])
        row["R-prec"] = r_precision(ranked, relevant)
        row["AP"] = average_precision(ranked, relevant)
        rows.append(row)
        per_query.append(row)

    summary = {"engine": engine, "n_query": len(rows)}
    for key in rows[0]:
        if key in ("query", "target", "n_relevant", "n_retrieved"):
            continue
        summary[key] = float(np.mean([r[key] for r in rows]))
    summary["MAP"] = summary.pop("AP")
    return {"summary": summary, "per_query": per_query}


def run_without_expansion(index: VoiceTagIndex, judgments: dict[str, set],
                          queries: list[tuple[str, str]], k_list=(5, 10)) -> dict:
    """
    Ablazione: disattiva il thesaurus e accetta come termini di query soltanto
    i nomi esatti dei tag. Misura il costo del vocabulary mismatch.
    """
    backup = dict(retrieval.THESAURUS)
    retrieval.THESAURUS.clear()
    retrieval.THESAURUS.update({e: {e: 1.0} for e in config.EMOTIONS})
    try:
        original_fuzzy = retrieval._fuzzy
        retrieval._fuzzy = lambda token: None
        out = run_queries(index, judgments, queries, engine="ranked", k_list=k_list)
        out["summary"]["engine"] = "ranked_no_expansion"
    finally:
        retrieval._fuzzy = original_fuzzy
        retrieval.THESAURUS.clear()
        retrieval.THESAURUS.update(backup)
    return out


# ----------------------------------------------------------------------
# Accuratezza del tagger a due livelli (misura di supporto, non IR)
# ----------------------------------------------------------------------
def tagger_accuracy(manifest: list[dict], documents: dict[str, dict]) -> dict:
    """
    Accuratezza a livello di turno: confronta l'emozione vera di ogni turno con
    quella predetta dal segmento che lo copre al centro. Riportata sia sulle 8
    emozioni sia sui 3 livelli di valenza.
    """
    n_fine, n_coarse = len(config.EMOTIONS), len(config.VALENCE_LEVELS)
    cm_fine = np.zeros((n_fine, n_fine), dtype=int)
    cm_coarse = np.zeros((n_coarse, n_coarse), dtype=int)
    fine_ok = fine_n = coarse_ok = coarse_n = 0
    for doc in manifest:
        d = documents.get(doc["conv_id"])
        if d is None:
            continue
        probs, spans = np.asarray(d["probs"]), d["spans"]
        centers = np.array([(s + e) / 2 for s, e in spans])
        for turn in doc["turns"]:
            mid = (turn["start"] + turn["end"]) / 2
            i = int(np.argmin(np.abs(centers - mid)))
            pred = config.EMOTIONS[int(np.argmax(probs[i]))]
            true = turn["emotion"]
            fine_n += 1
            fine_ok += int(pred == true)
            cm_fine[config.EMOTION_TO_ID[true], config.EMOTION_TO_ID[pred]] += 1

            vt = config.EMOTION_TO_VALENCE[true]
            vp = config.EMOTION_TO_VALENCE[pred]
            coarse_n += 1
            coarse_ok += int(vp == vt)
            cm_coarse[config.VALENCE_LEVELS.index(vt),
                      config.VALENCE_LEVELS.index(vp)] += 1
    # riferimenti onesti: livello di caso e classe di maggioranza
    support_fine = cm_fine.sum(axis=1)
    support_coarse = cm_coarse.sum(axis=1)
    return {
        "accuratezza_8_classi": fine_ok / max(fine_n, 1),
        "accuratezza_3_livelli": coarse_ok / max(coarse_n, 1),
        "caso_8_classi": 1.0 / n_fine,
        "caso_3_livelli": 1.0 / n_coarse,
        "maggioranza_8_classi": float(support_fine.max()) / max(fine_n, 1),
        "maggioranza_3_livelli": float(support_coarse.max()) / max(coarse_n, 1),
        "n_turni": fine_n,
        "confusione_8": cm_fine.tolist(),
        "confusione_3": cm_coarse.tolist(),
    }
