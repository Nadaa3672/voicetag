"""
Dai segmenti ai tag del documento.

Tre operazioni:

1. *soft term frequency* - la frequenza di un tag in una conversazione e' la
   somma delle probabilita' che il tagger gli assegna sui singoli segmenti.
   E' il numero atteso di segmenti che esprimono quel tag: un valore continuo
   che conserva l'incertezza del modello invece di buttarla via con un argmax.

2. *similarita' fra tag* - stimata empiricamente sulla collezione come coseno
   fra le colonne della matrice segmenti x tag. Due tag sono simili se il
   tagger tende ad attivarli insieme. Serve a MMR.

3. *selezione con MMR* - i tag mostrati vengono scelti uno alla volta
   bilanciando rilevanza rispetto al documento e non-ridondanza rispetto ai tag
   gia' scelti. Senza questo passaggio una conversazione tranquilla riceve
   'neutral' e 'calm', che il modello confonde sistematicamente e che per un
   utente sono lo stesso tag.
"""
from __future__ import annotations

import numpy as np

from src import config
from src.tagger import to_valence


# ----------------------------------------------------------------------
# 1. Soft term frequency
# ----------------------------------------------------------------------
def soft_tf(probs: np.ndarray) -> dict[str, float]:
    """
    Frequenze dei tag per un documento, su entrambi i livelli della tassonomia.
    `probs` ha forma (n_segmenti, 8).
    """
    fine = probs.sum(axis=0)
    coarse = to_valence(probs).sum(axis=0)
    tf = {e: float(fine[i]) for i, e in enumerate(config.EMOTIONS)}
    tf.update({v: float(coarse[j]) for j, v in enumerate(config.VALENCE_LEVELS)})
    return tf


def dominant_segment(probs: np.ndarray, tag: str) -> int:
    """Indice del segmento in cui il tag dato ha la probabilita' piu' alta."""
    if tag in config.EMOTION_TO_ID:
        col = probs[:, config.EMOTION_TO_ID[tag]]
    elif tag in config.VALENCE:
        idx = [config.EMOTION_TO_ID[e] for e in config.VALENCE[tag]]
        col = probs[:, idx].sum(axis=1)
    else:
        col = probs.max(axis=1)
    return int(np.argmax(col))


# ----------------------------------------------------------------------
# 2. Similarita' fra tag
# ----------------------------------------------------------------------
def tag_similarity(all_probs: list[np.ndarray]) -> np.ndarray:
    """
    Matrice 8x8 di similarita' fra emozioni, stimata sulla collezione.

    Impila le distribuzioni di tutti i segmenti di tutti i documenti e calcola
    il coseno fra le colonne: due emozioni sono tanto piu' simili quanto piu'
    il tagger le attiva sugli stessi segmenti. E' una misura di ridondanza
    percepita dal modello, non una distanza semantica imposta a priori.
    """
    M = np.vstack(all_probs)                       # (n_segmenti_totali, 8)
    M = M - M.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(M, axis=0, keepdims=True) + 1e-9
    S = (M / norms).T @ (M / norms)
    S = np.clip(S, 0.0, 1.0)
    np.fill_diagonal(S, 1.0)
    return S.astype(np.float32)


def similarity_from_confusion(cm: np.ndarray) -> np.ndarray:
    """
    Variante: similarita' derivata dalla matrice di confusione del tagger.
    Due classi sono simili se il modello le scambia fra loro.
    """
    cm = np.asarray(cm, dtype=np.float64)
    n = cm.shape[0]
    S = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            num = cm[i, j] + cm[j, i]
            den = cm[i, i] + cm[j, j] + num + 1e-9
            S[i, j] = S[j, i] = float(num / den)
    return S


# ----------------------------------------------------------------------
# 3. Maximal Marginal Relevance
# ----------------------------------------------------------------------
def mmr_select(tf: dict[str, float], sim: np.ndarray,
               k: int = config.MAX_TAGS,
               lam: float = config.MMR_LAMBDA,
               min_weight: float = config.MIN_TAG_WEIGHT) -> list[str]:
    """
    Seleziona fino a `k` tag fini massimizzando, a ogni passo:

        MMR(t) = lam * Sim_item(t, d) - (1 - lam) * max_{t' in T} Sim_tag(t', t)

    dove Sim_item e' la frequenza normalizzata del tag nel documento e Sim_tag
    e' la similarita' empirica fra tag. Con lam = 1 conta solo la rilevanza,
    con lam = 0 solo la novita'.
    """
    fine = np.array([tf.get(e, 0.0) for e in config.EMOTIONS], dtype=np.float32)
    total = float(fine.sum()) or 1.0
    rel = fine / total

    candidates = [i for i in range(len(config.EMOTIONS)) if rel[i] >= min_weight]
    if not candidates:
        candidates = [int(np.argmax(rel))]

    selected: list[int] = []
    while candidates and len(selected) < k:
        best, best_score = None, -np.inf
        for i in candidates:
            redundancy = max((sim[i, j] for j in selected), default=0.0)
            score = lam * rel[i] - (1.0 - lam) * redundancy
            if score > best_score:
                best, best_score = i, score
        if best is None or (selected and best_score <= 0):
            break
        selected.append(best)
        candidates.remove(best)

    return [config.EMOTIONS[i] for i in selected]


def top_valence(tf: dict[str, float]) -> str:
    """Livello di valenza dominante, usato come tag grosso del documento."""
    return max(config.VALENCE_LEVELS, key=lambda v: tf.get(v, 0.0))
