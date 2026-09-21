"""
Interrogazione e ranking.

Corrisponde alle fasi di *User Interaction* e *Ranking*: query input, query
transformation (espansione), scoring e result output (snippet).

Il problema principale e' il vocabulary mismatch: l'utente digita "cliente
insoddisfatto" oppure "nervoso", mentre l'indice contiene otto etichette
inglesi. La query transformation risolve il disallineamento espandendo i
termini della query nei tag d'indice tramite un thesaurus di dominio, che
include anche i tre livelli di valenza: "negativo" si espande nell'insieme
delle emozioni negative.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np

from src import config
from src.indexing import VoiceTagIndex

# ----------------------------------------------------------------------
# Thesaurus di dominio: termine di query -> tag d'indice pesati.
# Costruito sul lessico che un operatore userebbe parlando di chiamate.
# ----------------------------------------------------------------------
THESAURUS: dict[str, dict[str, float]] = {}


def _add(terms: str, mapping: dict[str, float]):
    for t in terms.split():
        THESAURUS[t] = mapping


# livello grosso della tassonomia -> espansione nei tag fini
for _level, _members in config.VALENCE.items():
    THESAURUS[_level] = {e: 1.0 for e in _members}
THESAURUS["negative"] = THESAURUS["negativo"]
THESAURUS["positive"] = THESAURUS["positivo"]
THESAURUS["neutral_level"] = THESAURUS["neutro"]

# i nomi dei tag valgono come query dirette
for _e in config.EMOTIONS:
    THESAURUS[_e] = {_e: 1.0}

_add("arrabbiato arrabbiata arrabbiati rabbia furioso furiosa furia irritato "
     "irritata alterato alterata infuriato incavolato angry anger mad furious "
     "urla urlare urla grida gridare aggressivo aggressiva escalation",
     {"angry": 1.0})

_add("nervoso nervosa nervosismo agitato agitata teso tesa",
     {"angry": 0.6, "fearful": 0.5})

_add("insoddisfatto insoddisfatta insoddisfazione scontento scontenta deluso "
     "delusa delusione lamentela lamentele reclamo reclami protesta "
     "dissatisfied complaint",
     {"angry": 0.6, "disgust": 0.7, "sad": 0.5})

_add("frustrato frustrata frustrazione esasperato esasperata frustrated",
     {"angry": 0.7, "disgust": 0.5, "sad": 0.4})

_add("disgustato disgusto schifo infastidito infastidita seccato seccata "
     "fastidio sdegnato indignato disgust disgusted annoyed",
     {"disgust": 1.0})

_add("triste tristezza abbattuto abbattuta sconfortato avvilito depresso "
     "rassegnato rassegnata amareggiato sad sadness unhappy",
     {"sad": 1.0})

_add("spaventato spaventata paura timoroso impaurito ansioso ansiosa ansia "
     "preoccupato preoccupata apprensione allarmato panico fearful fear "
     "scared anxious worried",
     {"fearful": 1.0})

_add("urgente emergenza urgenza problema guasto",
     {"fearful": 0.5, "angry": 0.5})

_add("felice contento contenta soddisfatto soddisfatta entusiasta allegro "
     "gioia gioioso happy joy joyful satisfied pleased",
     {"happy": 1.0})

_add("calmo calma tranquillo tranquilla sereno serena rilassato pacato "
     "pacata composto calm relaxed quiet",
     {"calm": 1.0})

_add("neutrale neutro piatto indifferente normale ordinario routine "
     "informativo neutral flat",
     {"neutral": 1.0})

_add("sorpreso sorpresa stupito stupita meravigliato spiazzato incredulo "
     "surprised surprise astonished",
     {"surprised": 1.0})

_add("soddisfazione contentezza apprezzamento ringraziamento grato",
     {"happy": 0.9, "calm": 0.4})

_add("critico difficile problematico",
     {"angry": 0.6, "disgust": 0.5})

# parole vuote del dominio: non portano informazione emotiva
STOPWORDS = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "a", "da",
    "in", "con", "su", "per", "tra", "fra", "e", "ed", "o", "che", "chi",
    "cui", "non", "del", "della", "dei", "delle", "dello", "al", "alla",
    "cliente", "clienti", "chiamata", "chiamate", "conversazione",
    "conversazioni", "registrazione", "registrazioni", "utente", "utenti",
    "call", "chiamante", "molto", "poco", "era", "erano", "sono", "stato",
    "the", "a", "of", "and", "customer", "caller",
}


# ----------------------------------------------------------------------
# Query transformation
# ----------------------------------------------------------------------
def normalize(text: str) -> list[str]:
    """Tokenizzazione: minuscole, rimozione accenti e punteggiatura, stopping."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    tokens = re.findall(r"[a-z]+", text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def expand(query: str) -> tuple[dict[str, float], list[str]]:
    """
    Espande la query nei tag d'indice.
    Restituisce (tag pesati, termini non riconosciuti).
    """
    weights: dict[str, float] = {}
    unknown: list[str] = []
    for token in normalize(query):
        mapping = THESAURUS.get(token)
        if mapping is None:
            mapping = _fuzzy(token)
        if mapping is None:
            unknown.append(token)
            continue
        for tag, w in mapping.items():
            weights[tag] = weights.get(tag, 0.0) + w
    return weights, unknown


def _fuzzy(token: str) -> dict[str, float] | None:
    """Ripiego per varianti morfologiche non elencate: match per prefisso."""
    if len(token) < 5:
        return None
    for key, mapping in THESAURUS.items():
        if len(key) >= 5 and (key.startswith(token[:5]) or token.startswith(key[:5])):
            return mapping
    return None


def query_vector(weights: dict[str, float], index: VoiceTagIndex) -> np.ndarray:
    """Vettore query pesato con l'idf dell'indice e normalizzato L2."""
    v = np.zeros(len(index.vocab), dtype=np.float32)
    for i, tag in enumerate(index.vocab):
        if tag in weights:
            v[i] = weights[tag] * index.idf.get(tag, 0.0)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


# ----------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------
def search(query: str, index: VoiceTagIndex, k: int = 10) -> dict:
    """
    Ricerca completa: espansione, recupero dei candidati dalle posting list,
    scoring a coseno, ordinamento, generazione degli snippet.
    """
    weights, unknown = expand(query)
    if not weights:
        return {"query": query, "expansion": {}, "unknown": unknown, "results": [],
                "n_candidates": 0}

    q = query_vector(weights, index)
    candidates = index.candidates(list(weights))

    scored = []
    for cid in candidates:
        score = float(np.dot(q, index.vectors[cid]))
        if score <= 0:
            continue
        scored.append((cid, score))
    scored.sort(key=lambda x: -x[1])

    results = []
    for cid, score in scored[:k]:
        doc = index.docs[cid]
        seg, span = snippet(doc, weights)
        results.append({
            "conv_id": cid,
            "score": round(score, 4),
            "path": doc.get("path"),
            "duration": doc.get("duration"),
            "tags": doc.get("tags", []),
            "valence": doc.get("valence"),
            "tf": doc.get("tf", {}),
            "snippet_index": seg,
            "snippet_span": span,
            "scenario": doc.get("scenario"),
        })

    return {"query": query, "expansion": weights, "unknown": unknown,
            "results": results, "n_candidates": len(candidates)}


def snippet(doc: dict, weights: dict[str, float]) -> tuple[int, tuple[float, float]]:
    """
    Snippet = il segmento della registrazione che massimizza l'accordo con la
    query. E' l'equivalente acustico del riassunto testuale mostrato accanto a
    un risultato: il punto esatto da ascoltare per verificarne la rilevanza.
    """
    probs = doc.get("probs")
    spans = doc.get("spans", [])
    if probs is None or len(spans) == 0:
        return 0, (0.0, 0.0)
    w = np.zeros(len(config.EMOTIONS), dtype=np.float32)
    for tag, val in weights.items():
        if tag in config.EMOTION_TO_ID:
            w[config.EMOTION_TO_ID[tag]] = val
    scores = np.asarray(probs) @ w
    i = int(np.argmax(scores))
    return i, tuple(spans[i])


# ----------------------------------------------------------------------
# Baseline di confronto: filtro booleano su etichetta unica
# ----------------------------------------------------------------------
def boolean_search(query: str, index: VoiceTagIndex, k: int = 10) -> dict:
    """
    Sistema ingenuo: ogni conversazione riceve una sola etichetta (l'emozione
    con tf massima) e la ricerca e' un filtro di uguaglianza, senza punteggio.
    Serve come termine di paragone nella valutazione.
    """
    weights, unknown = expand(query)
    target = {t for t, w in weights.items() if w > 0}
    results = []
    for cid, doc in index.docs.items():
        tf = doc.get("tf", {})
        label = max(config.EMOTIONS, key=lambda e: tf.get(e, 0.0))
        if label in target:
            results.append({"conv_id": cid, "score": 1.0, "label": label,
                            "path": doc.get("path")})
    return {"query": query, "expansion": weights, "unknown": unknown,
            "results": results[:k], "n_candidates": len(results)}
