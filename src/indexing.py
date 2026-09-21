"""
Creazione dell'indice.

Corrisponde alla fase di *Index Creation* dell'architettura di un motore di
ricerca: document statistics, weighting e inversion.

Il vocabolario d'indice e' costituito dagli 8 tag fini. I tre livelli di
valenza non vengono indicizzati come termini autonomi - sarebbero la somma dei
precedenti, e conterebbero due volte la stessa evidenza - ma sono gestiti in
fase di interrogazione come espansione della query, che e' il loro ruolo
naturale: una gerarchia di concetti sopra la folksonomia dei tag.

Pesatura: w(t,d) = log(1 + tf(t,d)) * idf(t), con normalizzazione L2 del
vettore documento. La tf e' continua (somma di probabilita'), quindi si usa
log(1 + tf) al posto di 1 + log(tf), che sarebbe negativa per tf < 1.
"""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np

from src import config


class VoiceTagIndex:
    """Indice inverso con pesatura tf-idf sui tag emozionali."""

    def __init__(self):
        self.docs: dict[str, dict] = {}          # conv_id -> metadati
        self.postings: dict[str, list] = {}      # tag -> [(conv_id, peso, seg)]
        self.idf: dict[str, float] = {}
        self.df: dict[str, int] = {}             # document frequency (su soglia)
        self.vectors: dict[str, np.ndarray] = {} # conv_id -> vettore L2-normalizzato
        self.tag_sim: np.ndarray | None = None
        self.prior: np.ndarray | None = None     # prior del tagger, per calibrare
                                                 # le registrazioni caricate a caldo
        self.vocab: list[str] = list(config.EMOTIONS)

    # ------------------------------------------------------------------
    @property
    def n_docs(self) -> int:
        return len(self.docs)

    # ------------------------------------------------------------------
    def build(self, documents: list[dict], tag_sim: np.ndarray | None = None,
              df_threshold: float = config.DF_THRESHOLD,
              posting_threshold: float = config.POSTING_THRESHOLD,
              prior: np.ndarray | None = None) -> "VoiceTagIndex":
        """
        `documents` e' una lista di dict con almeno:
            conv_id, path, duration, tf (dict tag->float), spans, probs (ndarray),
            e opzionalmente scenario / trajectory (solo per la valutazione).
        """
        self.docs = {}
        self.tag_sim = tag_sim
        if prior is not None:
            self.prior = prior
        n = len(documents)

        # --- document statistics: document frequency -------------------
        df = {t: 0 for t in self.vocab}
        for d in documents:
            total = sum(d["tf"].get(e, 0.0) for e in self.vocab) or 1.0
            for t in self.vocab:
                if d["tf"].get(t, 0.0) / total >= df_threshold:
                    df[t] += 1

        # --- weighting: idf --------------------------------------------
        self.idf = {t: (math.log(n / df[t]) if df[t] > 0 else 0.0) for t in self.vocab}

        # --- inversion --------------------------------------------------
        # Il vettore documento conserva tutti i tag con tf > 0; la posting list
        # contiene invece solo i documenti in cui il tag ha un peso non
        # trascurabile, altrimenti la lista coinciderebbe con l'intera
        # collezione e l'inversione non porterebbe alcun guadagno.
        self.df = df
        self.postings = {t: [] for t in self.vocab}
        for d in documents:
            total = sum(d["tf"].get(e, 0.0) for e in self.vocab) or 1.0
            vec = np.zeros(len(self.vocab), dtype=np.float32)
            for i, t in enumerate(self.vocab):
                tf = d["tf"].get(t, 0.0)
                if tf > 0:
                    vec[i] = math.log(1.0 + tf) * self.idf[t]
            norm = float(np.linalg.norm(vec)) or 1.0
            vec = vec / norm

            cid = d["conv_id"]
            self.vectors[cid] = vec
            self.docs[cid] = {k: v for k, v in d.items() if k != "probs"}
            self.docs[cid]["probs"] = d["probs"]

            for i, t in enumerate(self.vocab):
                if vec[i] > 0 and d["tf"].get(t, 0.0) / total >= posting_threshold:
                    self.postings[t].append((cid, float(vec[i]), int(d["best_seg"][t])))

        for t in self.postings:
            self.postings[t].sort(key=lambda x: -x[1])
        return self

    # ------------------------------------------------------------------
    def candidates(self, tags: list[str]) -> set[str]:
        """Unione delle posting list dei tag della query: i soli documenti da valutare."""
        out: set[str] = set()
        for t in tags:
            out.update(cid for cid, _, _ in self.postings.get(t, []))
        return out

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "n_documenti": self.n_docs,
            "vocabolario": len(self.vocab),
            "posting_totali": sum(len(v) for v in self.postings.values()),
            "idf": {t: round(v, 3) for t, v in self.idf.items()},
            "df": dict(self.df),
            "lunghezza_posting": {t: len(v) for t, v in self.postings.items()},
            "segmenti_per_documento": round(
                float(np.mean([len(d.get("spans", [])) for d in self.docs.values()])), 2)
            if self.docs else 0.0,
        }

    # ------------------------------------------------------------------
    def save(self, path: Path | None = None):
        path = Path(path or config.INDEX_DIR)
        path.mkdir(parents=True, exist_ok=True)
        probs = {cid: d.pop("probs") for cid, d in self.docs.items()}
        np.savez_compressed(path / "segment_probs.npz", **probs)
        with open(path / "index.pkl", "wb") as f:
            pickle.dump({"docs": self.docs, "postings": self.postings,
                         "idf": self.idf, "df": self.df, "vectors": self.vectors,
                         "tag_sim": self.tag_sim, "prior": self.prior,
                         "vocab": self.vocab}, f)
        for cid, p in probs.items():                 # ripristina lo stato in memoria
            self.docs[cid]["probs"] = p
        (path / "stats.json").write_text(
            json.dumps(self.stats(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | None = None) -> "VoiceTagIndex":
        path = Path(path or config.INDEX_DIR)
        idx = cls()
        with open(path / "index.pkl", "rb") as f:
            blob = pickle.load(f)
        idx.docs = blob["docs"]
        idx.postings = blob["postings"]
        idx.idf = blob["idf"]
        idx.df = blob.get("df", {})
        idx.vectors = blob["vectors"]
        idx.tag_sim = blob["tag_sim"]
        idx.prior = blob.get("prior")
        idx.vocab = blob["vocab"]
        probs = np.load(path / "segment_probs.npz")
        for cid in idx.docs:
            if cid in probs:
                idx.docs[cid]["probs"] = probs[cid]
        return idx
