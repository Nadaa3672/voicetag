"""
VoiceTag - interfaccia web.

    streamlit run app/streamlit_app.py

Tre viste, che ricalcano i due processi di un motore di ricerca:
  * Ricerca      - processo di interrogazione (query, ranking, snippet)
  * Indicizzazione - processo di indicizzazione di una nuova registrazione
  * Analisi      - lettura aggregata della collezione (prospettiva call center)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, retrieval, tagging          # noqa: E402
from src.indexing import VoiceTagIndex              # noqa: E402
from src.tagger import EmotionTagger                # noqa: E402

st.set_page_config(page_title="VoiceTag", page_icon="🎙", layout="wide")

VALENCE_COLOR = {"negativo": "#c0392b", "neutro": "#7f8c8d", "positivo": "#2e8b57"}


# ----------------------------------------------------------------------
# Risorse condivise
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Caricamento dell'indice...")
def load_index():
    return VoiceTagIndex.load()


@st.cache_resource(show_spinner="Caricamento del tagger...")
def load_tagger():
    return EmotionTagger()


def tag_chip(tag: str, weight: float | None = None) -> str:
    color = VALENCE_COLOR.get(config.EMOTION_TO_VALENCE.get(tag, "neutro"), "#7f8c8d")
    label = tag if weight is None else f"{tag} {weight:.2f}"
    return (f"<span style='background:{color};color:white;border-radius:10px;"
            f"padding:2px 9px;margin-right:5px;font-size:0.8rem'>{label}</span>")


def profile_frame(tf: dict) -> pd.DataFrame:
    total = sum(tf.get(e, 0.0) for e in config.EMOTIONS) or 1.0
    return pd.DataFrame({"emozione": config.EMOTIONS,
                         "peso": [tf.get(e, 0.0) / total for e in config.EMOTIONS]}
                        ).set_index("emozione")


try:
    index = load_index()
except FileNotFoundError:
    st.error("Indice non trovato. Esegui prima `python -m scripts.build_collection`.")
    st.stop()

st.title("VoiceTag")
st.caption("Indicizzazione e ricerca di conversazioni vocali "
           "mediante tag emozionali generati automaticamente")

with st.sidebar:
    st.subheader("Collezione")
    st.metric("Conversazioni indicizzate", index.n_docs)
    st.metric("Tag nel vocabolario d'indice", len(index.vocab))
    st.caption("Peso idf per tag: un tag raro nella collezione discrimina di piu'.")
    st.dataframe(pd.DataFrame({"tag": list(index.idf),
                               "df": [len(index.postings[t]) for t in index.idf],
                               "idf": [round(v, 3) for v in index.idf.values()]}),
                 hide_index=True)

tab_search, tab_index, tab_stats = st.tabs(
    ["Ricerca", "Indicizza una registrazione", "Analisi della collezione"])

# ----------------------------------------------------------------------
# Ricerca
# ----------------------------------------------------------------------
with tab_search:
    col_q, col_k = st.columns([4, 1])
    query = col_q.text_input("Query", value="cliente arrabbiato",
                             placeholder="es. clienti insoddisfatti, esperienza negativa, tono sereno")
    k = col_k.number_input("Risultati", 1, 50, 10)

    if query.strip():
        out = retrieval.search(query, index, k=int(k))

        exp = out["expansion"]
        if exp:
            st.markdown("**Query espansa nei tag d'indice:** " +
                        " ".join(tag_chip(t, w) for t, w in sorted(
                            exp.items(), key=lambda x: -x[1])), unsafe_allow_html=True)
        if out["unknown"]:
            st.caption("Termini non riconosciuti dal thesaurus: " +
                       ", ".join(out["unknown"]))
        if not exp:
            st.warning("Nessun termine della query e' riconducibile a un tag emozionale.")
        else:
            st.caption(f"{out['n_candidates']} documenti recuperati dalle posting list "
                       f"su {index.n_docs} totali; mostrati i primi {len(out['results'])} "
                       f"per punteggio.")

        for r in out["results"]:
            with st.container(border=True):
                head, meta = st.columns([3, 2])
                head.markdown(f"**{r['conv_id']}** &nbsp; punteggio `{r['score']:.3f}`",
                              unsafe_allow_html=True)
                head.markdown(" ".join(tag_chip(t) for t in r["tags"]),
                              unsafe_allow_html=True)
                start, end = r["snippet_span"]
                meta.caption(f"durata {r['duration']}s · valenza **{r['valence']}** · "
                             f"snippet {start:.1f}s–{end:.1f}s")
                audio_path = Path(r["path"])
                if audio_path.exists():
                    meta.audio(str(audio_path), start_time=int(start))
                with st.expander("Profilo emozionale del documento"):
                    st.bar_chart(profile_frame(r["tf"]), height=180)

# ----------------------------------------------------------------------
# Indicizzazione di una nuova registrazione
# ----------------------------------------------------------------------
with tab_index:
    st.write("Carica una registrazione: il sistema la segmenta, la etichetta e "
             "la aggiunge all'indice della sessione corrente.")
    up = st.file_uploader("Registrazione", type=["wav", "flac", "ogg", "mp3"])
    if up is not None:
        tmp = ROOT / "data" / "uploads"
        tmp.mkdir(parents=True, exist_ok=True)
        dest = tmp / up.name
        dest.write_bytes(up.getbuffer())

        tagger = load_tagger()
        probs, spans = tagger.tag_file(dest)
        tf = tagging.soft_tf(probs)
        sim = index.tag_sim if index.tag_sim is not None else np.eye(len(config.EMOTIONS))
        tags = tagging.mmr_select(tf, sim)
        valence = tagging.top_valence(tf)

        st.audio(str(dest))
        st.markdown("**Tag assegnati (selezione MMR):** " +
                    " ".join(tag_chip(t) for t in tags), unsafe_allow_html=True)
        st.markdown(f"**Valenza dominante:** `{valence}`")

        timeline = pd.DataFrame(
            probs, columns=config.EMOTIONS,
            index=[f"{s:.1f}s" for s, _ in spans])
        st.caption("Distribuzione per segmento: ogni riga e' una finestra di "
                   f"{config.WINDOW_SEC:.0f}s, con passo {config.HOP_SEC:.1f}s.")
        st.line_chart(timeline, height=220)

        if st.button("Aggiungi all'indice della sessione"):
            doc = {"conv_id": dest.stem, "path": str(dest),
                   "duration": round(float(spans[-1][1]), 2), "probs": probs,
                   "spans": spans, "tf": tf, "tags": tags, "valence": valence,
                   "best_seg": {t: tagging.dominant_segment(probs, t)
                                for t in config.TAG_VOCAB}}
            existing = [dict(d) for d in index.docs.values()]
            index.build(existing + [doc], tag_sim=index.tag_sim)
            st.success(f"Indicizzata. La collezione contiene ora {index.n_docs} documenti.")

# ----------------------------------------------------------------------
# Analisi aggregata
# ----------------------------------------------------------------------
with tab_stats:
    st.write("Lettura aggregata della collezione indicizzata: e' la prospettiva "
             "che rende il sistema utile a chi gestisce un servizio di assistenza "
             "telefonica.")
    rows = []
    for cid, doc in index.docs.items():
        tf = doc.get("tf", {})
        total = sum(tf.get(e, 0.0) for e in config.EMOTIONS) or 1.0
        row = {"conv_id": cid, "valenza": doc.get("valence"),
               "durata": doc.get("duration")}
        row.update({e: tf.get(e, 0.0) / total for e in config.EMOTIONS})
        rows.append(row)
    df = pd.DataFrame(rows)

    c1, c2, c3 = st.columns(3)
    for col, level in zip((c1, c2, c3), config.VALENCE_LEVELS):
        share = (df["valenza"] == level).mean() if len(df) else 0.0
        col.metric(f"Conversazioni {level}", f"{share:.0%}")

    st.bar_chart(df[config.EMOTIONS].mean(), height=240)
    st.caption("Peso medio di ciascun tag sulla collezione.")
    st.dataframe(df, hide_index=True)
