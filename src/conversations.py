"""
Costruzione della collezione di pseudo-conversazioni.

RAVDESS contiene clip isolate di 3-4 secondi: non sono conversazioni, e un
sistema di retrieval che indicizza una sola etichetta per clip degenera in un
filtro booleano. Qui le clip vengono concatenate in registrazioni piu' lunghe
che seguono una traiettoria emotiva prestabilita, ottenendo:

  1. documenti multi-segmento, quindi una *term frequency* reale per ogni tag;
  2. un ground truth noto per costruzione, che fornisce gratuitamente i
     giudizi di rilevanza necessari alla valutazione IR.

Le conversazioni usano soltanto attori mai visti dal tagger in addestramento
(config.HELDOUT_ACTORS), cosi' la valutazione resta speaker-independent.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src import config

# ----------------------------------------------------------------------
# Scenari: traiettorie emotive ispirate a tipiche chiamate di assistenza.
# Ogni scenario e' una sequenza di emozioni; la lunghezza effettiva della
# conversazione viene estesa ripetendo gli stati centrali.
# ----------------------------------------------------------------------
SCENARIOS = {
    "escalation":      ["neutral", "neutral", "angry", "angry", "angry"],
    "de_escalation":   ["angry", "angry", "neutral", "calm", "calm"],
    "frustrazione":    ["neutral", "disgust", "angry", "angry", "sad"],
    "ansia":           ["fearful", "fearful", "neutral", "sad", "sad"],
    "risoluzione":     ["neutral", "happy", "happy", "calm", "calm"],
    "sorpresa":        ["neutral", "surprised", "surprised", "happy", "happy"],
    "routine":         ["neutral", "calm", "neutral", "calm", "neutral"],
    "rassegnazione":   ["sad", "sad", "neutral", "sad", "calm"],
    "reclamo_freddo":  ["calm", "disgust", "disgust", "neutral", "angry"],
    "panico":          ["fearful", "fearful", "surprised", "fearful", "sad"],
}


# ----------------------------------------------------------------------
# Indice di RAVDESS
# ----------------------------------------------------------------------
def index_ravdess(root: Path | None = None) -> pd.DataFrame:
    """Scansiona la cartella RAVDESS e restituisce un DataFrame delle clip."""
    root = Path(root or config.RAVDESS_DIR)
    rows = []
    for path in sorted(root.rglob("*.wav")):
        parts = path.stem.split("-")
        if len(parts) != 7:
            continue
        emotion = config.EMOTION_MAP.get(parts[2])
        if emotion is None:
            continue
        rows.append({
            "path": str(path),
            "emotion": emotion,
            "emotion_id": config.EMOTION_TO_ID[emotion],
            "intensity": int(parts[3]),
            "statement": int(parts[4]),
            "repetition": int(parts[5]),
            "actor": int(parts[6]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"Nessuna clip RAVDESS trovata sotto {root}")
    return df


# ----------------------------------------------------------------------
# Generazione
# ----------------------------------------------------------------------
def _expand_trajectory(states: list[str], length: int, rng) -> list[str]:
    """Adatta uno scenario alla lunghezza richiesta ripetendo gli stati centrali."""
    if length <= len(states):
        return states[:length]
    traj = list(states)
    while len(traj) < length:
        i = int(rng.integers(1, max(2, len(states) - 1)))
        traj.insert(i, states[i])
    return traj[:length]


def build_conversations(df: pd.DataFrame,
                        out_dir: Path | None = None,
                        n_conversations: int = config.N_CONVERSATIONS,
                        seed: int = config.SEED) -> pd.DataFrame:
    """
    Genera le pseudo-conversazioni e le salva come .wav.

    Ogni conversazione usa clip di un solo attore, cosi' la registrazione
    corrisponde a una sola voce come in una chiamata reale.
    Restituisce il manifest della collezione (una riga per conversazione).
    """
    import librosa

    out_dir = Path(out_dir or config.CONV_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    pool = df[df["actor"].isin(config.HELDOUT_ACTORS)].copy()
    if pool.empty:
        raise ValueError("Nessuna clip disponibile per gli attori held-out.")

    usage: dict[str, int] = {p: 0 for p in pool["path"]}
    scenario_names = list(SCENARIOS)
    actors = sorted(pool["actor"].unique())
    gap = np.zeros(int(config.GAP_SEC * config.SAMPLE_RATE), dtype=np.float32)

    manifest = []
    for i in range(n_conversations):
        scenario = scenario_names[i % len(scenario_names)]
        actor = int(actors[i % len(actors)])
        n_turns = int(rng.integers(config.CLIPS_PER_CONV[0], config.CLIPS_PER_CONV[1] + 1))
        trajectory = _expand_trajectory(SCENARIOS[scenario], n_turns, rng)

        audio_parts, turns, ok = [], [], True
        cursor = 0.0
        for emotion in trajectory:
            cand = pool[(pool["actor"] == actor) & (pool["emotion"] == emotion)]
            cand = cand[cand["path"].map(lambda p: usage[p] < config.MAX_CLIP_REUSE)]
            if cand.empty:                       # es. 'neutral' ha meta' delle clip
                cand = pool[(pool["actor"] == actor) & (pool["emotion"] == emotion)]
            if cand.empty:
                ok = False
                break
            row = cand.iloc[int(rng.integers(0, len(cand)))]
            usage[row["path"]] = usage.get(row["path"], 0) + 1

            y, _ = librosa.load(row["path"], sr=config.SAMPLE_RATE, mono=True)
            y, _ = librosa.effects.trim(y, top_db=25)
            if len(y) == 0:
                ok = False
                break
            audio_parts.append(y.astype(np.float32))
            dur = len(y) / config.SAMPLE_RATE
            turns.append({"emotion": emotion, "start": round(cursor, 3),
                          "end": round(cursor + dur, 3), "source": Path(row["path"]).name})
            cursor += dur + config.GAP_SEC

        if not ok or not audio_parts:
            continue

        signal = audio_parts[0]
        for part in audio_parts[1:]:
            signal = np.concatenate([signal, gap, part])
        peak = float(np.max(np.abs(signal))) or 1.0
        signal = (signal / peak * 0.95).astype(np.float32)

        conv_id = f"conv_{i:04d}"
        wav_path = out_dir / f"{conv_id}.wav"
        sf.write(wav_path, signal, config.SAMPLE_RATE)

        counts = {e: trajectory.count(e) for e in set(trajectory)}
        manifest.append({
            "conv_id": conv_id,
            "path": str(wav_path),
            "actor": actor,
            "scenario": scenario,
            "duration": round(len(signal) / config.SAMPLE_RATE, 2),
            "n_turns": len(turns),
            "trajectory": trajectory,
            "turns": turns,
            "true_counts": counts,
        })

    if not manifest:
        raise RuntimeError("Nessuna conversazione generata: controlla i dati RAVDESS.")

    mf = pd.DataFrame(manifest)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return mf


def load_manifest(out_dir: Path | None = None) -> list[dict]:
    out_dir = Path(out_dir or config.CONV_DIR)
    return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Ground truth per la valutazione
# ----------------------------------------------------------------------
def relevance_judgments(manifest: list[dict], min_turns: int = 2) -> dict[str, set]:
    """
    Giudizi di rilevanza derivati dalla costruzione della collezione.

    Una conversazione e' rilevante per il tag fine `e` se almeno `min_turns`
    dei suoi turni esprimono `e`; e' rilevante per un tag di valenza se almeno
    `min_turns` turni appartengono a quel livello. La soglia evita di
    considerare rilevante una conversazione in cui l'emozione compare una volta
    sola e di sfuggita.
    """
    judgments = {t: set() for t in config.TAG_VOCAB}
    for doc in manifest:
        counts = doc["true_counts"]
        for emotion, n in counts.items():
            if n >= min_turns:
                judgments[emotion].add(doc["conv_id"])
        for level, members in config.VALENCE.items():
            if sum(counts.get(e, 0) for e in members) >= min_turns:
                judgments[level].add(doc["conv_id"])
    return judgments
