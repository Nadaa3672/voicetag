"""
Modulo di tagging acustico.

Corrisponde alla fase di *Text Transformation* dell'architettura classica di un
motore di ricerca: trasforma il documento grezzo (la registrazione) nei termini
d'indice. Qui i termini non vengono estratti da un testo ma prodotti da un
classificatore sul segnale, ed hanno un peso continuo invece che binario.

Pipeline per una registrazione:
    segmentazione in finestre  ->  log-Mel + delta + delta-delta
    ->  CNN  ->  distribuzione softmax su 8 emozioni per finestra
    ->  aggregazione su 3 livelli di valenza
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch

from src import config
from src.model_cnn import load_checkpoint


# ----------------------------------------------------------------------
# Segmentazione
# ----------------------------------------------------------------------
def segment_signal(y: np.ndarray, sr: int = config.SAMPLE_RATE,
                   window: float = config.WINDOW_SEC,
                   hop: float = config.HOP_SEC) -> list[tuple[float, float, np.ndarray]]:
    """
    Divide il segnale in finestre sovrapposte di durata fissa.

    La sovrapposizione al 50% evita che un cambio di stato emotivo cada
    esattamente sul confine fra due finestre e venga cosi' diluito in entrambe.
    Restituisce una lista di (inizio, fine, campioni) in secondi.
    """
    win_len = int(window * sr)
    hop_len = int(hop * sr)
    segments = []

    if len(y) <= win_len:
        return [(0.0, len(y) / sr, _fix_length(y, win_len))]

    start = 0
    while start + win_len <= len(y):
        segments.append((start / sr, (start + win_len) / sr, y[start:start + win_len]))
        start += hop_len

    tail = len(y) - (start - hop_len) - win_len          # coda non coperta
    if tail >= config.MIN_TAIL_SEC * sr:
        chunk = y[len(y) - win_len:]
        segments.append(((len(y) - win_len) / sr, len(y) / sr, chunk))
    return segments


def _fix_length(y: np.ndarray, target: int) -> np.ndarray:
    """Riempie con zeri centrati o ritaglia al centro, come in addestramento."""
    if len(y) < target:
        pad = target - len(y)
        left = pad // 2
        return np.pad(y, (left, pad - left), mode="constant").astype(np.float32)
    if len(y) > target:
        start = (len(y) - target) // 2
        return y[start:start + target].astype(np.float32)
    return y.astype(np.float32)


# ----------------------------------------------------------------------
# Feature
# ----------------------------------------------------------------------
def log_mel(y: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Spettrogramma log-Mel in dB, riferito al massimo della finestra."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=config.N_MELS,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def to_tensor(mels: np.ndarray, mean: float, std: float) -> torch.Tensor:
    """
    Standardizza con le statistiche del training set e impila i tre canali
    (energia, velocita' di variazione, accelerazione).
    """
    base = ((mels - mean) / (std + 1e-6)).astype(np.float32)   # (N, H, W)
    d1 = np.gradient(base, axis=2).astype(np.float32)
    d2 = np.gradient(d1, axis=2).astype(np.float32)
    return torch.from_numpy(np.stack([base, d1, d2], axis=1))  # (N, 3, H, W)


# ----------------------------------------------------------------------
# Tagger
# ----------------------------------------------------------------------
class EmotionTagger:
    """
    Incapsula il modello pre-addestrato ed espone l'inferenza a livello di
    segmento. Non prende mai una decisione secca: restituisce la distribuzione
    di probabilita', che a valle diventa il peso dei tag nell'indice.
    """

    def __init__(self, checkpoint: Path | None = None, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.mean, self.std = load_checkpoint(
            checkpoint or config.CHECKPOINT, device=self.device)

    # -- livello segmento ------------------------------------------------
    @torch.no_grad()
    def tag_segments(self, y: np.ndarray, sr: int = config.SAMPLE_RATE,
                     batch_size: int = 32) -> tuple[np.ndarray, list[tuple[float, float]]]:
        """
        Restituisce (probs, spans) dove probs ha forma (n_segmenti, 8) e spans
        contiene gli estremi temporali di ogni finestra, usati per gli snippet.
        """
        segments = segment_signal(y, sr)
        spans = [(s, e) for s, e, _ in segments]
        mels = np.stack([log_mel(chunk, sr) for _, _, chunk in segments])
        x = to_tensor(mels, self.mean, self.std)

        out = []
        for i in range(0, len(x), batch_size):
            logits = self.model(x[i:i + batch_size].to(self.device))
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float32), spans

    def tag_file(self, path, batch_size: int = 32):
        y, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
        return self.tag_segments(y, config.SAMPLE_RATE, batch_size=batch_size)


# ----------------------------------------------------------------------
# Aggregazione sui due livelli della tassonomia
# ----------------------------------------------------------------------
def to_valence(probs: np.ndarray) -> np.ndarray:
    """
    Proietta le probabilita' delle 8 emozioni sui 3 livelli di valenza sommando
    i membri di ciascun gruppo. Non serve un secondo modello: la proiezione e'
    una somma di probabilita' di eventi mutuamente esclusivi, quindi resta una
    distribuzione valida.
    """
    out = np.zeros((probs.shape[0], len(config.VALENCE_LEVELS)), dtype=np.float32)
    for j, level in enumerate(config.VALENCE_LEVELS):
        idx = [config.EMOTION_TO_ID[e] for e in config.VALENCE[level]]
        out[:, j] = probs[:, idx].sum(axis=1)
    return out
