"""
Generatore di un finto RAVDESS per il collaudo della pipeline.

Serve solo a verificare che il codice giri end-to-end in un ambiente privo del
corpus reale: produce file con la stessa convenzione di nome e un segnale
vocale sintetico (modello sorgente-filtro) i cui parametri prosodici variano
per emozione. Non va usato per produrre risultati.

    python -m tests.make_fake_ravdess
"""
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

SR = 16000
OUT = Path(__file__).resolve().parents[1] / "data" / "ravdess"

# f0 medio, escursione di f0, velocita' di eloquio, energia -> per emozione
PROFILE = {
    "01": (120, 8, 1.0, 0.35),   # neutral
    "02": (110, 6, 0.9, 0.30),   # calm
    "03": (190, 45, 1.2, 0.70),  # happy
    "04": (100, 10, 0.8, 0.25),  # sad
    "05": (210, 60, 1.4, 0.95),  # angry
    "06": (200, 55, 1.3, 0.55),  # fearful
    "07": (130, 25, 0.9, 0.50),  # disgust
    "08": (230, 70, 1.3, 0.80),  # surprised
}
FORMANTS = [(700, 110), (1220, 120), (2600, 180)]


def _voiced(dur, f0, jitter, rng):
    n = int(dur * SR)
    t = np.arange(n) / SR
    contour = f0 + jitter * np.sin(2 * np.pi * 0.8 * t) + rng.normal(0, jitter * 0.2, n)
    phase = np.cumsum(2 * np.pi * np.maximum(contour, 60) / SR)
    src = 0.5 * np.sign(np.sin(phase)) * np.exp(-3 * (np.mod(phase, 2 * np.pi) / (2 * np.pi)))
    src += 0.05 * rng.normal(0, 1, n)
    out = np.zeros(n)
    for fc, bw in FORMANTS:
        r = np.exp(-np.pi * bw / SR)
        th = 2 * np.pi * fc / SR
        out += lfilter([1.0], [1.0, -2 * r * np.cos(th), r ** 2], src)
    return out / (np.max(np.abs(out)) + 1e-9)


def synth(emotion_code, rng):
    f0, jit, rate, energy = PROFILE[emotion_code]
    total = 0.0
    parts = []
    for _ in range(int(rng.integers(3, 6))):                 # sillabe
        d = float(rng.uniform(0.12, 0.28)) / rate
        parts.append(_voiced(d, f0 * rng.uniform(0.9, 1.1), jit, rng))
        parts.append(np.zeros(int(rng.uniform(0.03, 0.09) * SR)))
        total += d
    y = np.concatenate(parts) * energy
    env = np.linspace(1.0, 0.75, len(y))
    return (y * env).astype(np.float32)


def main(n_actors=24, reps=2):
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)
    count = 0
    for actor in range(1, n_actors + 1):
        adir = OUT / f"Actor_{actor:02d}"
        adir.mkdir(exist_ok=True)
        for code in PROFILE:
            n = reps if code != "01" else max(1, reps // 2)   # neutral ha meta' clip
            for r in range(n):
                for stmt in (1, 2):
                    y = synth(code, rng)
                    name = f"03-01-{code}-01-{stmt:02d}-{r+1:02d}-{actor:02d}.wav"
                    sf.write(adir / name, y, SR)
                    count += 1
    print(f"Generate {count} clip sintetiche in {OUT}")


if __name__ == "__main__":
    main()
