"""
VoiceTag - configurazione globale.

Raccoglie in un unico punto: percorsi, tassonomia dei tag emozionali (due livelli),
parametri audio, parametri di segmentazione e parametri del modello di retrieval.
"""
from pathlib import Path

# ----------------------------------------------------------------------
# Percorsi
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAVDESS_DIR = DATA_DIR / "ravdess"          # .wav estratti di RAVDESS
CONV_DIR = DATA_DIR / "conversations"       # pseudo-conversazioni generate
INDEX_DIR = DATA_DIR / "index"              # indice inverso serializzato
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CHECKPOINT = RESULTS_DIR / "best_cnn.pt"    # tagger acustico pre-addestrato

RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"

# ----------------------------------------------------------------------
# Tassonomia dei tag - livello fine (8 emozioni RAVDESS)
# Nome file RAVDESS: "03-01-06-01-02-01-12.wav"
#   [modality]-[vocal_channel]-[EMOZIONE]-[intensity]-[statement]-[repetition]-[ATTORE].wav
# ----------------------------------------------------------------------
EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}

# L'ordine deve coincidere con quello usato per addestrare il tagger.
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
EMOTION_TO_ID = {e: i for i, e in enumerate(EMOTIONS)}
ID_TO_EMOTION = {i: e for e, i in EMOTION_TO_ID.items()}

# ----------------------------------------------------------------------
# Tassonomia dei tag - livello grosso (3 livelli di valenza)
#
# I tag grossi non richiedono un secondo modello: si ottengono sommando le
# probabilita' softmax delle emozioni che appartengono allo stesso gruppo.
# Nota: 'surprised' ha valenza ambigua in letteratura; in RAVDESS le clip
# corrispondono a sorpresa piacevole, quindi e' assegnata a 'positivo'.
# ----------------------------------------------------------------------
VALENCE = {
    "negativo": ["angry", "disgust", "fearful", "sad"],
    "neutro":   ["neutral", "calm"],
    "positivo": ["happy", "surprised"],
}
VALENCE_LEVELS = ["negativo", "neutro", "positivo"]
EMOTION_TO_VALENCE = {e: v for v, es in VALENCE.items() for e in es}

# Vocabolario completo dell'indice: 8 tag fini + 3 tag grossi.
TAG_VOCAB = EMOTIONS + VALENCE_LEVELS

# ----------------------------------------------------------------------
# Parametri audio (identici a quelli con cui il tagger e' stato addestrato)
# ----------------------------------------------------------------------
SAMPLE_RATE = 16000
DURATION = 3.0          # durata della finestra di analisi, in secondi
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

# ----------------------------------------------------------------------
# Segmentazione della conversazione
# ----------------------------------------------------------------------
WINDOW_SEC = 3.0        # ampiezza della finestra (vincolata: e' la durata su cui
                        # il tagger e' stato addestrato)
HOP_SEC = 1.0           # avanzamento fra finestre consecutive
MIN_TAIL_SEC = 1.0      # code piu' corte di cosi' vengono scartate

# ----------------------------------------------------------------------
# Parametri di tagging / indicizzazione
# ----------------------------------------------------------------------
DF_THRESHOLD = 0.15       # tf normalizzata minima perche' un tag conti nella df
POSTING_THRESHOLD = 0.02  # tf normalizzata minima perche' un documento entri
                          # nella posting list di un tag: la softmax non e' mai
                          # esattamente zero, quindi senza potatura ogni lista
                          # conterrebbe l'intera collezione e l'indice non
                          # ridurrebbe nulla
MMR_LAMBDA = 0.7        # bilanciamento rilevanza / non-ridondanza in MMR
MAX_TAGS = 4            # numero massimo di tag fini mostrati per conversazione
MIN_TAG_WEIGHT = 0.05   # peso minimo perche' un tag sia candidato

# ----------------------------------------------------------------------
# Generazione delle pseudo-conversazioni
# ----------------------------------------------------------------------
# Solo attori MAI visti dal tagger in addestramento: la valutazione del
# sistema di retrieval resta speaker-independent.
HELDOUT_ACTORS = [1, 2, 3, 4, 5, 6, 21, 22]
GAP_SEC = 0.25           # silenzio inserito fra due turni della conversazione
CLIPS_PER_CONV = (8, 13)  # numero di turni per conversazione (min, max): le clip
                          # RAVDESS durano circa 1,2 s una volta tolto il silenzio,
                          # quindi servono almeno otto turni perche' il documento
                          # abbia abbastanza finestre da giustificare una tf
MAX_CLIP_REUSE = 6       # quante volte una stessa clip puo' comparire nel corpus
N_CONVERSATIONS = 150

SEED = 42
