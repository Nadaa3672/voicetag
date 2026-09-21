# VoiceTag

**Indicizzazione e ricerca di conversazioni vocali mediante tag emozionali generati automaticamente**

Progetto per il corso di *Sistemi Intelligenti per Internet* — Università degli Studi Roma Tre.

---

## Che cosa fa

VoiceTag è un motore di ricerca per registrazioni vocali. Le conversazioni non hanno testo
associato, quindi non sono ricercabili con le tecniche classiche di Information Retrieval:
il sistema genera automaticamente i termini d'indice analizzando il segnale vocale, e su
quei termini costruisce un indice inverso interrogabile in linguaggio naturale.

```
registrazione → segmentazione → tagging acustico → tag pesati → indice inverso
                                                                      ↓
                    query in linguaggio naturale → espansione → ranking → snippet
```

Una query come *"clienti insoddisfatti"* restituisce una lista **ordinata** di
conversazioni, ciascuna con i tag assegnati e il punto esatto della registrazione da
ascoltare.

## Perché non è un classificatore con un filtro

La scelta progettuale centrale è che il sistema **ordina invece di classificare**:

* una conversazione è un documento **multi-segmento**, non una clip isolata: da qui una
  *term frequency* reale per ogni tag;
* i tag sono **pesati** con le probabilità del classificatore, non ridotti a un'etichetta
  secca: l'incertezza del modello si traduce in un peso basso e quindi in una posizione più
  bassa nel ranking, non in un errore netto;
* la pesatura è **tf-idf** e il ranking è a **coseno**, con recupero dei candidati dalle
  posting list;
* i tag mostrati sono scelti con **Maximal Marginal Relevance**, per non assegnare
  contemporaneamente tag che il tagger confonde sistematicamente;
* la **tag expansion** tramite thesaurus di dominio risolve il *vocabulary mismatch* fra il
  lessico dell'utente e le etichette dell'indice.

La valutazione quantifica ciascuna di queste scelte rispetto alla sua alternativa ingenua.

## Struttura

```
src/
  config.py         tassonomia dei tag (8 emozioni + 3 livelli di valenza), parametri
  model_cnn.py      architettura del tagger acustico
  tagger.py         segmentazione, feature log-Mel, inferenza per segmento
  conversations.py  generazione della collezione e giudizi di rilevanza
  tagging.py        soft term frequency, similarità fra tag, selezione MMR
  indexing.py       document statistics, pesatura tf-idf, indice inverso
  retrieval.py      thesaurus, espansione della query, ranking, snippet
  evaluation.py     P@k, Recall, F1, MAP e i run di confronto
scripts/
  build_collection.py   pipeline completa dai file audio all'indice
  run_eval.py           valutazione e confronto dei run
  make_figures.py       figure per la relazione
app/
  streamlit_app.py      interfaccia web
notebooks/
  00_pipeline_colab.ipynb   esecuzione completa su Google Colab
```

## Esecuzione

Su Colab: aprire `notebooks/00_pipeline_colab.ipynb` ed eseguire le celle in ordine.

In locale:

```bash
pip install -r requirements.txt

# RAVDESS in data/ravdess/Actor_01 ... Actor_24
python -m scripts.build_collection --n 200
python -m scripts.run_eval
python -m scripts.make_figures

streamlit run app/streamlit_app.py
```

Per un collaudo senza il corpus reale, `python -m tests.make_fake_ravdess` genera clip
sintetiche con la stessa convenzione di nome: serve solo a verificare che la pipeline giri.

## Dati

[RAVDESS](https://zenodo.org/record/1188976) — Ryerson Audio-Visual Database of Emotional
Speech and Song, porzione *speech*: 1440 clip, 24 attori, 8 emozioni, due frasi
semanticamente neutre. Licenza CC BY-NC-SA 4.0.

Le conversazioni indicizzate sono costruite concatenando clip di uno stesso attore secondo
traiettorie emotive prestabilite, usando **solo attori esclusi dall'addestramento del
tagger**: la valutazione resta speaker-independent e i giudizi di rilevanza sono noti per
costruzione.
