"""
Pipeline completa: dai file RAVDESS all'indice interrogabile.

    python -m scripts.build_collection [--n 200] [--skip-generation]

Fasi (corrispondenti all'architettura di un motore di ricerca):
  1. Acquisizione   - indicizzazione di RAVDESS e generazione delle conversazioni
  2. Trasformazione - segmentazione, tagging acustico, soft term frequency, MMR
  3. Creazione indice - document statistics, pesatura tf-idf, inversione
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from src import config, conversations, tagging
from src.indexing import VoiceTagIndex
from src.tagger import EmotionTagger


def main(n_conversations: int, skip_generation: bool):
    t0 = time.time()

    # --- 1. acquisizione ------------------------------------------------
    if skip_generation:
        manifest = conversations.load_manifest()
        print(f"[1/3] Manifest esistente: {len(manifest)} conversazioni")
    else:
        df = conversations.index_ravdess()
        print(f"[1/3] RAVDESS indicizzato: {len(df)} clip, "
              f"{df['actor'].nunique()} attori")
        mf = conversations.build_conversations(df, n_conversations=n_conversations)
        manifest = conversations.load_manifest()
        print(f"      Generate {len(mf)} conversazioni "
              f"({mf['duration'].mean():.1f}s in media) dagli attori held-out "
              f"{config.HELDOUT_ACTORS}")

    # --- 2. trasformazione ----------------------------------------------
    tagger = EmotionTagger()
    print(f"[2/3] Tagger caricato su {tagger.device}")

    documents, all_probs = [], []
    for i, doc in enumerate(manifest, start=1):
        probs, spans = tagger.tag_file(doc["path"])
        all_probs.append(probs)
        documents.append({
            "conv_id": doc["conv_id"],
            "path": doc["path"],
            "duration": doc["duration"],
            "actor": doc["actor"],
            "scenario": doc["scenario"],
            "trajectory": doc["trajectory"],
            "true_counts": doc["true_counts"],
            "probs": probs,
            "spans": spans,
            "tf": tagging.soft_tf(probs),
        })
        if i % 25 == 0 or i == len(manifest):
            print(f"      tagging {i}/{len(manifest)}")

    sim = tagging.tag_similarity(all_probs)
    for d in documents:
        d["tags"] = tagging.mmr_select(d["tf"], sim)
        d["valence"] = tagging.top_valence(d["tf"])
        d["best_seg"] = {t: tagging.dominant_segment(d["probs"], t)
                         for t in config.TAG_VOCAB}

    # --- 3. creazione indice --------------------------------------------
    index = VoiceTagIndex().build(documents, tag_sim=sim)
    index.save()
    stats = index.stats()
    print(f"[3/3] Indice creato: {stats['n_documenti']} documenti, "
          f"{stats['posting_totali']} posting")
    print("      idf per tag:", stats["idf"])

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.RESULTS_DIR / "tag_similarity.npy", sim)
    (config.RESULTS_DIR / "collection_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Completato in {time.time() - t0:.1f}s")
    return index


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=config.N_CONVERSATIONS)
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()
    main(args.n, args.skip_generation)
