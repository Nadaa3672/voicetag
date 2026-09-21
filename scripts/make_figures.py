"""
Figure per la relazione.

    python -m scripts.make_figures

Produce in results/figures:
  fig1_architettura_idf.png   document frequency e idf dei tag
  fig2_similarita_tag.png     similarita' empirica fra tag (motivazione di MMR)
  fig3_confronto_run.png      confronto dei quattro run di retrieval
  fig4_timeline.png           profilo emotivo nel tempo di una conversazione
  fig5_confusione.png         confusione del tagger a 8 classi e a 3 livelli
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src import config, conversations
from src.indexing import VoiceTagIndex

# --- parametri grafici -------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2dd"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # ordine categorico fisso
SEQ = LinearSegmentedColormap.from_list("voicetag_seq", ["#f4f8fd", "#2a78d6", "#123c6b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK_SOFT, "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "figure.dpi": 160, "axes.spines.top": False, "axes.spines.right": False,
})


def _clean(ax, ygrid=True):
    ax.grid(axis="y" if ygrid else "x", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------
def fig_idf(index: VoiceTagIndex, path):
    tags = index.vocab
    df = [index.df.get(t, 0) for t in tags]
    idf = [index.idf[t] for t in tags]
    order = np.argsort(idf)
    tags = [tags[i] for i in order]
    df = [df[i] for i in order]
    idf = [idf[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, vals, title, xlabel in (
            (axes[0], df, "Document frequency", "conversazioni che contengono il tag"),
            (axes[1], idf, "Peso idf", "log(N / df)")):
        bars = ax.barh(tags, vals, color=SERIES[0], height=0.62, zorder=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + max(vals) * 0.02, b.get_y() + b.get_height() / 2,
                    f"{v:.2f}" if isinstance(v, float) else str(v),
                    va="center", fontsize=8.5, color=INK_SOFT)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, (max(vals) or 1.0) * 1.18)
        _clean(ax, ygrid=False)
    fig.suptitle("Statistiche dei tag sulla collezione", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_similarity(sim: np.ndarray, path):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(sim, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(len(config.EMOTIONS)), config.EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(len(config.EMOTIONS)), config.EMOTIONS)
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            if i == j:
                continue
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if sim[i, j] > 0.55 else INK_SOFT)
    ax.set_title("Similarita' empirica fra tag\n(coseno fra le attivazioni sui segmenti)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="similarita'")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_runs(evaluation: dict, path):
    runs = evaluation["runs"]
    labels = {"canoniche_ranked": "Query canoniche\nranking tf-idf",
              "canoniche_boolean": "Query canoniche\nfiltro booleano",
              "naturali_ranked": "Query naturali\ncon espansione",
              "naturali_senza_espansione": "Query naturali\nsenza espansione"}
    metrics = ["P@5", "P@10", "MAP"]
    names = [k for k in labels if k in runs]

    x = np.arange(len(metrics))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    for i, name in enumerate(names):
        vals = [runs[name][m] for m in metrics]
        pos = x + i * width - 0.4 + width / 2
        bars = ax.bar(pos, vals, width * 0.88, label=labels[name],
                      color=SERIES[i % len(SERIES)], zorder=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    f"{v:.2f}", ha="center", fontsize=8, color=INK_SOFT)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, max(max(runs[n][m] for m in metrics) for n in names) * 1.28)
    ax.set_ylabel("valore medio sulle query")
    ax.set_title("Confronto dei run di retrieval")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left")
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_timeline(index: VoiceTagIndex, manifest: list[dict], path, conv_id=None):
    doc = index.docs[conv_id] if conv_id else next(iter(index.docs.values()))
    cid = conv_id or next(iter(index.docs))
    truth = next((m for m in manifest if m["conv_id"] == cid), None)
    probs = np.asarray(doc["probs"]).T                      # (8, n_segmenti)
    spans = doc["spans"]
    centers = [(s + e) / 2 for s, e in spans]

    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    im = ax.imshow(probs, aspect="auto", cmap=SEQ, vmin=0, vmax=1,
                   extent=[spans[0][0], spans[-1][1], len(config.EMOTIONS) - 0.5, -0.5])
    ax.set_yticks(range(len(config.EMOTIONS)), config.EMOTIONS)
    ax.set_xlabel("tempo (s)")
    ax.grid(False)

    if truth:
        ax.set_ylim(len(config.EMOTIONS) - 0.5, -1.6)     # spazio per le etichette
        for turn in truth["turns"]:
            ax.axvline(turn["start"], color=INK, lw=0.9, alpha=0.35)
            ax.text(turn["start"] + 0.06, -0.7, turn["emotion"], fontsize=7,
                    color=INK_SOFT, rotation=30, ha="left", va="bottom")
        ax.set_title(f"{cid} - scenario \"{truth['scenario']}\"\n"
                     "colore = probabilita' assegnata dal tagger; "
                     "le linee separano i turni reali", pad=14)
    else:
        ax.set_title(f"{cid} - profilo emotivo nel tempo", pad=14)
    fig.colorbar(im, ax=ax, shrink=0.85, label="probabilita'")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_confusion(evaluation: dict, path):
    cm8 = np.array(evaluation["tagger"]["confusione_8"], dtype=float)
    cm3 = np.array(evaluation["tagger"]["confusione_3"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4),
                             gridspec_kw={"width_ratios": [2, 1]})
    for ax, cm, labels, title in (
            (axes[0], cm8, config.EMOTIONS,
             f"8 tag fini - accuratezza {evaluation['tagger']['accuratezza_8_classi']:.1%}"),
            (axes[1], cm3, config.VALENCE_LEVELS,
             f"3 livelli - accuratezza {evaluation['tagger']['accuratezza_3_livelli']:.1%}")):
        norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        im = ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predetto")
        ax.set_ylabel("reale")
        ax.set_title(title)
        ax.grid(False)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if norm[i, j] >= 0.02:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                            fontsize=7.5,
                            color="white" if norm[i, j] > 0.55 else INK_SOFT)
    fig.suptitle("Confusione del tagger sui turni delle conversazioni",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_sensitivity(sens: dict, path):
    """Efficienza contro efficacia nella potatura, e stabilita' rispetto a tau."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))

    # a) soglia di document frequency
    ax = axes[0]
    taus = [r["soglia_df"] for r in sens["soglia_df"]]
    for j, (key, label) in enumerate((("MAP", "MAP"), ("R-prec", "R-precision"))):
        ax.plot(taus, [r[key] for r in sens["soglia_df"]], marker="o", markersize=6,
                lw=2, color=SERIES[j], label=label, zorder=3)
    ax.set_xlabel("soglia di document frequency  $\\tau$")
    ax.set_ylabel("valore medio")
    ax.set_title("Stabilita' rispetto a $\\tau$")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9)
    _clean(ax)

    # b) potatura: lunghezza dell'indice contro MAP
    ax = axes[1]
    rows = sens["potatura_posting"]
    post = [r["posting"] for r in rows]
    maps = [r["MAP"] for r in rows]
    ax.plot(post, maps, marker="o", markersize=6, lw=2, color=SERIES[0], zorder=3)
    for r in rows:
        ax.annotate(f"{r['soglia_posting']:.2f}", (r["posting"], r["MAP"]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8, color=INK_SOFT)
    ax.set_xlabel("posting totali nell'indice")
    ax.set_ylabel("MAP")
    ax.set_title("Potatura: dimensione dell'indice contro efficacia")
    ax.set_ylim(min(maps) * 0.9, max(maps) * 1.08)
    _clean(ax)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    index = VoiceTagIndex.load()
    manifest = conversations.load_manifest()
    evaluation = json.loads((config.RESULTS_DIR / "evaluation.json").read_text(encoding="utf-8"))
    sim = np.load(config.RESULTS_DIR / "tag_similarity.npy")

    fig_idf(index, config.FIGURES_DIR / "fig1_architettura_idf.png")
    fig_similarity(sim, config.FIGURES_DIR / "fig2_similarita_tag.png")
    fig_runs(evaluation, config.FIGURES_DIR / "fig3_confronto_run.png")

    # una conversazione con traiettoria marcata, per rendere leggibile la figura
    pick = next((m["conv_id"] for m in manifest if m["scenario"] == "escalation"),
                manifest[0]["conv_id"])
    fig_timeline(index, manifest, config.FIGURES_DIR / "fig4_timeline.png", conv_id=pick)
    fig_confusion(evaluation, config.FIGURES_DIR / "fig5_confusione.png")

    sens_path = config.RESULTS_DIR / "sensitivity.json"
    if sens_path.exists():
        sens = json.loads(sens_path.read_text(encoding="utf-8"))
        fig_sensitivity(sens, config.FIGURES_DIR / "fig6_sensibilita.png")
    print(f"Figure salvate in {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
