"""
Script 13 — Poster-quality visualizations.
Dark-neutral background, red accent palette.
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import polars as pl
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import ANALYSIS_DIR, NETWORKS_DIR, PARQUET_DIR, PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "data" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
BG         = "#FAFAFA"   # near-white page background
BG2        = "#FFFFFF"   # panel background
RED        = "#C0392B"
RED2       = "#E74C3C"
RED_LIGHT  = "#E57373"
RED_PALE   = "#FFCDD2"
WHITE      = "#FFFFFF"
OFFWHITE   = "#FAFAFA"
DARK       = "#1a1a1a"   # text / axes
GREY       = "#757575"
GRIDCOL    = "#E0E0E0"

mpl.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG2,
    "axes.edgecolor":    "#CCCCCC",
    "axes.labelcolor":   DARK,
    "axes.titlecolor":   DARK,
    "xtick.color":       DARK,
    "ytick.color":       DARK,
    "text.color":        DARK,
    "grid.color":        GRIDCOL,
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    10,
    "legend.facecolor":  WHITE,
    "legend.edgecolor":  "#CCCCCC",
    "legend.labelcolor": DARK,
})

# ── club colours ──────────────────────────────────────────────────────────────
CLUB_COLOURS = {
    "KO":              "#E8C4C4",
    "PiS":             "#8B0000",
    "PSL-TD":          "#C0392B",
    "Lewica":          "#FF5252",
    "Polska2050":      "#FF8C69",
    "Polska2050-TD":   "#FFAB91",
    "Konfederacja":    "#5D0000",
    "Konfederacja_KP": "#7A0000",
    "Razem":           "#FF7675",
    "PSL":             "#D4826B",
    "Centrum":         "#F0A090",
    "niez.":           "#777777",
    "Demokracja":      "#B05050",
}

def cc(name: str) -> str:
    return CLUB_COLOURS.get(str(name), GREY)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Agreement network
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_network():
    print("  fig1: network graph …")
    agreement  = np.load(NETWORKS_DIR / "agreement_matrix.npy")
    copresence = np.load(NETWORKS_DIR / "copresence_matrix.npy")
    mp_ids     = np.load(NETWORKS_DIR / "mp_ids.npy")
    mps_df     = pl.read_parquet(PARQUET_DIR / "term10/mps.parquet")
    id_to_club = dict(zip(mps_df["mp_id"].to_list(), mps_df["club"].cast(pl.Utf8).to_list()))

    THRESH, MIN_COP = 0.70, 50
    N = len(mp_ids)
    G = nx.Graph()
    for i in range(N):
        G.add_node(int(mp_ids[i]), club=id_to_club.get(int(mp_ids[i]), "niez."))
    for i in range(N):
        for j in range(i + 1, N):
            if copresence[i, j] >= MIN_COP and agreement[i, j] >= THRESH:
                G.add_edge(int(mp_ids[i]), int(mp_ids[j]), weight=float(agreement[i, j]))

    print(f"    nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    pos = nx.spring_layout(G, seed=42, k=0.35, iterations=60, weight="weight")

    node_colours = [cc(G.nodes[n]["club"]) for n in G.nodes()]
    deg_c = nx.degree_centrality(G)
    node_sizes = [10 + 70 * deg_c[n] for n in G.nodes()]

    edge_weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    edge_alpha   = np.clip((edge_weights - THRESH) / (1.0 - THRESH), 0, 1) * 0.30

    fig, ax = plt.subplots(figsize=(14, 12), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    for idx, (u, v) in enumerate(G.edges()):
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=RED, alpha=float(edge_alpha[idx]) * 0.9, linewidth=0.35)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colours,
                           node_size=node_sizes, linewidths=0.4,
                           edgecolors="#000000", alpha=0.93)

    seen = {}
    for n in G.nodes():
        c = G.nodes[n]["club"]
        if c not in seen:
            seen[c] = cc(c)
    handles = [mpatches.Patch(color=col, label=club) for club, col in sorted(seen.items())]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5,
              framealpha=0.6, ncol=2, title="Klub", title_fontsize=9)

    ax.set_title(
        f"Sieć głosowań — próg zgodności {int(THRESH*100)}%\n"
        f"Kadencja X  |  {G.number_of_nodes()} posłów  ·  {G.number_of_edges():,} krawędzi",
        fontsize=13, pad=14, color=DARK,
    )
    fig.tight_layout(pad=0.5)
    out = FIG_DIR / "fig1_network.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Inter-party heatmap
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_heatmap():
    print("  fig2: party heatmap …")
    df    = pl.read_parquet(ANALYSIS_DIR / "party_correlation_matrix.parquet")
    clubs = sorted(df["club1"].cast(pl.Utf8).unique().to_list())
    n     = len(clubs)
    idx   = {c: i for i, c in enumerate(clubs)}
    mat   = np.full((n, n), np.nan)
    for row in df.iter_rows(named=True):
        i, j = idx.get(str(row["club1"])), idx.get(str(row["club2"]))
        if i is not None and j is not None:
            mat[i, j] = mat[j, i] = row["agreement_rate"]
    np.fill_diagonal(mat, 1.0)

    cmap = LinearSegmentedColormap.from_list(
        "rm", ["#FFE8E8", "#FF9999", "#E74C3C", "#C0392B", "#8B0000", "#3d0000"]
    )

    fig, ax = plt.subplots(figsize=(10, 8.5), facecolor=BG)
    ax.set_facecolor(BG)
    im = ax.imshow(mat, cmap=cmap, vmin=0.3, vmax=1.0, aspect="auto")

    short = [c.replace("Konfederacja", "Konf.").replace("Polska2050", "P2050") for c in clubs]
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=9)

    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                col = WHITE if mat[i, j] > 0.65 else DARK
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7.5, color=col)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Zgodność głosowań", color=DARK, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=DARK)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=DARK)

    ax.set_title("Macierz zgodności między klubami\nKadencja X Sejmu RP", fontsize=13, pad=12)
    fig.tight_layout(pad=1.0)
    out = FIG_DIR / "fig2_party_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Rebels
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_rebels():
    print("  fig3: rebels …")
    rebels     = pl.read_parquet(ANALYSIS_DIR / "rebels.parquet")
    centrality = pl.read_parquet(ANALYSIS_DIR / "centrality_rebels.parquet")

    top = (rebels.filter(pl.col("total_votes") >= 500)
           .sort("rebel_rate", descending=True).head(15))

    fig = plt.figure(figsize=(14, 6.5), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

    # left — bar chart
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(BG2)
    names   = [f"{r['first_name'][0]}. {r['last_name']}" for r in top.iter_rows(named=True)]
    rates   = top["rebel_rate"].to_list()
    colours = [cc(c) for c in top["club"].cast(pl.Utf8).to_list()]
    y = np.arange(len(names))
    bars = ax1.barh(y, rates, color=colours, edgecolor="#111", linewidth=0.4, height=0.65)
    for bar, rate in zip(bars, rates):
        ax1.text(rate + 0.007, bar.get_y() + bar.get_height() / 2,
                 f"{rate:.1%}", va="center", fontsize=8.5, color=DARK)
    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=9.5)
    ax1.set_xlabel("Wskaźnik buntownictwa", fontsize=10)
    ax1.set_title("Top 15 buntowników", fontsize=13)
    ax1.set_xlim(0, max(rates) * 1.25)
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.25)
    ax1.spines[["top", "right"]].set_visible(False)
    seen = {}
    for r in top.iter_rows(named=True):
        c = str(r["club"])
        if c not in seen: seen[c] = cc(c)
    handles = [mpatches.Patch(color=col, label=club) for club, col in seen.items()]
    ax1.legend(handles=handles, fontsize=7.5, loc="lower right", framealpha=0.4)

    # right — scatter
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(BG2)
    valid   = centrality.filter(pl.col("rebel_rate").is_not_null())
    xarr    = valid["eigenvector"].to_numpy()
    yarr    = valid["rebel_rate"].to_numpy()
    cols_sc = [cc(c) for c in valid["club"].cast(pl.Utf8).to_list()]
    ax2.scatter(xarr, yarr, c=cols_sc, s=20, alpha=0.7, linewidths=0)
    mask = ~np.isnan(xarr) & ~np.isnan(yarr)
    coef = np.polyfit(xarr[mask], yarr[mask], 1)
    xfit = np.linspace(xarr[mask].min(), xarr[mask].max(), 100)
    ax2.plot(xfit, np.polyval(coef, xfit), color=RED_PALE, linewidth=1.8, linestyle="--")
    ax2.set_xlabel("Eigenvector centrality", fontsize=10)
    ax2.set_ylabel("Wskaźnik buntownictwa", fontsize=10)
    ax2.set_title("Buntownicy vs pozycja w sieci\nρ = −0.78  (p < 10⁻¹⁰⁰)", fontsize=11)
    ax2.grid(alpha=0.25)
    ax2.spines[["top", "right"]].set_visible(False)
    labels = [f"{r['first_name'][0]}. {r['last_name']}" for r in valid.iter_rows(named=True)]
    for k in np.argsort(yarr)[::-1][:4]:
        ax2.annotate(labels[k], (xarr[k], yarr[k]),
                     xytext=(5, 3), textcoords="offset points", fontsize=8, color=RED)

    fig.suptitle("Lojalność głosowania — buntownicy sejmowi  |  Kadencja X",
                 fontsize=14, y=1.01, color=DARK)
    fig.tight_layout(pad=1.0)
    out = FIG_DIR / "fig3_rebels.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Temporal evolution
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_temporal():
    print("  fig4: temporal …")
    df = pl.read_parquet(ANALYSIS_DIR / "temporal_metrics.parquet")
    df = df.filter(pl.col("n_edges") > 0)

    thresholds = [0.5, 0.7, 0.9]
    th_cols    = [OFFWHITE, RED2, "#7a0000"]
    th_labels  = ["próg 50%", "próg 70%", "próg 90%"]

    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.32)

    configs = [
        (gs[0, 0], "density",        "Gęstość sieci",                 "Gęstość"),
        (gs[0, 1], "n_components",   "Liczba składowych spójnych",     "Składowe"),
        (gs[1, 0], "avg_clustering", "Średni współczynnik skupienia",  "Clustering"),
        (gs[1, 1], "modularity",     "Modularność (Leiden)",           "Modularność"),
    ]

    for gspec, col, title, ylabel in configs:
        ax = fig.add_subplot(gspec)
        ax.set_facecolor(BG2)
        months = None
        for th, col_c, lbl in zip(thresholds, th_cols, th_labels):
            sub    = df.filter(pl.col("threshold") == th).sort("year_month")
            months = sub["year_month"].to_list()
            vals   = sub[col].to_list()
            ax.plot(range(len(months)), vals, color=col_c, linewidth=2.0,
                    label=lbl, marker="o", markersize=4, alpha=0.9)
        if months:
            ax.set_xticks(range(0, len(months), 3))
            ax.set_xticklabels(months[::3], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        if col == "density":
            ax.legend(fontsize=8.5, framealpha=0.45)

    fig.suptitle("Ewolucja sieci głosowań w czasie  |  Kadencja X  (XI 2023 – IV 2026)",
                 fontsize=14, y=1.01, color=DARK)
    fig.tight_layout(pad=1.0)
    out = FIG_DIR / "fig4_temporal.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Intra-party cohesion (Y axis 0.75–1.0 for readability)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_cohesion():
    print("  fig5: cohesion …")
    df = pl.read_parquet(ANALYSIS_DIR / "party_cohesion_by_month.parquet")
    main_clubs = ["KO", "PiS", "PSL-TD", "Lewica", "Polska2050-TD", "Konfederacja", "Razem"]
    df = df.filter(pl.col("club").cast(pl.Utf8).is_in(main_clubs))

    all_months = sorted(df["year_month"].unique().to_list())

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
    ax.set_facecolor(BG2)

    line_styles = ["-", "--", "-.", ":", "-", "--", "-."]
    for i, club in enumerate(main_clubs):
        sub = df.filter(pl.col("club").cast(pl.Utf8) == club).sort("year_month")
        if len(sub) == 0:
            continue
        months = sub["year_month"].to_list()
        vals   = sub["cohesion_score"].to_list()
        x_pos  = [all_months.index(m) for m in months]
        ax.plot(x_pos, vals, color=cc(club), linewidth=2.2, label=club,
                marker="o", markersize=4.5, alpha=0.92, linestyle=line_styles[i % len(line_styles)])

    ax.set_xticks(range(len(all_months)))
    ax.set_xticklabels(all_months, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Spójność głosowań (Rice index)", fontsize=10)
    ax.set_ylim(0.72, 1.03)
    ax.set_yticks(np.arange(0.75, 1.01, 0.05))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0.75, 1.01, 0.05)], fontsize=9)
    ax.set_title("Spójność wewnątrzpartyjna w czasie  |  Kadencja X", fontsize=13)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9.5, framealpha=0.5, ncol=2, loc="lower left")

    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig5_cohesion.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Topic controversy
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_topics():
    print("  fig6: topics …")
    df = (pl.read_parquet(ANALYSIS_DIR / "topic_summary.parquet")
          .filter(~pl.col("is_outlier"))
          .filter(pl.col("n_votings") >= 10))

    def shorten(words: str, n: int = 5) -> str:
        parts = [w.strip().rstrip(",") for w in words.split(",")][:n]
        return ", ".join(parts)

    red_cmap = LinearSegmentedColormap.from_list(
        "rm2", ["#FFE8E8", "#FF9999", "#E74C3C", "#C0392B", "#8B0000", "#5d0000"]
    )
    norm = mpl.colors.Normalize(vmin=0.4, vmax=1.0)

    fig = plt.figure(figsize=(15, 7.5), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.48)

    for ax_idx, (descending, subtitle) in enumerate(
        [(False, "Najbardziej kontrowersyjne"), (True, "Największy konsensus")]
    ):
        ax  = fig.add_subplot(gs[ax_idx])
        ax.set_facecolor(BG2)
        sub = df.sort("mean_pair_agreement", descending=descending).head(12)
        labels = [shorten(r["top_words"]) for r in sub.iter_rows(named=True)]
        vals   = sub["mean_pair_agreement"].to_list()
        n_v    = sub["n_votings"].to_list()
        cols   = [red_cmap(norm(v)) for v in vals]

        y    = np.arange(len(labels))
        bars = ax.barh(y, vals, color=cols, edgecolor="#111", linewidth=0.3, height=0.65)
        for bar, v, nv in zip(bars, vals, n_v):
            ax.text(v + 0.008, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}  (n={nv})", va="center", fontsize=8.5, color=DARK)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Średnia zgodność par posłów", fontsize=10)
        ax.set_title(subtitle, fontsize=12)
        ax.set_xlim(0, 1.18)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Tematy głosowań — kontrowersyjność (BERTopic)  |  Kadencja X",
                 fontsize=14, y=1.01, color=DARK)
    fig.tight_layout(pad=1.0)
    out = FIG_DIR / "fig6_topics.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 10 — Edges vs agreement threshold
# ═══════════════════════════════════════════════════════════════════════════════
def fig10_edges_vs_threshold():
    print("  fig10: edges vs threshold …")
    agreement  = np.load(NETWORKS_DIR / "agreement_matrix.npy")
    copresence = np.load(NETWORKS_DIR / "copresence_matrix.npy")

    MIN_COP = 50
    mask = np.triu(copresence >= MIN_COP, k=1)  # valid pairs (upper triangle)
    valid_agreement = agreement[mask]

    thresholds = np.arange(0.0, 1.001, 0.01)
    n_edges = [(valid_agreement >= t).sum() for t in thresholds]
    max_edges = int(mask.sum())

    fig, ax1 = plt.subplots(figsize=(11, 5.5), facecolor=BG)
    ax1.set_facecolor(BG2)

    ax1.plot(thresholds, n_edges, color=RED, linewidth=2.5)
    ax1.fill_between(thresholds, n_edges, alpha=0.12, color=RED)

    for t in [0.30, 0.50, 0.70, 0.90]:
        n = int((valid_agreement >= t).sum())
        ax1.axvline(t, color=GREY, linewidth=1.0, linestyle="--", alpha=0.7)
        ax1.annotate(f"{t:.0%}\n{n:,}", xy=(t, n),
                     xytext=(6, 6), textcoords="offset points",
                     fontsize=8.5, color=GREY)

    ax2 = ax1.twinx()
    ax2.set_facecolor(BG2)
    density = [e / max_edges for e in n_edges]
    ax2.plot(thresholds, density, color=RED_LIGHT, linewidth=1.5, linestyle=":")
    ax2.set_ylabel("Gęstość grafu", fontsize=10, color=GREY)
    ax2.tick_params(colors=GREY)
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_tick_params(labelcolor=GREY)

    ax1.set_xlabel("Próg zgodności głosowań", fontsize=11)
    ax1.set_ylabel("Liczba krawędzi", fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, max_edges * 1.05)
    ax1.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax1.set_title(
        f"Liczba krawędzi w sieci w zależności od progu zgodności\n"
        f"Kadencja X  |  {len(agreement)} posłów  ·  {max_edges:,} par z min. {MIN_COP} wspólnych głosowań",
        fontsize=13, pad=12,
    )
    ax1.grid(alpha=0.25)
    ax1.spines[["top", "right"]].set_visible(False)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=RED, linewidth=2.5, label="Liczba krawędzi"),
        Line2D([0], [0], color=RED_LIGHT, linewidth=1.5, linestyle=":", label="Gęstość grafu"),
    ]
    ax1.legend(handles=legend_handles, fontsize=9.5, framealpha=0.5, loc="upper right")

    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig10_edges_vs_threshold.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating poster figures …")
    fig2_heatmap()
    fig3_rebels()
    fig4_temporal()
    fig5_cohesion()
    fig6_topics()
    fig1_network()
    fig10_edges_vs_threshold()
    print(f"\nAll figures saved to {FIG_DIR}")
