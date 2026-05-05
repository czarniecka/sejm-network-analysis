"""
Script 20 — k-core decomposition.

The k-core of a graph is the maximal subgraph in which every node has
at least k neighbours. Each MP gets a 'coreness' (shell number) = the
maximum k for which they belong to the k-core.

Figures:
  fig23_kcore_distribution.png  — coreness distribution by club + innermost core members
  fig24_kcore_network.png       — spring layout coloured by shell number
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import ANALYSIS_DIR, NETWORKS_DIR, PROJECT_ROOT
from src.data.store import load_mps
from src.analysis.network import adjacency_to_networkx

FIG_DIR = PROJECT_ROOT / "data" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BG    = "#FAFAFA"; BG2  = "#FFFFFF"; RED  = "#C0392B"; RED2 = "#E74C3C"
DARK  = "#1a1a1a"; GREY = "#757575"; GRID = "#E0E0E0"

CLUB_COLOURS = {
    "KO": "#E8C4C4", "PiS": "#8B0000", "PSL-TD": "#C0392B",
    "Lewica": "#FF5252", "Polska2050": "#FF8C69", "Polska2050-TD": "#FFAB91",
    "Konfederacja": "#5D0000", "Konfederacja_KP": "#7A0000",
    "Razem": "#FF7675", "PSL": "#D4826B", "Centrum": "#F0A090",
    "niez.": "#777777", "Demokracja": "#B05050",
}
MAIN_CLUBS = ["KO", "PiS", "PSL-TD", "Lewica", "Polska2050-TD", "Konfederacja", "Razem"]

mpl.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG2, "axes.edgecolor": "#CCCCCC",
    "axes.labelcolor": DARK, "text.color": DARK, "xtick.color": DARK, "ytick.color": DARK,
    "grid.color": GRID, "grid.linewidth": 0.6, "font.family": "sans-serif", "font.size": 11,
})

THRESHOLD = 0.70
MIN_COP   = 50


def build_kcore(term: int = 10):
    agreement  = np.load(NETWORKS_DIR / "agreement_matrix.npy")
    copresence = np.load(NETWORKS_DIR / "copresence_matrix.npy")
    mp_ids     = np.load(NETWORKS_DIR / "mp_ids.npy").tolist()
    mps        = load_mps(term)
    id_to_club = dict(zip(mps["mp_id"].to_list(), mps["club"].cast(pl.Utf8).to_list()))
    id_to_name = {r["mp_id"]: f"{r['first_name'][0]}. {r['last_name']}"
                  for r in mps.iter_rows(named=True)}

    G = adjacency_to_networkx(agreement, THRESHOLD, mp_ids, MIN_COP, copresence)
    # Annotate with club and name
    for i, mid in enumerate(mp_ids):
        G.nodes[i]["club"] = id_to_club.get(mid, "niez.")
        G.nodes[i]["name"] = id_to_name.get(mid, str(mid))
        G.nodes[i]["mp_id"] = mid

    core_nums = nx.core_number(G)
    nx.set_node_attributes(G, core_nums, "coreness")

    rows = [{"mp_id": mp_ids[i], "coreness": core_nums[i],
             "club": id_to_club.get(mp_ids[i], "niez."),
             "name": id_to_name.get(mp_ids[i], str(mp_ids[i]))}
            for i in range(len(mp_ids))]
    df = pl.DataFrame(rows)
    return G, df, core_nums


def fig23_kcore_distribution(G, df: pl.DataFrame, core_nums: dict) -> None:
    print("  fig23: k-core distribution …")
    max_k = max(core_nums.values())
    print(f"    max coreness = {max_k}")

    fig = plt.figure(figsize=(15, 6.5), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.40)

    # Left: stacked bar — how many MPs per club per coreness shell
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(BG2)
    shell_vals = sorted(df["coreness"].unique().to_list())
    cmap_k = LinearSegmentedColormap.from_list("kc", ["#FFE8E8", RED2, "#8B0000", "#3d0000"])
    norm_k = mpl.colors.Normalize(vmin=min(shell_vals), vmax=max(shell_vals))

    # Group by coreness — bar per shell, colour by coreness
    counts = df.group_by("coreness").agg(pl.len().alias("n")).sort("coreness")
    bars = ax1.bar(counts["coreness"].to_list(), counts["n"].to_list(),
                   color=[cmap_k(norm_k(k)) for k in counts["coreness"].to_list()],
                   edgecolor="#333", linewidth=0.4, width=0.75)
    for bar, n in zip(bars, counts["n"].to_list()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(n), ha="center", va="bottom", fontsize=8.5, color=DARK)
    ax1.set_xlabel("Numer powłoki (coreness)", fontsize=10)
    ax1.set_ylabel("Liczba posłów", fontsize=10)
    ax1.set_title("Rozkład posłów wg powłoki k-core", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)
    sm = mpl.cm.ScalarMappable(cmap=cmap_k, norm=norm_k)
    plt.colorbar(sm, ax=ax1, fraction=0.03, pad=0.02, label="Coreness")

    # Right: innermost core members by club
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(BG2)
    inner_k = max_k - 1  # second-innermost to get more interesting group
    inner = df.filter(pl.col("coreness") >= inner_k)
    club_counts = (inner.filter(pl.col("club").is_in(MAIN_CLUBS))
                   .group_by("club").agg(pl.len().alias("n"))
                   .sort("n", descending=True))
    total_by_club = df.filter(pl.col("club").is_in(MAIN_CLUBS)).group_by("club").agg(pl.len().alias("total"))
    club_data = club_counts.join(total_by_club, on="club", how="left").with_columns(
        (pl.col("n") / pl.col("total")).alias("frac")
    ).sort("frac", descending=True)

    clubs_t = club_data["club"].to_list()
    fracs   = club_data["frac"].to_numpy()
    ns      = club_data["n"].to_list()
    y       = np.arange(len(clubs_t))
    cols    = [CLUB_COLOURS.get(c, GREY) for c in clubs_t]
    ax2.barh(y, fracs, color=cols, alpha=0.85, edgecolor="#333", linewidth=0.4, height=0.65)
    for i, (f, n) in enumerate(zip(fracs, ns)):
        ax2.text(f + 0.01, i, f"{f:.0%}  (n={n})", va="center", fontsize=8.5)
    ax2.set_yticks(y); ax2.set_yticklabels(clubs_t, fontsize=9.5)
    ax2.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax2.set_xlabel("Odsetek posłów klubu w innermost k-core", fontsize=10)
    ax2.set_title(f"Skład klubowy innermost k-core\n(coreness ≥ {inner_k}, próg {THRESHOLD:.0%})", fontsize=11)
    ax2.set_xlim(0, 1.15)
    ax2.grid(axis="x", alpha=0.3); ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"K-core decomposition sieci głosowań  |  Kadencja X  |  próg {THRESHOLD:.0%}",
                 fontsize=14, color=DARK, y=1.01)
    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig23_kcore_distribution.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    saved {out}")


def fig24_kcore_network(G, df: pl.DataFrame, core_nums: dict) -> None:
    print("  fig24: k-core network (spring layout) …")
    max_k = max(core_nums.values())
    cmap_k = LinearSegmentedColormap.from_list("kc", ["#FFE8E8", "#FF9999", RED2, "#8B0000", "#3d0000"])
    norm_k = mpl.colors.Normalize(vmin=min(core_nums.values()), vmax=max_k)

    print("    computing layout (may take a minute) …")
    pos = nx.spring_layout(G, seed=42, k=0.25, iterations=50, weight="weight")

    node_cols = [cmap_k(norm_k(core_nums[n])) for n in G.nodes()]
    node_sizes= [5 + 40 * (core_nums[n] / max_k) ** 2 for n in G.nodes()]
    edges = list(G.edges())
    edge_alpha = 0.04

    fig, ax = plt.subplots(figsize=(14, 12), facecolor=BG)
    ax.set_facecolor(BG); ax.axis("off")

    for u, v in edges:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=RED, alpha=edge_alpha, linewidth=0.25, zorder=1)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_cols,
                           node_size=node_sizes, linewidths=0.3,
                           edgecolors="#00000044", alpha=0.92)

    # Annotate innermost core
    inner_k = max_k
    inner_nodes = [n for n in G.nodes() if core_nums[n] == inner_k]
    for n in inner_nodes[:20]:
        ax.annotate(G.nodes[n]["name"], pos[n], fontsize=5.5, color=DARK,
                    ha="center", va="bottom", zorder=3,
                    xytext=(0, 4), textcoords="offset points")

    sm = mpl.cm.ScalarMappable(cmap=cmap_k, norm=norm_k)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, shrink=0.6)
    cbar.set_label("Coreness (numer powłoki)", fontsize=10)

    ax.set_title(
        f"Sieć głosowań ubarwiona wg powłoki k-core  |  Kadencja X  |  próg {THRESHOLD:.0%}\n"
        f"max coreness = {max_k}  ·  innermost core: {len(inner_nodes)} posłów",
        fontsize=13, pad=12,
    )
    fig.tight_layout(pad=0.5)
    out = FIG_DIR / "fig24_kcore_network.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    saved {out}")


if __name__ == "__main__":
    print("Script 20 — k-core decomposition …")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    G, df, core_nums = build_kcore()
    df.write_parquet(ANALYSIS_DIR / "kcore.parquet")
    print(f"  Coreness range: {min(core_nums.values())} – {max(core_nums.values())}")
    fig23_kcore_distribution(G, df, core_nums)
    fig24_kcore_network(G, df, core_nums)
    print(f"\nDone. Figures in {FIG_DIR}")
