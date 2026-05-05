"""
Script 17 — Echo chamber heatmap + bridge MPs.

Analyses:
  1. Hierarchical clustering heatmap of MP agreement (echo chamber)
  2. Bridge MPs — MPs with unusually high agreement with the opposite bloc

Figures:
  fig17_echo_chamber.png
  fig18_bridge_mps.png
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import polars as pl
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import ANALYSIS_DIR, NETWORKS_DIR, PROJECT_ROOT
from src.data.store import load_mps
from src.scripts.poster_style import apply_style, CLUB_COLOURS, cc, MAIN_CLUBS, PALETTE, COALITION, OPPOSITION, club_en

apply_style()

FIG_DIR = PROJECT_ROOT / "data" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data(term: int = 10):
    agreement  = np.load(NETWORKS_DIR / "agreement_matrix.npy")
    copresence = np.load(NETWORKS_DIR / "copresence_matrix.npy")
    mp_ids     = np.load(NETWORKS_DIR / "mp_ids.npy").tolist()
    mps        = load_mps(term)
    id_to_club = dict(zip(mps["mp_id"].to_list(), mps["club"].cast(pl.Utf8).to_list()))
    id_to_name = {
        r["mp_id"]: f"{r['first_name'][0]}. {r['last_name']}"
        for r in mps.iter_rows(named=True)
    }
    clubs = [id_to_club.get(mid, "niez.") for mid in mp_ids]
    names = [id_to_name.get(mid, str(mid)) for mid in mp_ids]
    return agreement, copresence, mp_ids, clubs, names


def fig17_echo_chamber(agreement, copresence, mp_ids, clubs, names, min_cop: int = 50) -> None:
    print("  fig17: echo chamber heatmap …")

    # Mask out pairs with insufficient co-presence; replace with global mean
    mask = copresence >= min_cop
    mat  = agreement.copy()
    global_mean = mat[mask].mean()
    mat[~mask] = global_mean
    np.fill_diagonal(mat, 1.0)

    # Distance matrix for clustering: 1 - agreement
    dist = 1.0 - mat
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0, None)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")

    from scipy.cluster.hierarchy import leaves_list
    order = leaves_list(Z)

    mat_ord   = mat[np.ix_(order, order)]
    clubs_ord = [clubs[i] for i in order]
    node_cols = [cc(c) for c in clubs_ord]

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "red_heat", ["#FFFFFF", "#FFE0E0", "#FF9999", "#E74C3C", "#8B0000", "#3d0000"]
    )

    N = len(order)
    fig = plt.figure(figsize=(14, 13), facecolor="white")
    # Layout: dendrogram top (thin), color bar left, heatmap
    gs = mpl.gridspec.GridSpec(
        3, 3,
        figure=fig,
        height_ratios=[0.08, 0.02, 1],
        width_ratios=[0.08, 0.02, 1],
        hspace=0.02, wspace=0.02,
    )

    # Top dendrogram
    ax_dend_top = fig.add_subplot(gs[0, 2])
    ax_dend_top.set_facecolor("white")
    dend = dendrogram(Z, ax=ax_dend_top, no_labels=True, orientation="top",
                      color_threshold=0, above_threshold_color=PALETTE["accent"])
    ax_dend_top.set_axis_off()

    # Left dendrogram
    ax_dend_left = fig.add_subplot(gs[2, 0])
    ax_dend_left.set_facecolor("white")
    dendrogram(Z, ax=ax_dend_left, no_labels=True, orientation="left",
               color_threshold=0, above_threshold_color=PALETTE["accent"])
    ax_dend_left.set_axis_off()

    # Club colour strip (top)
    ax_strip_top = fig.add_subplot(gs[1, 2])
    ax_strip_top.set_facecolor("white")
    strip_top = np.array([[cc(c) for c in clubs_ord]])
    ax_strip_top.imshow([[mpl.colors.to_rgba(c) for c in strip_top[0]]],
                        aspect="auto", interpolation="nearest")
    ax_strip_top.set_axis_off()

    # Club colour strip (left)
    ax_strip_left = fig.add_subplot(gs[2, 1])
    ax_strip_left.set_facecolor("white")
    strip_left = np.array([[cc(c) for c in clubs_ord]]).T
    ax_strip_left.imshow([[mpl.colors.to_rgba(cc(c))] for c in clubs_ord],
                         aspect="auto", interpolation="nearest")
    ax_strip_left.set_axis_off()

    # Heatmap
    ax = fig.add_subplot(gs[2, 2])
    ax.set_facecolor("white")
    im = ax.imshow(mat_ord, cmap=cmap, vmin=0.3, vmax=1.0,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        f"Echo chamber — voting agreement matrix (MPs sorted hierarchically)\n"
        f"Term X  |  {N} MPs  |  min. {min_cop} joint votes",
        fontsize=12, pad=10,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Voting agreement", fontsize=9)

    # Legend
    seen = {}
    for c in clubs_ord:
        if c not in seen:
            seen[c] = cc(c)
    handles = [mpatches.Patch(color=col, label=club) for club, col in sorted(seen.items())]
    ax.legend(handles=handles, loc="lower left", fontsize=7.5,
              framealpha=0.7, ncol=2, title="Club", title_fontsize=8,
              bbox_to_anchor=(-0.01, 0))

    out = FIG_DIR / "fig17_echo_chamber.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out}")


def fig18_bridge_mps(agreement, copresence, mp_ids, clubs, names,
                     min_cop: int = 50, term: int = 10) -> None:
    print("  fig18: bridge MPs …")
    N = len(mp_ids)

    coal_idx  = [i for i, c in enumerate(clubs) if c in COALITION]
    oppo_idx  = [i for i, c in enumerate(clubs) if c in OPPOSITION]

    if not coal_idx or not oppo_idx:
        print("    skipped — no coalition/opposition MPs found")
        return

    rows = []
    for i in range(N):
        club = clubs[i]
        if club not in (COALITION | OPPOSITION):
            continue

        # Average agreement with each bloc (only pairs with sufficient co-presence)
        def bloc_mean(idxs):
            vals = [
                agreement[i, j]
                for j in idxs
                if j != i and copresence[i, j] >= min_cop
            ]
            return float(np.mean(vals)) if vals else np.nan

        avg_coal = bloc_mean(coal_idx)
        avg_oppo = bloc_mean(oppo_idx)
        if np.isnan(avg_coal) or np.isnan(avg_oppo):
            continue

        # Bridge score = how similar to the OTHER bloc relative to own bloc
        own_mean  = avg_coal if club in COALITION else avg_oppo
        other_mean= avg_oppo if club in COALITION else avg_coal
        bridge    = other_mean - (own_mean - other_mean)  # how close to the other side

        rows.append({
            "mp_id":      mp_ids[i],
            "name":       names[i],
            "club":       club,
            "bloc":       "Coalition" if club in COALITION else "Opposition",
            "avg_coal":   avg_coal,
            "avg_oppo":   avg_oppo,
            "bridge_score": other_mean,  # absolute agreement with other side
        })

    bridge_df = pl.DataFrame(rows).sort("bridge_score", descending=True)
    bridge_df.write_parquet(ANALYSIS_DIR / "bridge_mps.parquet")

    top_coal = (bridge_df.filter(pl.col("bloc") == "Coalition")
                .sort("bridge_score", descending=True).head(15))
    top_oppo = (bridge_df.filter(pl.col("bloc") == "Opposition")
                .sort("bridge_score", descending=True).head(15))

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), facecolor="white")
    fig.suptitle("Bridge MPs — highest agreement with the opposing bloc  |  Term X",
                 fontsize=14, color=PALETTE["dark"], y=1.01)

    for ax, top, title in zip(axes,
                               [top_coal, top_oppo],
                               ["From coalition → agreement with opposition",
                                "From opposition → agreement with coalition"]):
        ax.set_facecolor("white")
        names_t = top["name"].to_list()
        scores  = top["bridge_score"].to_numpy()
        clubs_t = top["club"].to_list()
        own_avg = top["avg_coal" if title.startswith("From coal") else "avg_oppo"].to_numpy()
        y       = np.arange(len(names_t))

        ax.barh(y, own_avg,    color="#CCCCCC", alpha=0.6, height=0.55, label="Own bloc")
        ax.barh(y, scores, color=PALETTE["accent"], alpha=0.85, height=0.55, label="Opposing bloc")
        for yi, (s, o) in enumerate(zip(scores, own_avg)):
            ax.text(max(s, o) + 0.003, yi, f"{s:.3f}", va="center", fontsize=8, color=PALETTE["dark"])

        ax.set_yticks(y)
        ax.set_yticklabels([f"{n}  [{c}]" for n, c in zip(names_t, clubs_t)], fontsize=8.5)
        ax.set_xlabel("Mean voting agreement", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlim(0.3, 1.0)
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8.5, framealpha=0.5, loc="lower right")

    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig18_bridge_mps.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out}")


if __name__ == "__main__":
    print("Script 17 — Echo chamber + bridge MPs …")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    agreement, copresence, mp_ids, clubs, names = load_data()
    fig17_echo_chamber(agreement, copresence, mp_ids, clubs, names)
    fig18_bridge_mps(agreement, copresence, mp_ids, clubs, names)
    print(f"\nDone. Figures in {FIG_DIR}")
