"""
Script 21 — Structural balance (signed network).

Heider's balance theory: a signed network is balanced when all triangles
have an even number of negative edges (0 or 2). The "enemy of my enemy
is my friend" principle. Unbalanced triangles signal political anomalies.

Positive edge : agreement >= POS_THRESH  (strong agreement)
Negative edge : agreement <= NEG_THRESH  (strong disagreement)
Neutral       : between thresholds (excluded from signed graph)

Triangle types:
  +++  balanced   (mutual friends)
  +--  balanced   (enemy-of-my-enemy)
  ++-  unbalanced (two friends with a shared enemy)
  ---  unbalanced (three mutual enemies — rare)

Matrix method: for signed adjacency S,
  Trace(S³) / 6  =  (balanced triangles) − (unbalanced triangles)
  Trace(|S|³) / 6 = total triangles in signed graph

Figures:
  fig25_balance_triangles.png  — triangle type counts + frustration by club
  fig26_balance_network.png    — network with +/- edges
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import ANALYSIS_DIR, NETWORKS_DIR, PROJECT_ROOT
from src.data.store import load_mps
from src.scripts.poster_style import apply_style, CLUB_COLOURS, cc, MAIN_CLUBS, PALETTE, COALITION, OPPOSITION, club_en

apply_style()

FIG_DIR = PROJECT_ROOT / "data" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

POS_THRESH = 0.65
NEG_THRESH = 0.50
MIN_COP    = 50


def build_signed_network(term: int = 10):
    agreement  = np.load(NETWORKS_DIR / "agreement_matrix.npy")
    copresence = np.load(NETWORKS_DIR / "copresence_matrix.npy")
    mp_ids     = np.load(NETWORKS_DIR / "mp_ids.npy").tolist()
    mps        = load_mps(term)
    id_to_club = dict(zip(mps["mp_id"].to_list(), mps["club"].cast(pl.Utf8).to_list()))
    id_to_name = {r["mp_id"]: f"{r['first_name'][0]}. {r['last_name']}"
                  for r in mps.iter_rows(named=True)}

    N = len(mp_ids)
    cop_ok = copresence >= MIN_COP

    # Signed adjacency matrix
    S = np.zeros((N, N), dtype=np.float32)
    S[cop_ok & (agreement >= POS_THRESH)] =  1.0
    S[cop_ok & (agreement <= NEG_THRESH)] = -1.0
    np.fill_diagonal(S, 0.0)

    n_pos = int((S ==  1).sum() / 2)
    n_neg = int((S == -1).sum() / 2)
    print(f"  Signed network: {n_pos} positive edges, {n_neg} negative edges")

    # Triangle analysis via matrix trace method
    # |S| (binary adjacency of signed graph)
    A_bin = (np.abs(S) > 0).astype(np.float32)
    A2    = A_bin @ A_bin
    total_triangles = int(np.trace(A2 @ A_bin) / 6)

    S2              = S @ S
    signed_trace    = int(round(np.trace(S2 @ S) / 6))
    balanced        = (total_triangles + signed_trace) // 2
    unbalanced      = (total_triangles - signed_trace) // 2
    frustration     = unbalanced / total_triangles if total_triangles > 0 else 0.0

    print(f"  Total triangles: {total_triangles:,}")
    print(f"  Balanced: {balanced:,}  ({balanced/total_triangles:.1%})")
    print(f"  Unbalanced: {unbalanced:,}  ({frustration:.1%})")

    # Detailed triangle types using sampling for large networks
    # Enumerate exact types for the signed graph edges
    rows_i, rows_j = np.where(np.triu(A_bin, k=1))
    # Build edge list and adjacency for fast lookup
    # For type counting, sample up to 500k triangles
    print("  Counting triangle types (sampling) …")
    type_counts = {"+++": 0, "+--": 0, "++-": 0, "---": 0}
    rng = np.random.default_rng(42)
    sample_size = min(len(rows_i), 50_000)
    idx = rng.choice(len(rows_i), sample_size, replace=False)
    for k in idx:
        i, j = int(rows_i[k]), int(rows_j[k])
        for m in range(N):
            if m == i or m == j:
                continue
            if A_bin[i, m] > 0 and A_bin[j, m] > 0:
                signs = sorted([int(S[i, j]), int(S[i, m]), int(S[j, m])])
                s_neg = signs.count(-1)
                if   s_neg == 0: type_counts["+++"] += 1
                elif s_neg == 1: type_counts["++-"] += 1
                elif s_neg == 2: type_counts["+--"] += 1
                else:            type_counts["---"] += 1

    # Per-MP frustration: what fraction of their triangles are unbalanced
    # using diagonal of S^2 (counts of paths of length 2) and S^3
    mp_frustration = []
    for i in range(N):
        # Count triangles through node i in signed graph
        signed_i = int(round((S2 @ S)[i, i] / 2))   # signed count
        total_i  = int(round((A2 @ A_bin)[i, i] / 2))  # total
        if total_i == 0:
            mp_frustration.append({"mp_id": mp_ids[i], "frustration": None,
                                   "n_triangles": 0,
                                   "club": id_to_club.get(mp_ids[i], "niez."),
                                   "name": id_to_name.get(mp_ids[i], "")})
        else:
            bal_i  = (total_i + signed_i) // 2
            unb_i  = (total_i - signed_i) // 2
            frust  = max(0.0, unb_i / total_i)
            mp_frustration.append({"mp_id": mp_ids[i], "frustration": frust,
                                   "n_triangles": total_i,
                                   "club": id_to_club.get(mp_ids[i], "niez."),
                                   "name": id_to_name.get(mp_ids[i], "")})

    df = pl.DataFrame(mp_frustration)

    summary = {
        "total_triangles": total_triangles, "balanced": balanced,
        "unbalanced": unbalanced, "frustration_index": frustration,
        "n_pos_edges": n_pos, "n_neg_edges": n_neg,
    }
    return S, mp_ids, id_to_club, id_to_name, df, summary, type_counts


def fig25_balance_triangles(df: pl.DataFrame, summary: dict, type_counts: dict) -> None:
    print("  fig25: balance triangles …")
    fig = plt.figure(figsize=(15, 6.5), facecolor="white")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42)

    # Left: triangle type pie
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("white")
    balanced_types   = {"+++": type_counts["+++"], "+--": type_counts["+--"]}
    unbalanced_types = {"++-": type_counts["++-"], "---": type_counts["---"]}
    total_sampled = sum(type_counts.values())

    labels = list(type_counts.keys())
    vals   = [type_counts[l] for l in labels]
    colors = ["#8B0000", PALETTE["accent"], "#aec7e8", PALETTE["primary"]]
    explode= [0, 0, 0.07, 0.07]
    wedges, texts, autotexts = ax1.pie(
        vals, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 10, "color": PALETTE["dark"]},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax1.set_title(
        f"Triangle types in the signed network\n"
        f"(sample of {total_sampled:,} triangles)\n"
        f"Balanced: {summary['balanced']:,}  |  Unbal.: {summary['unbalanced']:,}",
        fontsize=10, pad=10,
    )
    red_p  = mpatches.Patch(color="#8B0000", label="Balanced")
    blue_p = mpatches.Patch(color="#aec7e8", label="Unbalanced")
    ax1.legend(handles=[red_p, blue_p], fontsize=8.5, loc="lower center",
               bbox_to_anchor=(0.5, -0.12), framealpha=0.5)

    # Middle: frustration index by club
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("white")
    club_frust = (
        df.filter(
            pl.col("frustration").is_not_null()
            & pl.col("club").is_in(MAIN_CLUBS)
            & (pl.col("n_triangles") >= 10)
        )
        .group_by("club")
        .agg(pl.col("frustration").mean().alias("mean_frust"))
        .sort("mean_frust", descending=True)
    )
    clubs_f  = club_frust["club"].to_list()
    fractions= club_frust["mean_frust"].to_numpy()
    y        = np.arange(len(clubs_f))
    cols     = [CLUB_COLOURS.get(c, PALETTE["neutral"]) for c in clubs_f]
    ax2.barh(y, fractions, color=cols, alpha=0.85, edgecolor="#333", linewidth=0.4, height=0.65)
    for i, f in enumerate(fractions):
        ax2.text(f + 0.003, i, f"{f:.1%}", va="center", fontsize=9)
    ax2.set_yticks(y); ax2.set_yticklabels(clubs_f, fontsize=9.5)
    ax2.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax2.set_xlabel("Frustration index (% unbalanced triangles)", fontsize=10)
    ax2.set_title("Frustration by club", fontsize=11)
    ax2.axvline(summary["frustration_index"], color=PALETTE["dark"], linewidth=1.2,
                linestyle="--", alpha=0.6, label=f"Mean {summary['frustration_index']:.1%}")
    ax2.legend(fontsize=8.5, framealpha=0.5)
    ax2.grid(axis="x", alpha=0.3); ax2.spines[["top", "right"]].set_visible(False)

    # Right: top MPs by frustration (political anomalies)
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("white")
    top_frust = (
        df.filter(
            pl.col("frustration").is_not_null()
            & (pl.col("n_triangles") >= 50)
        )
        .sort("frustration", descending=True)
        .head(15)
    )
    names_t  = top_frust["name"].to_list()
    fracs_t  = top_frust["frustration"].to_numpy()
    clubs_t  = top_frust["club"].to_list()
    y2       = np.arange(len(names_t))
    cols2    = [CLUB_COLOURS.get(c, PALETTE["neutral"]) for c in clubs_t]
    ax3.barh(y2, fracs_t, color=cols2, alpha=0.85, edgecolor="#333", linewidth=0.4, height=0.65)
    for i, f in enumerate(fracs_t):
        ax3.text(f + 0.003, i, f"{f:.1%}", va="center", fontsize=8.5)
    ax3.set_yticks(y2)
    ax3.set_yticklabels([f"{n}  [{c}]" for n, c in zip(names_t, clubs_t)], fontsize=8)
    ax3.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax3.set_xlabel("Frustration index", fontsize=10)
    ax3.set_title("Top MPs by frustration\n(network anomalies)", fontsize=11)
    ax3.invert_yaxis()
    ax3.grid(axis="x", alpha=0.3); ax3.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Structural balance of voting network  |  Term X\n"
        f"+ edge: agreement ≥ {POS_THRESH:.0%}  ·  − edge: agreement ≤ {NEG_THRESH:.0%}  "
        f"·  Global frustration: {summary['frustration_index']:.2%}",
        fontsize=13, color=PALETTE["dark"], y=1.02,
    )
    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig25_balance_triangles.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out}")


def fig26_balance_network(S: np.ndarray, mp_ids: list, id_to_club: dict,
                           id_to_name: dict) -> None:
    print("  fig26: signed network visualisation …")
    N = len(mp_ids)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        G.nodes[i]["club"] = id_to_club.get(mp_ids[i], "niez.")

    pos_edges, neg_edges = [], []
    for i in range(N):
        for j in range(i + 1, N):
            if S[i, j] == 1.0:
                G.add_edge(i, j, sign=1)
                pos_edges.append((i, j))
            elif S[i, j] == -1.0:
                G.add_edge(i, j, sign=-1)
                neg_edges.append((i, j))

    print(f"    layout on {G.number_of_nodes()} nodes, {G.number_of_edges()} edges …")
    pos = nx.spring_layout(G, seed=42, k=0.30, iterations=60)

    node_cols  = [CLUB_COLOURS.get(G.nodes[n]["club"], PALETTE["neutral"]) for n in G.nodes()]
    degree_c   = nx.degree_centrality(G)
    node_sizes = [8 + 50 * degree_c[n] for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(14, 12), facecolor="white")
    ax.set_facecolor("white"); ax.axis("off")

    for u, v in pos_edges:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=PALETTE["accent"], alpha=0.08, linewidth=0.3, zorder=1)
    for u, v in neg_edges:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=PALETTE["primary"], alpha=0.25, linewidth=0.5, zorder=1)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_cols,
                           node_size=node_sizes, linewidths=0.3,
                           edgecolors="#00000044", alpha=0.92)

    seen = {}
    for n in G.nodes():
        c = G.nodes[n]["club"]
        if c not in seen: seen[c] = CLUB_COLOURS.get(c, PALETTE["neutral"])
    club_handles = [mpatches.Patch(color=col, label=club)
                    for club, col in sorted(seen.items())]
    edge_handles = [
        mpl.lines.Line2D([0], [0], color=PALETTE["accent"],  linewidth=2, label=f"Positive edges (agreement ≥ {POS_THRESH:.0%})"),
        mpl.lines.Line2D([0], [0], color=PALETTE["primary"], linewidth=2, label=f"Negative edges (agreement ≤ {NEG_THRESH:.0%})"),
    ]
    ax.legend(handles=club_handles + edge_handles, fontsize=7.5, loc="lower left",
              framealpha=0.6, ncol=2, title="Legend", title_fontsize=8)

    ax.set_title(
        f"Signed voting network  |  {len(pos_edges):,} '+' edges (red)  ·  "
        f"{len(neg_edges):,} '−' edges (blue)\nTerm X",
        fontsize=12, pad=12,
    )
    fig.tight_layout(pad=0.5)
    out = FIG_DIR / "fig26_balance_network.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out}")


if __name__ == "__main__":
    print("Script 21 — Structural balance …")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    S, mp_ids, id_to_club, id_to_name, df, summary, type_counts = build_signed_network()
    df.write_parquet(ANALYSIS_DIR / "structural_balance.parquet")
    print(f"  Summary: {summary}")
    fig25_balance_triangles(df, summary, type_counts)
    fig26_balance_network(S, mp_ids, id_to_club, id_to_name)
    print(f"\nDone. Figures in {FIG_DIR}")
