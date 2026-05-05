"""
Script 22 — Assortativity (homophily) analysis.

Newman's assortativity coefficient r ∈ [−1, 1]:
  r > 0  nodes prefer to connect to similar nodes (homophily)
  r = 0  random mixing
  r < 0  nodes prefer to connect to dissimilar nodes (heterophily)

Computed for: club, voivodeship, degree, age, gender.
Also: club mixing matrix and evolution across thresholds.

Figures:
  fig27_assortativity.png    — r coefficients + mixing matrix
  fig28_assortativity_thresh.png — assortativity vs threshold
"""

import sys
from pathlib import Path
from datetime import date as _date

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import ANALYSIS_DIR, NETWORKS_DIR, PROJECT_ROOT, AGREEMENT_THRESHOLDS
from src.data.store import load_mps
from src.analysis.network import adjacency_to_networkx
from src.scripts.poster_style import apply_style, CLUB_COLOURS, cc, MAIN_CLUBS, PALETTE, COALITION, OPPOSITION, club_en

apply_style()

FIG_DIR = PROJECT_ROOT / "data" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REF_DATE  = _date(2023, 11, 13)
MIN_COP   = 50
THRESHOLD = 0.70


def build_annotated_graph(term: int = 10, threshold: float = THRESHOLD):
    agreement  = np.load(NETWORKS_DIR / "agreement_matrix.npy")
    copresence = np.load(NETWORKS_DIR / "copresence_matrix.npy")
    mp_ids     = np.load(NETWORKS_DIR / "mp_ids.npy").tolist()
    mps        = load_mps(term)

    id_attrs = {}
    for r in mps.iter_rows(named=True):
        age = None
        if r["birth_date"] is not None:
            try:
                age = (REF_DATE - r["birth_date"]).days // 365
            except Exception:
                pass
        id_attrs[r["mp_id"]] = {
            "club":        str(r["club"]),
            "voivodeship": str(r["voivodeship"]),
            "age":         age,
            "is_female":   int(str(r["first_name"]).endswith("a")),
        }

    G = adjacency_to_networkx(agreement, threshold, mp_ids, MIN_COP, copresence)
    for i, mid in enumerate(mp_ids):
        attrs = id_attrs.get(mid, {})
        for k, v in attrs.items():
            G.nodes[i][k] = v

    return G, mp_ids


def compute_assortativity(G, term: int = 10) -> dict:
    results = {}

    # Categorical: club
    try:
        results["Club (r)"] = nx.attribute_assortativity_coefficient(G, "club")
    except Exception as e:
        results["Club (r)"] = float("nan")
        print(f"    club assortativity failed: {e}")

    # Categorical: voivodeship
    try:
        results["Voivodeship (r)"] = nx.attribute_assortativity_coefficient(G, "voivodeship")
    except Exception:
        results["Voivodeship (r)"] = float("nan")

    # Numeric: degree
    try:
        results["Node degree (r)"] = nx.degree_assortativity_coefficient(G)
    except Exception:
        results["Node degree (r)"] = float("nan")

    # Numeric: age
    has_age = [n for n in G.nodes() if G.nodes[n].get("age") is not None]
    if len(has_age) > 10:
        G_age = G.subgraph(has_age).copy()
        try:
            results["Age (r)"] = nx.numeric_assortativity_coefficient(G_age, "age")
        except Exception:
            results["Age (r)"] = float("nan")
    else:
        results["Age (r)"] = float("nan")

    # Numeric: gender (binary)
    try:
        results["Gender (r)"] = nx.numeric_assortativity_coefficient(G, "is_female")
    except Exception:
        results["Gender (r)"] = float("nan")

    return results


def compute_mixing_matrix(G) -> tuple[np.ndarray, list]:
    """Club-level mixing matrix: fraction of edges that go between club pairs."""
    clubs = sorted({G.nodes[n]["club"] for n in G.nodes() if G.nodes[n].get("club")})
    idx   = {c: i for i, c in enumerate(clubs)}
    n     = len(clubs)
    mat   = np.zeros((n, n))
    for u, v in G.edges():
        cu = G.nodes[u].get("club", "")
        cv = G.nodes[v].get("club", "")
        if cu in idx and cv in idx:
            mat[idx[cu], idx[cv]] += 1
            if cu != cv:
                mat[idx[cv], idx[cu]] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat_norm = mat / row_sums
    return mat_norm, clubs


def fig27_assortativity(G, results: dict) -> None:
    print("  fig27: assortativity coefficients + mixing matrix …")
    fig = plt.figure(figsize=(15, 6.5), facecolor="white")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.42)

    # Left: bar chart of assortativity coefficients
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("white")
    labels = list(results.keys())
    vals   = [results[l] for l in labels]
    cols   = [PALETTE["accent"] if v > 0 else PALETTE["neutral"] for v in vals]
    y      = np.arange(len(labels))
    bars   = ax1.barh(y, vals, color=cols, alpha=0.85, edgecolor="#333", linewidth=0.4, height=0.55)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax1.text(v + (0.01 if v >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                     f"{v:.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=10)
    ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=10)
    ax1.axvline(0, color=PALETTE["dark"], linewidth=0.9, linestyle="-", alpha=0.5)
    ax1.set_xlabel("Newman's assortativity coefficient r", fontsize=10)
    ax1.set_title(
        f"Voting network assortativity\n(threshold {THRESHOLD:.0%}, r > 0 = homophily)",
        fontsize=12,
    )
    ax1.set_xlim(-0.5, 1.05)
    ax1.grid(axis="x", alpha=0.3); ax1.spines[["top", "right"]].set_visible(False)

    # Right: mixing matrix (clubs)
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("white")
    mat_norm, clubs = compute_mixing_matrix(G)

    cmap = LinearSegmentedColormap.from_list(
        "rm", ["#FFFFFF", "#FFE0E0", "#FF9999", "#E74C3C", "#8B0000", "#3d0000"]
    )
    im = ax2.imshow(mat_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    short = [c.replace("Konfederacja", "Konf.").replace("Polska2050", "P2050") for c in clubs]
    ax2.set_xticks(range(len(clubs))); ax2.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(range(len(clubs))); ax2.set_yticklabels(short, fontsize=8)
    for i in range(len(clubs)):
        for j in range(len(clubs)):
            v = mat_norm[i, j]
            col = "#FFFFFF" if v > 0.5 else PALETTE["dark"]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=col)
    ax2.set_title("Edge mixing matrix by club\n(row = source club, value = edge fraction)", fontsize=10)
    fig.colorbar(im, ax=ax2, fraction=0.03, pad=0.02, label="Edge fraction")

    fig.suptitle("Voting network assortativity analysis  |  Term X",
                 fontsize=14, color=PALETTE["dark"], y=1.01)
    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig27_assortativity.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out}")


def fig28_assortativity_thresh(term: int = 10) -> None:
    print("  fig28: assortativity vs threshold …")
    agreement  = np.load(NETWORKS_DIR / "agreement_matrix.npy")
    copresence = np.load(NETWORKS_DIR / "copresence_matrix.npy")
    mp_ids     = np.load(NETWORKS_DIR / "mp_ids.npy").tolist()

    thresholds = np.arange(0.40, 0.96, 0.05)
    r_club = []; r_degree = []; r_voiv = []

    for t in thresholds:
        G_t, _ = build_annotated_graph(term=term, threshold=float(t))
        if G_t.number_of_edges() == 0:
            r_club.append(np.nan); r_degree.append(np.nan); r_voiv.append(np.nan)
            continue
        try:
            r_club.append(nx.attribute_assortativity_coefficient(G_t, "club"))
        except Exception:
            r_club.append(np.nan)
        try:
            r_degree.append(nx.degree_assortativity_coefficient(G_t))
        except Exception:
            r_degree.append(np.nan)
        try:
            r_voiv.append(nx.attribute_assortativity_coefficient(G_t, "voivodeship"))
        except Exception:
            r_voiv.append(np.nan)

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(thresholds, r_club,   color=PALETTE["accent"],    linewidth=2.5, marker="o", markersize=5,
            label="Club (categorical)", zorder=3)
    ax.plot(thresholds, r_voiv,   color=PALETTE["secondary"], linewidth=2.0, marker="s", markersize=4,
            linestyle="--", label="Voivodeship (categorical)", zorder=3)
    ax.plot(thresholds, r_degree, color=PALETTE["neutral"],   linewidth=2.0, marker="^", markersize=4,
            linestyle=":", label="Node degree (numeric)", zorder=3)

    ax.axhline(0, color=PALETTE["dark"], linewidth=0.8, linestyle="-", alpha=0.4)
    ax.set_xlabel("Agreement threshold", fontsize=11)
    ax.set_ylabel("Assortativity coefficient r", fontsize=11)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.set_title("Assortativity vs agreement threshold  |  Term X", fontsize=13, pad=12)
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.5)
    ax.set_ylim(-0.6, 1.05)

    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig28_assortativity_thresh.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out}")


if __name__ == "__main__":
    print("Script 22 — Assortativity analysis …")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    G, mp_ids = build_annotated_graph()
    results   = compute_assortativity(G)
    print(f"  Assortativity results:")
    for k, v in results.items():
        print(f"    {k}: {v:.4f}")
    pl.DataFrame([{"attribute": k, "r": v, "threshold": THRESHOLD}
                  for k, v in results.items()]).write_parquet(ANALYSIS_DIR / "assortativity.parquet")
    fig27_assortativity(G, results)
    fig28_assortativity_thresh()
    print(f"\nDone. Figures in {FIG_DIR}")
