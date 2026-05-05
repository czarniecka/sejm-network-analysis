"""
Script 23 — Rich-club coefficient + small-world test.

Rich-club coefficient φ(k): among nodes with degree ≥ k, what fraction of
possible edges actually exist?  If φ_observed > φ_random → rich-club effect
(high-degree MPs preferentially connect with each other).

Small-world test (Watts-Strogatz):
  σ = (C/C_rand) / (L/L_rand)  where σ > 1 indicates a small-world network.

Figures:
  fig29_rich_club.png        — φ(k) observed vs random + normalised coefficient
  fig30_rich_club_members.png — who is in the rich club? top-N by degree
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
import polars as pl

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
N_RAND    = 20       # number of random graphs for null model


def build_graph(term: int = 10):
    agreement  = np.load(NETWORKS_DIR / "agreement_matrix.npy")
    copresence = np.load(NETWORKS_DIR / "copresence_matrix.npy")
    mp_ids     = np.load(NETWORKS_DIR / "mp_ids.npy").tolist()
    mps        = load_mps(term)
    id_to_club = dict(zip(mps["mp_id"].to_list(), mps["club"].cast(pl.Utf8).to_list()))
    id_to_name = {r["mp_id"]: f"{r['first_name'][0]}. {r['last_name']}"
                  for r in mps.iter_rows(named=True)}
    G = adjacency_to_networkx(agreement, THRESHOLD, mp_ids, MIN_COP, copresence)
    for i, mid in enumerate(mp_ids):
        G.nodes[i]["club"] = id_to_club.get(mid, "niez.")
        G.nodes[i]["name"] = id_to_name.get(mid, str(mid))
        G.nodes[i]["mp_id"] = mid
    return G, mp_ids, id_to_club, id_to_name


def compute_rich_club(G) -> tuple[dict, dict, dict]:
    """
    Returns:
      rc_obs   : {k: phi_observed}
      rc_rand  : {k: phi_random_mean}
      rc_norm  : {k: phi_obs / phi_rand}
    """
    print("  computing observed rich-club coefficient …")
    rc_obs = nx.rich_club_coefficient(G, normalized=False, seed=42)

    # Null model: degree-preserving random rewiring
    print(f"  building {N_RAND} random graphs for null model …")
    rand_rcs = []
    for seed in range(N_RAND):
        G_rand = G.copy()
        try:
            nx.double_edge_swap(G_rand, nswap=G.number_of_edges() * 10,
                                max_tries=G.number_of_edges() * 100, seed=seed)
        except Exception:
            pass
        rand_rcs.append(nx.rich_club_coefficient(G_rand, normalized=False, seed=seed))

    all_k = set(rc_obs.keys())
    for r in rand_rcs:
        all_k |= set(r.keys())
    all_k = sorted(all_k)

    rc_rand = {}
    for k in all_k:
        vals = [r.get(k, np.nan) for r in rand_rcs]
        vals = [v for v in vals if not np.isnan(v)]
        rc_rand[k] = float(np.mean(vals)) if vals else np.nan

    rc_norm = {}
    for k in all_k:
        obs  = rc_obs.get(k, np.nan)
        rand = rc_rand.get(k, np.nan)
        if not np.isnan(obs) and not np.isnan(rand) and rand > 0:
            rc_norm[k] = obs / rand
        else:
            rc_norm[k] = np.nan

    return rc_obs, rc_rand, rc_norm


def small_world_test(G) -> dict:
    """Compute small-world coefficient σ compared to random graphs."""
    print("  small-world test …")
    C = nx.average_clustering(G)
    components = list(nx.connected_components(G))
    lcc = G.subgraph(max(components, key=len)).copy()
    try:
        L = nx.average_shortest_path_length(lcc)
    except Exception:
        L = float("nan")

    # Random graph with same n, m
    n, m = G.number_of_nodes(), G.number_of_edges()
    C_rand_vals, L_rand_vals = [], []
    for seed in range(10):
        G_r = nx.gnm_random_graph(n, m, seed=seed)
        C_rand_vals.append(nx.average_clustering(G_r))
        lcc_r = G_r.subgraph(max(nx.connected_components(G_r), key=len)).copy()
        try:
            L_rand_vals.append(nx.average_shortest_path_length(lcc_r))
        except Exception:
            pass

    C_rand = float(np.mean(C_rand_vals)) if C_rand_vals else np.nan
    L_rand = float(np.mean(L_rand_vals)) if L_rand_vals else np.nan

    sigma  = (C / C_rand) / (L / L_rand) if (C_rand > 0 and L_rand > 0 and L > 0) else np.nan
    omega  = L_rand / L - C / C_rand if (L > 0 and C_rand > 0) else np.nan  # Telesford's ω

    return {
        "C_observed": C, "C_random": C_rand, "C_ratio": C / C_rand if C_rand > 0 else np.nan,
        "L_observed": L, "L_random": L_rand, "L_ratio": L / L_rand if L_rand > 0 else np.nan,
        "sigma": sigma, "omega": omega,
        "n_nodes": n, "n_edges": m,
    }


def fig29_rich_club(rc_obs, rc_rand, rc_norm, sw: dict) -> None:
    print("  fig29: rich-club coefficient …")
    ks_obs  = sorted(k for k in rc_obs  if not np.isnan(rc_obs[k]))
    ks_norm = sorted(k for k in rc_norm if not np.isnan(rc_norm[k]))

    fig = plt.figure(figsize=(14, 6.5), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(BG2)
    ax1.plot(ks_obs, [rc_obs[k]  for k in ks_obs], color=RED, linewidth=2.5,
             label="Obserwowane φ(k)", zorder=3)
    ks_r = sorted(k for k in rc_rand if not np.isnan(rc_rand.get(k, np.nan)))
    ax1.plot(ks_r,   [rc_rand[k] for k in ks_r],   color=GREY, linewidth=2.0,
             linestyle="--", label="Losowe φ(k) (null model)", zorder=2)
    ax1.fill_between(ks_obs, [rc_obs[k] for k in ks_obs],
                     [rc_rand.get(k, np.nan) for k in ks_obs],
                     alpha=0.15, color=RED, label="Nadwyżka rich-club")
    ax1.set_xlabel("Stopień k (minimalna liczba sąsiadów)", fontsize=10)
    ax1.set_ylabel("φ(k) — rich-club coefficient", fontsize=10)
    ax1.set_title(f"Rich-club coefficient obserwowany vs losowy\n(próg {THRESHOLD:.0%})", fontsize=12)
    ax1.legend(fontsize=9, framealpha=0.5)
    ax1.grid(alpha=0.25); ax1.spines[["top", "right"]].set_visible(False)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(BG2)
    ax2.plot(ks_norm, [rc_norm[k] for k in ks_norm], color=RED, linewidth=2.5,
             marker="o", markersize=3.5)
    ax2.axhline(1.0, color=GREY, linewidth=1.2, linestyle="--", alpha=0.7,
                label="φ_norm = 1  (jak losowy)")
    ax2.fill_between(ks_norm, 1, [rc_norm[k] for k in ks_norm],
                     where=[rc_norm[k] > 1 for k in ks_norm],
                     alpha=0.15, color=RED, label="Rich-club efekt")
    ax2.set_xlabel("Stopień k", fontsize=10)
    ax2.set_ylabel("φ_norm(k) = φ_obs / φ_rand", fontsize=10)
    ax2.set_title("Znormalizowany rich-club coefficient\n(> 1 = efekt rich-club)", fontsize=12)
    ax2.legend(fontsize=9, framealpha=0.5)
    ax2.grid(alpha=0.25); ax2.spines[["top", "right"]].set_visible(False)

    # Small-world annotation
    sw_text = (f"Small-world test:\n"
               f"  C = {sw['C_observed']:.3f}  (C_rand = {sw['C_random']:.3f})\n"
               f"  L = {sw['L_observed']:.3f}  (L_rand = {sw['L_random']:.3f})\n"
               f"  C/C_rand = {sw['C_ratio']:.2f}   L/L_rand = {sw['L_ratio']:.2f}\n"
               f"  σ = {sw['sigma']:.2f}  {'→ SMALL-WORLD' if sw['sigma'] > 1 else '→ nie small-world'}")
    ax2.text(0.97, 0.05, sw_text, transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=8.5, color=DARK, bbox=dict(boxstyle="round", facecolor=BG2,
             edgecolor=GRID, alpha=0.85))

    fig.suptitle("Rich-club phenomenon + small-world test  |  Kadencja X",
                 fontsize=14, color=DARK, y=1.01)
    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig29_rich_club.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    saved {out}")


def fig30_rich_club_members(G, rc_norm: dict) -> None:
    print("  fig30: rich-club members …")
    # Find k* = k with max normalized rc
    valid = {k: v for k, v in rc_norm.items() if not np.isnan(v) and k > 5}
    if not valid:
        print("    no valid rich-club data"); return
    k_star = max(valid, key=lambda k: valid[k])

    # Members of rich-club: nodes with degree >= k*
    degrees = dict(G.degree())
    rich_nodes = [n for n, d in degrees.items() if d >= k_star]
    print(f"    k* = {k_star}, rich-club size = {len(rich_nodes)}")

    rich_data = [{"name": G.nodes[n]["name"], "club": G.nodes[n]["club"],
                  "degree": degrees[n], "mp_id": G.nodes[n]["mp_id"]}
                 for n in rich_nodes]
    rich_df = pl.DataFrame(rich_data).sort("degree", descending=True)

    top_n = min(40, len(rich_data))
    top_df= rich_df.head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(15, max(7, top_n * 0.22 + 2)), facecolor=BG)
    fig.suptitle(
        f"Członkowie Rich Club (stopień ≥ {k_star})  |  {len(rich_nodes)} posłów  |  Kadencja X",
        fontsize=14, color=DARK, y=1.01,
    )

    ax1 = axes[0]
    ax1.set_facecolor(BG2)
    names   = top_df["name"].to_list()
    degs    = top_df["degree"].to_numpy()
    clubs_t = top_df["club"].to_list()
    cols    = [CLUB_COLOURS.get(c, GREY) for c in clubs_t]
    y       = np.arange(len(names))
    ax1.barh(y, degs, color=cols, alpha=0.85, edgecolor="#333", linewidth=0.4, height=0.7)
    for i, d in enumerate(degs):
        ax1.text(d + 1, i, str(d), va="center", fontsize=8)
    ax1.set_yticks(y); ax1.set_yticklabels([f"{n}  [{c}]" for n, c in zip(names, clubs_t)],
                                            fontsize=8)
    ax1.set_xlabel("Stopień węzła (liczba sąsiadów)", fontsize=10)
    ax1.set_title(f"Top {top_n} posłów wg stopnia węzła", fontsize=11)
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.3); ax1.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    ax2.set_facecolor(BG2)
    club_counts = (rich_df.filter(pl.col("club").is_in(MAIN_CLUBS))
                   .group_by("club").agg(pl.len().alias("n_rich"))
                   .sort("n_rich", descending=True))
    total_club  = (pl.DataFrame([{"club": G.nodes[n]["club"], "mp_id": G.nodes[n]["mp_id"]}
                                  for n in G.nodes()])
                   .filter(pl.col("club").is_in(MAIN_CLUBS))
                   .group_by("club").agg(pl.len().alias("total")))
    cc = club_counts.join(total_club, on="club", how="left").with_columns(
        (pl.col("n_rich") / pl.col("total")).alias("frac")
    ).sort("frac", descending=True)
    clubs_c = cc["club"].to_list()
    fracs_c = cc["frac"].to_numpy()
    ns_c    = cc["n_rich"].to_list()
    y2      = np.arange(len(clubs_c))
    cols2   = [CLUB_COLOURS.get(c, GREY) for c in clubs_c]
    ax2.barh(y2, fracs_c, color=cols2, alpha=0.85, edgecolor="#333", linewidth=0.4, height=0.65)
    for i, (f, n) in enumerate(zip(fracs_c, ns_c)):
        ax2.text(f + 0.01, i, f"{f:.0%}  (n={n})", va="center", fontsize=9)
    ax2.set_yticks(y2); ax2.set_yticklabels(clubs_c, fontsize=9.5)
    ax2.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax2.set_xlabel("Odsetek posłów klubu w rich club", fontsize=10)
    ax2.set_title(f"Skład klubowy rich club (stopień ≥ {k_star})", fontsize=11)
    ax2.set_xlim(0, 1.15)
    ax2.grid(axis="x", alpha=0.3); ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig30_rich_club_members.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    saved {out}")


if __name__ == "__main__":
    print("Script 23 — Rich-club coefficient + small-world test …")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    G, mp_ids, id_to_club, id_to_name = build_graph()
    rc_obs, rc_rand, rc_norm = compute_rich_club(G)
    sw = small_world_test(G)
    print(f"  Small-world σ = {sw['sigma']:.3f}  (> 1 = small world)")

    pl.DataFrame([{"k": k, "phi_obs": rc_obs.get(k, None),
                   "phi_rand": rc_rand.get(k, None), "phi_norm": rc_norm.get(k, None)}
                  for k in sorted(rc_obs.keys())]).write_parquet(ANALYSIS_DIR / "rich_club.parquet")
    pl.DataFrame([sw]).write_parquet(ANALYSIS_DIR / "small_world.parquet")

    fig29_rich_club(rc_obs, rc_rand, rc_norm, sw)
    fig30_rich_club_members(G, rc_norm)
    print(f"\nDone. Figures in {FIG_DIR}")
