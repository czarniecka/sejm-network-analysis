"""
Script 13 — Poster-quality visualizations.
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
from src.scripts.poster_style import apply_style, CLUB_COLOURS, cc, MAIN_CLUBS, PALETTE, COALITION, OPPOSITION, club_en

apply_style()

FIG_DIR = PROJECT_ROOT / "data" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Agreement network
# ═══════════════════════════════════════════════════════════════════════════════
def _build_graph(agreement, copresence, mp_ids, id_to_club, thresh: float, min_cop: int = 50):
    N = len(mp_ids)
    G = nx.Graph()
    for i in range(N):
        G.add_node(int(mp_ids[i]), club=id_to_club.get(int(mp_ids[i]), "niez."))
    for i in range(N):
        for j in range(i + 1, N):
            if copresence[i, j] >= min_cop and agreement[i, j] >= thresh:
                G.add_edge(int(mp_ids[i]), int(mp_ids[j]), weight=float(agreement[i, j]))
    return G


_LAYOUT_BLOCS = {
    "coalition": {"KO", "PSL-TD", "Lewica", "Polska2050", "Polska2050-TD", "Razem"},
    "opposition": {"PiS", "Konfederacja", "Konfederacja_KP"},
}

def _compact_layout(G, seed: int = 42) -> dict:
    """
    Club-seeded layout: initialise each node in a bloc-specific region so that
    coalition (left half) and opposition (right half) start separated.
    A short spring relaxation (unweighted) then spreads nodes within each bloc
    without allowing dense cliques to collapse into a single ball.
    """
    import random as _random
    rng = _random.Random(seed)
    coalition  = _LAYOUT_BLOCS["coalition"]
    opposition = _LAYOUT_BLOCS["opposition"]

    # Assign initial positions by bloc (coalition left, opposition right, others centre)
    init_pos: dict = {}
    for node in G.nodes():
        club = G.nodes[node].get("club", "niez.")
        if club in coalition:
            x = rng.uniform(-0.9, -0.05)
            y = rng.uniform(-0.85, 0.85)
        elif club in opposition:
            x = rng.uniform(0.05, 0.9)
            y = rng.uniform(-0.85, 0.85)
        else:
            x = rng.uniform(-0.12, 0.12)
            y = rng.uniform(-0.85, 0.85)
        init_pos[node] = np.array([x, y])

    # Short unweighted relaxation — spreads within-bloc positions without
    # collapsing dense cliques; few iterations prevent inter-bloc convergence
    pos = nx.spring_layout(G, pos=init_pos, seed=seed,
                           k=0.18, iterations=25, weight=None)
    return pos


def _draw_network_ax(ax, G, pos, node_colours, node_sizes, title: str, legend_handles=None):
    """Draw one network panel: black edges, club-coloured nodes."""
    ax.set_facecolor("white")
    ax.axis("off")
    # Edges — thin black, semi-transparent
    for u, v in G.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#222222", alpha=0.07, linewidth=0.25, zorder=1)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colours,
                           node_size=node_sizes, linewidths=0.3,
                           edgecolors="#000000", alpha=0.92)
    # Set tight axis limits around actual node positions (with small padding)
    if pos:
        all_xs = [p[0] for p in pos.values()]
        all_ys = [p[1] for p in pos.values()]
        pad = max(max(all_xs) - min(all_xs), max(all_ys) - min(all_ys)) * 0.06 + 0.02
        ax.set_xlim(min(all_xs) - pad, max(all_xs) + pad)
        ax.set_ylim(min(all_ys) - pad, max(all_ys) + pad)
    ax.set_title(title, fontsize=13, pad=12, color=PALETTE["dark"])
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower left", fontsize=8,
                  framealpha=0.7, ncol=2, title="Club", title_fontsize=8.5)


def fig1_network():
    print("  fig1: network graph …")
    agreement  = np.load(NETWORKS_DIR / "agreement_matrix.npy")
    copresence = np.load(NETWORKS_DIR / "copresence_matrix.npy")
    mp_ids     = np.load(NETWORKS_DIR / "mp_ids.npy")
    mps_df     = pl.read_parquet(PARQUET_DIR / "term10/mps.parquet")
    id_to_club = dict(zip(mps_df["mp_id"].to_list(), mps_df["club"].cast(pl.Utf8).to_list()))

    bt_cmap = mpl.colormaps.get_cmap("plasma")

    for thresh in [0.70, 0.99]:
        G_full = _build_graph(agreement, copresence, mp_ids, id_to_club, thresh)
        # Remove isolated nodes
        G = G_full.copy()
        isolates = list(nx.isolates(G))
        G.remove_nodes_from(isolates)
        n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
        print(f"    thresh={thresh:.0%}: {n_nodes} nodes (−{len(isolates)} isolated), {n_edges:,} edges")

        pos = _compact_layout(G, seed=42)

        # ── club-coloured variant ───────────────────────────────────────────
        node_colours = [cc(G.nodes[n]["club"]) for n in G.nodes()]
        deg_c = nx.degree_centrality(G)
        node_sizes = [12 + 80 * deg_c[n] for n in G.nodes()]

        seen = {}
        for n in G.nodes():
            c = G.nodes[n]["club"]
            if c not in seen:
                seen[c] = cc(c)
        handles = [mpatches.Patch(color=col, label=club) for club, col in sorted(seen.items())]

        fig, ax = plt.subplots(figsize=(14, 12), facecolor="white")
        _draw_network_ax(ax, G, pos, node_colours, node_sizes,
                         title=f"Voting network — threshold {int(thresh*100)}%  |  Term X\n"
                               f"{n_nodes} MPs  ·  {n_edges:,} edges",
                         legend_handles=handles)
        out = FIG_DIR / f"fig1_network_{int(thresh*100)}.png"
        fig.tight_layout(pad=0.5)
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"    saved {out}")

        # ── betweenness-centrality variant ─────────────────────────────────
        print(f"    computing betweenness centrality (thresh={thresh:.0%}) …")
        bt = nx.betweenness_centrality(G, normalized=True, weight="weight")
        bt_vals = np.array([bt[n] for n in G.nodes()])
        norm_bt = mpl.colors.Normalize(vmin=bt_vals.min(), vmax=bt_vals.max())
        bt_colours = [bt_cmap(norm_bt(v)) for v in bt_vals]
        bt_sizes   = [15 + 200 * bt[n] for n in G.nodes()]

        fig, ax = plt.subplots(figsize=(14, 12), facecolor="white")
        _draw_network_ax(ax, G, pos, bt_colours, bt_sizes,
                         title=f"Betweenness centrality — threshold {int(thresh*100)}%  |  Term X\n"
                               f"{n_nodes} MPs  ·  {n_edges:,} edges")
        sm = mpl.cm.ScalarMappable(cmap=bt_cmap, norm=norm_bt)
        plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, shrink=0.55,
                     label="Betweenness centrality")
        # Annotate top-10 nodes by betweenness
        top10 = sorted(bt.items(), key=lambda x: x[1], reverse=True)[:10]
        for node_id, _ in top10:
            lbl = mps_df.filter(pl.col("mp_id") == node_id)
            if len(lbl) > 0:
                r = lbl.row(0, named=True)
                name = f"{r['first_name'][0]}. {r['last_name']}"
            else:
                name = str(node_id)
            x, y = pos[node_id]
            ax.annotate(name, (x, y), xytext=(0, 5), textcoords="offset points",
                        fontsize=6.5, ha="center", color=PALETTE["dark"], zorder=5)
        out = FIG_DIR / f"fig1_betweenness_{int(thresh*100)}.png"
        fig.tight_layout(pad=0.5)
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
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

    fig, ax = plt.subplots(figsize=(10, 8.5), facecolor="white")
    ax.set_facecolor("white")
    im = ax.imshow(mat, cmap=cmap, vmin=0.3, vmax=1.0, aspect="auto")

    short = [c.replace("Konfederacja", "Konf.").replace("Polska2050", "P2050") for c in clubs]
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=9)

    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                col = "white" if mat[i, j] > 0.65 else PALETTE["dark"]
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7.5, color=col)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Voting agreement", color=PALETTE["dark"], fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["dark"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["dark"])

    ax.set_title("Inter-party agreement matrix\nPolish Sejm, Term X", fontsize=13, pad=12)
    fig.tight_layout(pad=1.0)
    out = FIG_DIR / "fig2_party_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
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

    fig = plt.figure(figsize=(14, 6.5), facecolor="white")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

    # left — bar chart
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("white")
    names   = [f"{r['first_name'][0]}. {r['last_name']}" for r in top.iter_rows(named=True)]
    rates   = top["rebel_rate"].to_list()
    colours = [cc(c) for c in top["club"].cast(pl.Utf8).to_list()]
    y = np.arange(len(names))
    bars = ax1.barh(y, rates, color=colours, edgecolor="#111", linewidth=0.4, height=0.65)
    for bar, rate in zip(bars, rates):
        ax1.text(rate + 0.007, bar.get_y() + bar.get_height() / 2,
                 f"{rate:.1%}", va="center", fontsize=8.5, color=PALETTE["dark"])
    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=9.5)
    ax1.set_xlabel("Rebel rate", fontsize=10)
    ax1.set_title("Top 15 rebels", fontsize=13)
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
    ax2.set_facecolor("white")
    valid   = centrality.filter(pl.col("rebel_rate").is_not_null())
    xarr    = valid["eigenvector"].to_numpy()
    yarr    = valid["rebel_rate"].to_numpy()
    cols_sc = [cc(c) for c in valid["club"].cast(pl.Utf8).to_list()]
    ax2.scatter(xarr, yarr, c=cols_sc, s=20, alpha=0.7, linewidths=0)
    mask = ~np.isnan(xarr) & ~np.isnan(yarr)
    coef = np.polyfit(xarr[mask], yarr[mask], 1)
    xfit = np.linspace(xarr[mask].min(), xarr[mask].max(), 100)
    ax2.plot(xfit, np.polyval(coef, xfit), color=PALETTE["secondary"], linewidth=1.8, linestyle="--")
    ax2.set_xlabel("Eigenvector centrality", fontsize=10)
    ax2.set_ylabel("Rebel rate", fontsize=10)
    ax2.set_title("Rebels vs network position\nρ = −0.78  (p < 10⁻¹⁰⁰)", fontsize=11)
    ax2.grid(alpha=0.25)
    ax2.spines[["top", "right"]].set_visible(False)
    labels = [f"{r['first_name'][0]}. {r['last_name']}" for r in valid.iter_rows(named=True)]
    for k in np.argsort(yarr)[::-1][:4]:
        ax2.annotate(labels[k], (xarr[k], yarr[k]),
                     xytext=(5, 3), textcoords="offset points", fontsize=8, color=PALETTE["accent"])

    fig.suptitle("Voting loyalty — Sejm rebels  |  Term X",
                 fontsize=14, y=1.01, color=PALETTE["dark"])
    fig.tight_layout(pad=1.0)
    out = FIG_DIR / "fig3_rebels.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Temporal evolution
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_temporal():
    print("  fig4: temporal …")
    df = pl.read_parquet(ANALYSIS_DIR / "temporal_metrics.parquet")
    df = df.filter(pl.col("n_edges") > 0)

    thresholds = [0.5, 0.7, 0.9, 0.99]
    th_cols    = ["#cccccc", PALETTE["accent"], "#7a0000", PALETTE["primary"]]
    th_labels  = ["threshold 50%", "threshold 70%", "threshold 90%", "threshold 99%"]

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.32)

    configs = [
        (gs[0, 0], "density",        "Network density",               "Density"),
        (gs[0, 1], "n_components",   "Connected components",          "Components"),
        (gs[1, 0], "avg_clustering", "Mean clustering coefficient",   "Clustering"),
        (gs[1, 1], "modularity",     "Modularity (Leiden)",           "Modularity"),
    ]

    for gspec, col, title, ylabel in configs:
        ax = fig.add_subplot(gspec)
        ax.set_facecolor("white")
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

    fig.suptitle("Voting network evolution over time  |  Term X  (Nov 2023 – Apr 2026)",
                 fontsize=14, y=1.01, color=PALETTE["dark"])
    fig.tight_layout(pad=1.0)
    out = FIG_DIR / "fig4_temporal.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Intra-party cohesion (Y axis 0.75–1.0 for readability)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_cohesion():
    print("  fig5: cohesion …")
    df = pl.read_parquet(ANALYSIS_DIR / "party_cohesion_by_month.parquet")
    main_clubs = ["KO", "PiS", "PSL-TD", "Lewica", "Polska2050-TD", "Konfederacja", "Razem"]
    all_months = sorted(df["year_month"].unique().to_list())  # full range before club filter
    df = df.filter(pl.col("club").cast(pl.Utf8).is_in(main_clubs))

    # ── political context periods: (start, end, label, colour) ──────────────
    PERIODS = [
        ("2024-01", "2024-04", "Farmers'\nstrikes",      "#b5651d"),
        ("2024-06", "2024-09", "Konfederacja\nsplit",     "#9467bd"),
        ("2025-01", "2025-06", "Presidential\ncampaign", PALETTE["accent"]),
    ]

    fig, ax = plt.subplots(figsize=(15, 6.5), facecolor="white")
    ax.set_facecolor("white")

    # Shade periods behind data lines
    for start, end, label, col in PERIODS:
        xs = all_months.index(start) if start in all_months else -0.5
        xe = all_months.index(end)   if end   in all_months else len(all_months) - 0.5
        ax.axvspan(xs - 0.5, xe + 0.5, color=col, alpha=0.10, zorder=0)
        ax.axvline(xs - 0.5, color=col, linewidth=1.4, linestyle="--", alpha=0.55, zorder=1)
        ax.axvline(xe + 0.5, color=col, linewidth=1.4, linestyle="--", alpha=0.55, zorder=1)
        ax.text((xs + xe) / 2, 1.025, label, color=col, fontsize=8.5,
                ha="center", va="bottom", fontweight="bold",
                transform=ax.get_xaxis_transform())

    line_styles = ["-", "--", "-.", ":", "-", "--", "-."]
    for i, club in enumerate(main_clubs):
        sub = df.filter(pl.col("club").cast(pl.Utf8) == club).sort("year_month")
        if len(sub) == 0:
            continue
        months = sub["year_month"].to_list()
        vals   = sub["cohesion_score"].to_list()
        x_pos  = [all_months.index(m) for m in months]
        ax.plot(x_pos, vals, color=cc(club), linewidth=2.2, label=club,
                marker="o", markersize=4.5, alpha=0.92,
                linestyle=line_styles[i % len(line_styles)], zorder=2)

    ax.set_xticks(range(0, len(all_months), 3))
    ax.set_xticklabels(all_months[::3], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Voting cohesion (Rice index)", fontsize=10)
    ax.set_ylim(0.72, 1.03)
    ax.set_yticks(np.arange(0.75, 1.01, 0.05))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0.75, 1.01, 0.05)], fontsize=9)
    # title removed — section heading is provided by the poster layout
    ax.grid(alpha=0.20, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9.5, framealpha=0.6, ncol=2, loc="lower left")

    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig5_cohesion.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5b — Cohesion zoom: pre-presidential election period
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_cohesion_presidential_zoom():
    print("  fig5b: cohesion zoom (presidential election) …")
    df = pl.read_parquet(ANALYSIS_DIR / "party_cohesion_by_month.parquet")
    main_clubs = ["KO", "PiS", "PSL-TD", "Lewica", "Polska2050-TD", "Konfederacja", "Razem"]
    df = df.filter(pl.col("club").cast(pl.Utf8).is_in(main_clubs))

    # Zoom window: 2024-09 to 2025-07 (3 months before and after election)
    ZOOM_START = "2024-09"
    ZOOM_END   = "2025-07"
    all_months = sorted(df["year_month"].unique().to_list())
    zoom_months = [m for m in all_months if ZOOM_START <= m <= ZOOM_END]

    # Presidential election dates
    ROUND1 = "2025-05"   # 18 May 2025
    ROUND2 = "2025-06"   # 1 June 2025

    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor="white")
    ax.set_facecolor("white")

    line_styles = ["-", "--", "-.", ":", "-", "--", "-."]
    for i, club in enumerate(main_clubs):
        sub = (df.filter(pl.col("club").cast(pl.Utf8) == club)
                 .sort("year_month")
                 .filter(pl.col("year_month").is_in(zoom_months)))
        if len(sub) == 0:
            continue
        months = sub["year_month"].to_list()
        vals   = sub["cohesion_score"].to_list()
        x_pos  = [zoom_months.index(m) for m in months]
        ax.plot(x_pos, vals, color=cc(club), linewidth=2.4, label=club,
                marker="o", markersize=6, alpha=0.93,
                linestyle=line_styles[i % len(line_styles)])

    # Mark election rounds
    for rnd_month, label, offset in [
        (ROUND1, "Round 1\n18 May", -0.35),
        (ROUND2, "Round 2\n1 Jun",  +0.15),
    ]:
        if rnd_month in zoom_months:
            xv = zoom_months.index(rnd_month)
            ax.axvline(xv, color=PALETTE["accent"], linewidth=1.8,
                       linestyle="--", alpha=0.75, zorder=3)
            ax.text(xv + offset, 0.745, label, color=PALETTE["accent"],
                    fontsize=8.5, ha="left", va="bottom", style="italic")

    # Shade the two election months
    for rnd_month in [ROUND1, ROUND2]:
        if rnd_month in zoom_months:
            xv = zoom_months.index(rnd_month)
            ax.axvspan(xv - 0.5, xv + 0.5,
                       color=PALETTE["accent"], alpha=0.06, zorder=0)

    ax.set_xticks(range(len(zoom_months)))
    ax.set_xticklabels(zoom_months, rotation=45, ha="right", fontsize=9.5)
    ax.set_ylabel("Voting cohesion (Rice index)", fontsize=11)
    ax.set_ylim(0.72, 1.03)
    ax.set_yticks(np.arange(0.75, 1.01, 0.05))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0.75, 1.01, 0.05)], fontsize=9.5)
    # title removed — section heading is provided by the poster layout
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.6, ncol=2, loc="lower left")

    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig5b_cohesion_presidential_zoom.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
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

    fig = plt.figure(figsize=(15, 7.5), facecolor="white")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.48)

    for ax_idx, (descending, subtitle) in enumerate(
        [(False, "Most controversial"), (True, "Greatest consensus")]
    ):
        ax  = fig.add_subplot(gs[ax_idx])
        ax.set_facecolor("white")
        sub = df.sort("mean_pair_agreement", descending=descending).head(12)
        labels = [shorten(r["top_words"]) for r in sub.iter_rows(named=True)]
        vals   = sub["mean_pair_agreement"].to_list()
        n_v    = sub["n_votings"].to_list()
        cols   = [red_cmap(norm(v)) for v in vals]

        y    = np.arange(len(labels))
        bars = ax.barh(y, vals, color=cols, edgecolor="#111", linewidth=0.3, height=0.65)
        for bar, v, nv in zip(bars, vals, n_v):
            ax.text(v + 0.008, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}  (n={nv})", va="center", fontsize=8.5, color=PALETTE["dark"])

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Mean pairwise MP agreement", fontsize=10)
        ax.set_title(subtitle, fontsize=12)
        ax.set_xlim(0, 1.18)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Voting topics — controversy (BERTopic)  |  Term X",
                 fontsize=14, y=1.01, color=PALETTE["dark"])
    fig.tight_layout(pad=1.0)
    out = FIG_DIR / "fig6_topics.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
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

    fig, ax1 = plt.subplots(figsize=(11, 5.5), facecolor="white")
    ax1.set_facecolor("white")

    ax1.plot(thresholds, n_edges, color=PALETTE["accent"], linewidth=2.5)
    ax1.fill_between(thresholds, n_edges, alpha=0.12, color=PALETTE["accent"])

    for t in [0.30, 0.50, 0.70, 0.90, 0.99]:
        n = int((valid_agreement >= t).sum())
        ax1.axvline(t, color=PALETTE["neutral"], linewidth=1.0, linestyle="--", alpha=0.7)
        ax1.annotate(f"{t:.0%}\n{n:,}", xy=(t, n),
                     xytext=(6, 6), textcoords="offset points",
                     fontsize=8.5, color=PALETTE["neutral"])

    ax2 = ax1.twinx()
    ax2.set_facecolor("white")
    density = [e / max_edges for e in n_edges]
    ax2.plot(thresholds, density, color=PALETTE["secondary"], linewidth=1.5, linestyle=":")
    ax2.set_ylabel("Graph density", fontsize=10, color=PALETTE["neutral"])
    ax2.tick_params(colors=PALETTE["neutral"])
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_tick_params(labelcolor=PALETTE["neutral"])

    ax1.set_xlabel("Agreement threshold", fontsize=11)
    ax1.set_ylabel("Number of edges", fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, max_edges * 1.05)
    ax1.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax1.set_title(
        f"Number of edges in the network by agreement threshold\n"
        f"Term X  |  {len(agreement)} MPs  ·  {max_edges:,} pairs with min. {MIN_COP} joint votes",
        fontsize=13, pad=12,
    )
    ax1.grid(alpha=0.25)
    ax1.spines[["top", "right"]].set_visible(False)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=PALETTE["accent"], linewidth=2.5, label="Number of edges"),
        Line2D([0], [0], color=PALETTE["secondary"], linewidth=1.5, linestyle=":", label="Graph density"),
    ]
    ax1.legend(handles=legend_handles, fontsize=9.5, framealpha=0.5, loc="upper right")

    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig10_edges_vs_threshold.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1c — Community detection (Leiden) overlaid on betweenness network
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_community_betweenness(thresh: float = 0.70):
    """
    One figure per threshold.
    Node colour  = Leiden community (auto-detected from topology only)
    Node size    = betweenness centrality
    Node border  = club colour (from CLUB_COLOURS)
    Border width = thick (3 px) if Leiden community ≠ party bloc, thin (0.5 px) if match
    NMI between Leiden communities and club labels shown in title.
    """
    import leidenalg, igraph as ig
    from sklearn.metrics import normalized_mutual_info_score
    from matplotlib.lines import Line2D

    print(f"  fig1c community+betweenness (thresh={thresh:.0%}) …")
    agreement  = np.load(NETWORKS_DIR / "agreement_matrix.npy")
    copresence = np.load(NETWORKS_DIR / "copresence_matrix.npy")
    mp_ids     = np.load(NETWORKS_DIR / "mp_ids.npy")
    mps_df     = pl.read_parquet(PARQUET_DIR / "term10/mps.parquet")
    id_to_club = dict(zip(mps_df["mp_id"].to_list(), mps_df["club"].cast(pl.Utf8).to_list()))

    # ── build networkx graph, drop isolates ───────────────────────────────────
    G_full = _build_graph(agreement, copresence, mp_ids, id_to_club, thresh)
    G = G_full.copy()
    G.remove_nodes_from(list(nx.isolates(G)))
    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {v: i for i, v in enumerate(nodes)}

    # ── Leiden community detection via igraph ─────────────────────────────────
    ig_graph = ig.Graph(n=n, directed=False)
    for u, v, d in G.edges(data=True):
        ig_graph.add_edge(node_idx[u], node_idx[v], weight=d["weight"])
    partition = leidenalg.find_partition(
        ig_graph, leidenalg.ModularityVertexPartition,
        weights="weight", seed=42,
    )
    comm_labels = [0] * n
    for comm_id, members in enumerate(partition):
        for m in members:
            comm_labels[m] = comm_id
    n_comms = max(comm_labels) + 1
    print(f"    Leiden: {n_comms} communities")

    # Map node → community
    node_to_comm = {nodes[i]: comm_labels[i] for i in range(n)}

    # ── betweenness centrality ────────────────────────────────────────────────
    print(f"    computing betweenness …")
    bt = nx.betweenness_centrality(G, normalized=True, weight="weight")

    # ── NMI: Leiden community vs club ─────────────────────────────────────────
    clubs_ordered = [id_to_club.get(v, "niez.") for v in nodes]
    nmi = normalized_mutual_info_score(clubs_ordered, comm_labels)
    print(f"    NMI (Leiden vs club) = {nmi:.3f}")

    # ── majority club per community ───────────────────────────────────────────
    from collections import Counter
    comm_majority_club = {}
    for c in range(n_comms):
        members_clubs = [id_to_club.get(nodes[i], "niez.")
                         for i, lbl in enumerate(comm_labels) if lbl == c]
        comm_majority_club[c] = Counter(members_clubs).most_common(1)[0][0]

    # Community fill colour = CLUB_COLOURS of majority club → same as border when matching
    comm_fill_colour = {c: cc(comm_majority_club[c]) for c in range(n_comms)}

    # Per-node: fill = community colour, border = actual club colour
    node_fill   = [comm_fill_colour[node_to_comm[v]] for v in nodes]
    node_border = [cc(id_to_club.get(v, "niez.")) for v in nodes]
    node_sizes  = [20 + 2500 * bt[v] for v in nodes]

    # Mismatch = fill colour ≠ border colour (immediately visible)
    def is_mismatch(node_id):
        return id_to_club.get(node_id, "niez.") != comm_majority_club[node_to_comm[node_id]]

    border_widths = [2.5 if is_mismatch(v) else 0.4 for v in nodes]
    n_mismatches  = sum(1 for v in nodes if is_mismatch(v))
    print(f"    mismatches: {n_mismatches}/{n} ({100*n_mismatches/n:.1f}%)")

    # ── layout ────────────────────────────────────────────────────────────────
    pos = _compact_layout(G, seed=42)

    # ── draw ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 13), facecolor="white")
    ax.set_facecolor("white")
    ax.axis("off")

    # Edges — thin black
    for u, v in G.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#111111", alpha=0.04, linewidth=0.2, zorder=1)

    # Draw nodes one by one (per-node border width)
    for i, v in enumerate(nodes):
        ax.scatter(pos[v][0], pos[v][1],
                   s=node_sizes[i], c=node_fill[i],
                   edgecolors=node_border[i], linewidths=border_widths[i],
                   alpha=0.92, zorder=2)

    # Set tight axis limits
    all_xs = [p[0] for p in pos.values()]
    all_ys = [p[1] for p in pos.values()]
    pad = max(max(all_xs) - min(all_xs), max(all_ys) - min(all_ys)) * 0.06 + 0.02
    ax.set_xlim(min(all_xs) - pad, max(all_xs) + pad)
    ax.set_ylim(min(all_ys) - pad, max(all_ys) + pad)

    # Annotate top-12 by betweenness
    top12 = sorted(bt.items(), key=lambda x: x[1], reverse=True)[:12]
    for node_id, _ in top12:
        row = mps_df.filter(pl.col("mp_id") == node_id)
        name = f"{row['first_name'][0][0]}. {row['last_name'][0]}" if len(row) > 0 else str(node_id)
        ax.annotate(name, pos[node_id], xytext=(0, 7), textcoords="offset points",
                    fontsize=8, ha="center", color=PALETTE["dark"],
                    fontweight="bold", zorder=5)

    # ── legends ───────────────────────────────────────────────────────────────
    comm_sizes   = Counter(comm_labels)
    sorted_comms = sorted(comm_sizes, key=lambda c: comm_sizes[c], reverse=True)

    # Community legend: fill = community colour
    comm_handles = [
        mpatches.Patch(
            facecolor=comm_fill_colour[c],
            edgecolor=comm_fill_colour[c], linewidth=1,
            label=f"Community {c+1}  (n={comm_sizes[c]}, majority: {comm_majority_club[c]})",
        )
        for c in sorted_comms
    ]
    leg1 = ax.legend(handles=comm_handles, loc="lower left",
                     title="Leiden communities  (fill colour)", title_fontsize=11,
                     fontsize=10, framealpha=0.88, ncol=2,
                     bbox_to_anchor=(0.0, 0.0))

    # Club border legend
    border_handles = [
        mpatches.Patch(facecolor="#dddddd", edgecolor=cc(club), linewidth=2, label=club)
        for club in MAIN_CLUBS
    ] + [
        mpatches.Patch(facecolor="#dddddd", edgecolor="#333333", linewidth=2.5,
                       label=f"Mismatch border  (n={n_mismatches})"),
    ]
    ax.add_artist(leg1)
    # Club border legend moved to upper-left (above community legend)
    ax.legend(handles=border_handles, loc="upper left",
              title="Official club  (border colour)", title_fontsize=11,
              fontsize=10, framealpha=0.88, ncol=2)

    # title removed — provided by poster layout
    fig.tight_layout(pad=0.8)

    # ── bridge MP table: placed AFTER tight_layout in absolute figure coords ──
    # This guarantees true bottom-right positioning independent of axes layout.
    mismatch_items = sorted(
        [(v, bt[v], id_to_club.get(v, "niez."))
         for v in nodes if is_mismatch(v)],
        key=lambda x: -x[1]
    )[:10]
    table_rows = []
    for rank, (node_id, bt_val, actual_club) in enumerate(mismatch_items, 1):
        row = mps_df.filter(pl.col("mp_id") == node_id)
        name = f"{row['first_name'][0][0]}. {row['last_name'][0]}" if len(row) > 0 else str(node_id)
        table_rows.append([str(rank), name, actual_club, f"{bt_val:.4f}"])

    if table_rows:
        # inset_axes with bottom=0.0 pins table to the axes bottom edge,
        # the same level as the community legend (loc="lower left")
        t_ax = ax.inset_axes([0.63, 0.025, 0.36, 0.16])
        t_ax.axis("off")
        t_ax.set_title("Top-10 Bridge MPs by betweenness centrality",
                       fontsize=9, pad=3, color=PALETTE["dark"],
                       fontweight="bold", loc="left")
        tbl = t_ax.table(
            cellText=table_rows,
            colLabels=["#", "MP", "Club", "Btw."],
            cellLoc="left", loc="upper center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1, 1.22)
        col_widths = [0.06, 0.42, 0.32, 0.20]
        for (r, c), cell in tbl.get_celld().items():
            cell.set_width(col_widths[c])
            cell.PAD = 0.02
        for j in range(4):
            tbl[0, j].set_facecolor(PALETTE.get("primary", "#AA151B"))
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(1, len(table_rows) + 1):
            for j in range(4):
                tbl[i, j].set_facecolor("#f5f5f5" if i % 2 == 0 else "white")

    out = FIG_DIR / f"fig1_community_betweenness_{int(thresh*100)}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"    saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Regenerating modified figures …")
    fig5_cohesion()
    fig5_cohesion_presidential_zoom()
    fig1_community_betweenness(thresh=0.99)
    print(f"\nDone. Figures saved to {FIG_DIR}")
    # Full regeneration (uncomment to rebuild all):
    # fig2_heatmap(); fig3_rebels(); fig4_temporal(); fig6_topics()
    # fig1_network(); fig1_community_betweenness(thresh=0.70); fig10_edges_vs_threshold()
