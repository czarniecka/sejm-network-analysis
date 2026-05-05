"""
Script 24 — Graph embedding: spectral + t-SNE.

Embeds MPs in 2D purely from network topology (no party labels used).
If the clusters align with parties → voting behaviour alone recovers
political structure.

Methods:
  1. Laplacian spectral embedding — eigenvectors of the normalised graph Laplacian
     (Fiedler vector = 2nd eigenvector splits the graph optimally)
  2. t-SNE on the agreement matrix rows (each MP = vector of agreements with others)

Figures:
  fig31_spectral_embedding.png
  fig32_tsne_embedding.png
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import ANALYSIS_DIR, NETWORKS_DIR, PROJECT_ROOT
from src.data.store import load_mps
from src.scripts.poster_style import apply_style, CLUB_COLOURS, cc, MAIN_CLUBS, PALETTE, COALITION, OPPOSITION, club_en

apply_style()

FIG_DIR = PROJECT_ROOT / "data" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MIN_COP = 50


def load_data(term: int = 10):
    agreement  = np.load(NETWORKS_DIR / "agreement_matrix.npy")
    copresence = np.load(NETWORKS_DIR / "copresence_matrix.npy")
    mp_ids     = np.load(NETWORKS_DIR / "mp_ids.npy").tolist()
    mps        = load_mps(term)
    id_to_club = dict(zip(mps["mp_id"].to_list(), mps["club"].cast(pl.Utf8).to_list()))
    id_to_name = {r["mp_id"]: f"{r['first_name'][0]}. {r['last_name']}"
                  for r in mps.iter_rows(named=True)}
    clubs = [id_to_club.get(mid, "niez.") for mid in mp_ids]
    names = [id_to_name.get(mid, str(mid)) for mid in mp_ids]
    return agreement, copresence, mp_ids, clubs, names


def spectral_embedding(agreement: np.ndarray, copresence: np.ndarray,
                        n_components: int = 4) -> np.ndarray:
    """
    Compute normalised Laplacian spectral embedding.
    Returns (N, n_components) array of eigenvectors 1..n_components+1
    (skipping the trivial eigenvector 0).
    """
    N = agreement.shape[0]
    # Weight matrix: use agreement where copresence sufficient, else 0
    W = agreement.copy()
    W[copresence < MIN_COP] = 0.0
    np.fill_diagonal(W, 0.0)
    W = np.clip(W, 0, None)

    # Degree matrix
    d = W.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)

    # Normalised Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt

    # Sparse eigendecomposition: smallest n_components+1 eigenvalues
    L_sp = sparse.csr_matrix(L_sym.astype(np.float64))
    eigenvalues, eigenvectors = eigsh(L_sp, k=n_components + 1, which="SM")

    # Sort by eigenvalue
    order = np.argsort(eigenvalues)
    eigenvalues  = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    print(f"  Spectral eigenvalues (first {n_components+1}): {eigenvalues.round(4)}")
    # Skip trivial eigenvector (eigenvalue ≈ 0)
    return eigenvectors[:, 1:n_components+1], eigenvalues


def tsne_embedding(agreement: np.ndarray, copresence: np.ndarray) -> np.ndarray:
    """t-SNE on the agreement matrix rows (each MP = their agreement profile)."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  sklearn not available — skipping t-SNE")
        return None

    # Fill missing values with row mean
    W = agreement.copy()
    W[copresence < MIN_COP] = np.nan
    np.fill_diagonal(W, np.nan)
    row_means = np.nanmean(W, axis=1)
    for i in range(W.shape[0]):
        W[i, np.isnan(W[i])] = row_means[i]
    np.fill_diagonal(W, 1.0)

    print("  computing t-SNE (perplexity=30) …")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                max_iter=1000, learning_rate="auto", init="pca")
    return tsne.fit_transform(W)


def _scatter_plot(ax, x, y, clubs, names, title: str,
                  xlabel: str = "Dim 1", ylabel: str = "Dim 2",
                  annotate_n: int = 8) -> None:
    """Shared scatter logic for spectral and t-SNE plots."""
    cols = [cc(c) for c in clubs]
    ax.scatter(x, y, c=cols, s=22, alpha=0.80, linewidths=0, zorder=2)

    # Annotate outliers (extremes in each quadrant)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, fontsize=10); ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(alpha=0.20); ax.spines[["top", "right"]].set_visible(False)

    # Draw convex hulls per main club (optional, only if scipy available)
    try:
        from scipy.spatial import ConvexHull
        for club in MAIN_CLUBS:
            idx = [i for i, c in enumerate(clubs) if c == club]
            if len(idx) < 4:
                continue
            pts = np.array([[x[i], y[i]] for i in idx])
            try:
                hull = ConvexHull(pts)
                for simplex in hull.simplices:
                    ax.plot(pts[simplex, 0], pts[simplex, 1],
                            color=cc(club), alpha=0.25,
                            linewidth=0.8, zorder=1)
            except Exception:
                pass
    except ImportError:
        pass

    # Annotate top outliers
    dists = np.sqrt(x**2 + y**2)
    top_idx = np.argsort(dists)[-annotate_n:]
    for i in top_idx:
        ax.annotate(names[i], (x[i], y[i]), fontsize=6.5, color=PALETTE["dark"],
                    xytext=(3, 3), textcoords="offset points", zorder=4)


def fig31_spectral_embedding(evecs: np.ndarray, evals: np.ndarray,
                               clubs: list, names: list) -> None:
    print("  fig31: spectral embedding …")
    # 4 panels: (ev1, ev2), (ev1, ev3), (ev2, ev3), (ev3, ev4)
    pairs = [(0, 1), (0, 2), (1, 2), (2, 3)] if evecs.shape[1] >= 4 else [(0, 1)]

    fig = plt.figure(figsize=(15, 13), facecolor="white")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(len(pairs))]

    for ax, (d1, d2) in zip(axes, pairs):
        ax.set_facecolor("white")
        x = evecs[:, d1]; y = evecs[:, d2]
        _scatter_plot(ax, x, y, clubs, names,
                      title=f"Eigenvectors {d1+2} vs {d2+2}\n"
                            f"(λ = {evals[d1+1]:.4f} vs λ = {evals[d2+1]:.4f})",
                      xlabel=f"Eigenvector {d1+2}  (λ={evals[d1+1]:.4f})",
                      ylabel=f"Eigenvector {d2+2}  (λ={evals[d2+1]:.4f})")

    # Shared legend
    seen = {}
    for c in clubs:
        if c not in seen: seen[c] = cc(c)
    handles = [mpatches.Patch(color=col, label=club) for club, col in sorted(seen.items())]
    fig.legend(handles=handles, fontsize=8.5, loc="lower center",
               ncol=6, framealpha=0.6, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Laplacian Spectral Embedding of the voting network  |  Term X\n"
        "Node positions determined solely from network topology (no party labels)",
        fontsize=14, color=PALETTE["dark"], y=1.01,
    )
    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig31_spectral_embedding.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out}")


def fig32_tsne_embedding(tsne_coords: np.ndarray, clubs: list, names: list) -> None:
    if tsne_coords is None:
        print("  fig32: skipped (sklearn not available)")
        return
    print("  fig32: t-SNE embedding …")

    fig = plt.figure(figsize=(14, 7.5), facecolor="white")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

    x = tsne_coords[:, 0]; y = tsne_coords[:, 1]

    # Left: coloured by club
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("white")
    _scatter_plot(ax1, x, y, clubs, names,
                  title="t-SNE — coloured by club", annotate_n=10)
    seen = {}
    for c in clubs:
        if c not in seen: seen[c] = cc(c)
    handles = [mpatches.Patch(color=col, label=club) for club, col in sorted(seen.items())]
    ax1.legend(handles=handles, fontsize=7.5, framealpha=0.6, ncol=2,
               loc="lower left", title="Club")

    # Right: coloured by bloc (coalition / opposition / other)
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("white")
    bloc_cols = []
    for c in clubs:
        if c in COALITION:
            bloc_cols.append(PALETTE["accent"])
        elif c in OPPOSITION:
            bloc_cols.append("#2C3E50")
        else:
            bloc_cols.append(PALETTE["neutral"])
    ax2.scatter(x, y, c=bloc_cols, s=22, alpha=0.82, linewidths=0, zorder=2)
    ax2.set_title("t-SNE — coloured by bloc\n(red=coalition, dark=opposition)", fontsize=12)
    ax2.set_xlabel("t-SNE dim 1", fontsize=10); ax2.set_ylabel("t-SNE dim 2", fontsize=10)
    ax2.grid(alpha=0.20); ax2.spines[["top", "right"]].set_visible(False)
    handles2 = [
        mpatches.Patch(color=PALETTE["accent"],  label="Coalition"),
        mpatches.Patch(color="#2C3E50",           label="Opposition"),
        mpatches.Patch(color=PALETTE["neutral"],  label="Other"),
    ]
    ax2.legend(handles=handles2, fontsize=9, framealpha=0.6, loc="lower left")

    # Compute silhouette score
    try:
        from sklearn.metrics import silhouette_score
        bloc_labels = [1 if c in COALITION else (2 if c in OPPOSITION else 0) for c in clubs]
        mask = [l > 0 for l in bloc_labels]
        if sum(mask) > 10:
            sil = silhouette_score(tsne_coords[mask], np.array(bloc_labels)[mask])
            ax2.text(0.97, 0.03, f"Silhouette score\n(coalition vs opposition): {sil:.3f}",
                     transform=ax2.transAxes, ha="right", va="bottom", fontsize=9,
                     bbox=dict(boxstyle="round", facecolor="white", edgecolor=PALETTE["light_grey"], alpha=0.85))
    except Exception:
        pass

    fig.suptitle(
        "t-SNE embedding of the voting network  |  Term X\n"
        "Each MP = agreement vector with others (topology, no labels)",
        fontsize=14, color=PALETTE["dark"], y=1.01,
    )
    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig32_tsne_embedding.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out}")


if __name__ == "__main__":
    print("Script 24 — Graph embedding …")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    agreement, copresence, mp_ids, clubs, names = load_data()

    evecs, evals = spectral_embedding(agreement, copresence, n_components=4)
    emb_df = pl.DataFrame({
        "mp_id":  mp_ids,
        "club":   clubs,
        "ev1":    evecs[:, 0].tolist(),
        "ev2":    evecs[:, 1].tolist(),
        "ev3":    evecs[:, 2].tolist() if evecs.shape[1] > 2 else [0.0] * len(mp_ids),
    })
    emb_df.write_parquet(ANALYSIS_DIR / "spectral_embedding.parquet")
    fig31_spectral_embedding(evecs, evals, clubs, names)

    tsne_coords = tsne_embedding(agreement, copresence)
    if tsne_coords is not None:
        tsne_df = pl.DataFrame({
            "mp_id": mp_ids, "club": clubs,
            "tsne1": tsne_coords[:, 0].tolist(),
            "tsne2": tsne_coords[:, 1].tolist(),
        })
        tsne_df.write_parquet(ANALYSIS_DIR / "tsne_embedding.parquet")
    fig32_tsne_embedding(tsne_coords, clubs, names)

    print(f"\nDone. Figures in {FIG_DIR}")
