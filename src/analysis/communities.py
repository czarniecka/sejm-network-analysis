"""
Leiden community detection on MP agreement networks.
Compares detected communities to official club labels using NMI, ARI, etc.
"""

import logging

import igraph as ig
import leidenalg
import numpy as np
import polars as pl
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import LabelEncoder

from src.config import AGREEMENT_THRESHOLDS, LEIDEN_ITERATIONS, LEIDEN_SEED, MIN_COPRESENCE

logger = logging.getLogger(__name__)


def build_igraph(
    agreement_frac: np.ndarray,
    threshold: float,
    mp_ids: list[int],
    copresence: np.ndarray | None = None,
    min_copresence: int = MIN_COPRESENCE,
) -> ig.Graph:
    """Build a weighted igraph Graph from the agreement matrix."""
    N = agreement_frac.shape[0]
    mask = agreement_frac >= threshold  # (N, N) bool
    np.fill_diagonal(mask, False)

    # Upper triangle only (undirected)
    rows, cols = np.where(np.triu(mask, k=1))
    if copresence is not None:
        valid = copresence[rows, cols] >= min_copresence
        rows, cols = rows[valid], cols[valid]

    weights = agreement_frac[rows, cols].tolist()
    edges = list(zip(rows.tolist(), cols.tolist()))

    g = ig.Graph(n=N, edges=edges, directed=False)
    g.es["weight"] = weights
    g.vs["mp_id"] = mp_ids
    return g


def run_leiden(g: ig.Graph) -> leidenalg.VertexPartition:
    """Run Leiden community detection with fixed seed."""
    return leidenalg.find_partition(
        g,
        leidenalg.ModularityVertexPartition,
        weights="weight",
        seed=LEIDEN_SEED,
        n_iterations=LEIDEN_ITERATIONS,
    )


def compare_to_clubs(
    membership: list[int],
    club_labels: list[str],
) -> dict:
    """
    Compare a Leiden partition to official club labels.

    Args:
        membership:  Leiden community index per MP (list of ints)
        club_labels: official club per MP (same order)

    Returns:
        dict with nmi, ari, homogeneity, completeness
    """
    enc = LabelEncoder()
    club_encoded = enc.fit_transform(club_labels)

    return {
        "nmi": float(normalized_mutual_info_score(club_encoded, membership)),
        "ari": float(adjusted_rand_score(club_encoded, membership)),
        "homogeneity": float(homogeneity_score(club_encoded, membership)),
        "completeness": float(completeness_score(club_encoded, membership)),
    }


def detect_communities(
    agreement_frac: np.ndarray,
    copresence: np.ndarray,
    mp_ids: list[int],
    club_labels: list[str],
    term: int,
    thresholds: list[float] = AGREEMENT_THRESHOLDS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Run Leiden at each threshold. Returns (communities_df, metrics_df).
    """
    all_community_rows: list[dict] = []
    all_metric_rows: list[dict] = []

    for t in thresholds:
        logger.info("Running Leiden at threshold %.2f …", t)
        g = build_igraph(agreement_frac, t, mp_ids, copresence)

        if g.ecount() == 0:
            logger.warning("No edges at threshold %.2f — skipping", t)
            all_metric_rows.append({
                "threshold": t, "n_communities": 0,
                "modularity": float("nan"), "nmi": float("nan"),
                "ari": float("nan"), "homogeneity": float("nan"),
                "completeness": float("nan"), "term": term,
            })
            continue

        partition = run_leiden(g)
        membership = partition.membership
        n_communities = len(set(membership))
        modularity = partition.modularity

        scores = compare_to_clubs(membership, club_labels)

        for i, (mp_id, community_id) in enumerate(zip(mp_ids, membership)):
            all_community_rows.append({
                "mp_id": mp_id,
                "community_id": community_id,
                "club": club_labels[i],
                "threshold": t,
                "term": term,
            })

        all_metric_rows.append({
            "threshold": t,
            "n_communities": n_communities,
            "modularity": modularity,
            **scores,
            "term": term,
        })
        logger.info(
            "  Threshold %.2f: %d communities, modularity=%.4f, NMI=%.4f, ARI=%.4f",
            t, n_communities, modularity, scores["nmi"], scores["ari"],
        )

    communities_df = pl.DataFrame(all_community_rows).with_columns([
        pl.col("mp_id").cast(pl.Int32),
        pl.col("community_id").cast(pl.Int32),
        pl.col("club").cast(pl.Categorical),
        pl.col("threshold").cast(pl.Float32),
        pl.col("term").cast(pl.Int8),
    ])

    metrics_df = pl.DataFrame(all_metric_rows).with_columns([
        pl.col("threshold").cast(pl.Float32),
        pl.col("term").cast(pl.Int8),
    ])

    return communities_df, metrics_df
