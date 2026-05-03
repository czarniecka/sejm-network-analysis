"""
Temporal network analysis: how the MP voting network evolves over time.
Supports cross-term comparison (term 9 vs 10) and within-term monthly snapshots.
"""

import logging
from datetime import date

import numpy as np
import polars as pl

from src.config import AGREEMENT_THRESHOLDS, LEIDEN_SEED, MIN_COPRESENCE_MONTHLY

logger = logging.getLogger(__name__)


def build_monthly_vote_matrices(
    vote_matrix: np.ndarray,
    presence_matrix: np.ndarray,
    voting_keys: list[str],
    votings_df: pl.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Slice vote_matrix by year-month.

    Returns a dict: {"2024-03": (vote_matrix_slice, presence_matrix_slice), ...}
    """
    key_to_ym: dict[str, str] = {}
    for row in votings_df.iter_rows(named=True):
        dt = row["date"]
        if dt is None:
            continue
        if hasattr(dt, "strftime"):
            ym = dt.strftime("%Y-%m")
        else:
            ym = str(dt)[:7]
        key_to_ym[row["voting_key"]] = ym

    # Group row indices by year-month
    ym_indices: dict[str, list[int]] = {}
    for i, vkey in enumerate(voting_keys):
        ym = key_to_ym.get(vkey)
        if ym:
            ym_indices.setdefault(ym, []).append(i)

    slices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ym, idxs in sorted(ym_indices.items()):
        idx_arr = np.array(idxs)
        slices[ym] = (vote_matrix[idx_arr], presence_matrix[idx_arr])

    return slices


def compute_temporal_metrics(
    vote_matrix: np.ndarray,
    presence_matrix: np.ndarray,
    voting_keys: list[str],
    mp_ids: list[int],
    club_labels: list[str],
    votings_df: pl.DataFrame,
    term: int,
    thresholds: list[float] = AGREEMENT_THRESHOLDS,
) -> pl.DataFrame:
    """
    Compute network metrics for each (year_month, threshold) combination.

    Returns temporal_metrics.parquet rows.
    """
    from src.analysis.agreement import compute_agreement_matrix
    from src.analysis.communities import build_igraph, run_leiden, compare_to_clubs
    from src.analysis.network import compute_network_metrics

    monthly_slices = build_monthly_vote_matrices(
        vote_matrix, presence_matrix, voting_keys, votings_df
    )

    all_rows: list[dict] = []

    for ym, (vm_slice, pm_slice) in sorted(monthly_slices.items()):
        n_votings = vm_slice.shape[0]
        if n_votings < 5:
            logger.debug("Skipping %s: only %d votings", ym, n_votings)
            continue

        logger.info("Computing temporal metrics for %s (%d votings) …", ym, n_votings)
        agreement_frac, copresence = compute_agreement_matrix(
            vm_slice, pm_slice, min_copresence=MIN_COPRESENCE_MONTHLY
        )

        for t in thresholds:
            metrics = compute_network_metrics(agreement_frac, copresence, t, mp_ids, term)

            # Leiden modularity + NMI for this snapshot
            modularity = float("nan")
            nmi = float("nan")
            try:
                g = build_igraph(agreement_frac, t, mp_ids, copresence, MIN_COPRESENCE_MONTHLY)
                if g.ecount() > 0:
                    partition = run_leiden(g)
                    modularity = partition.modularity
                    scores = compare_to_clubs(partition.membership, club_labels)
                    nmi = scores["nmi"]
            except Exception as exc:
                logger.debug("Leiden failed for %s t=%.2f: %s", ym, t, exc)

            all_rows.append({
                "term": term,
                "year_month": ym,
                "n_votings": n_votings,
                "threshold": t,
                "n_nodes": metrics["n_nodes"],
                "n_edges": metrics["n_edges"],
                "density": metrics["density"],
                "n_components": metrics["n_components"],
                "largest_component_frac": metrics["largest_component_frac"],
                "avg_clustering": metrics["avg_clustering"],
                "modularity": modularity,
                "nmi": nmi,
            })

    if not all_rows:
        return pl.DataFrame()

    return pl.DataFrame(all_rows).with_columns([
        pl.col("term").cast(pl.Int8),
        pl.col("n_votings").cast(pl.Int32),
        pl.col("threshold").cast(pl.Float32),
        pl.col("n_nodes").cast(pl.Int32),
        pl.col("n_edges").cast(pl.Int32),
        pl.col("density").cast(pl.Float64),
        pl.col("n_components").cast(pl.Int32),
        pl.col("largest_component_frac").cast(pl.Float32),
        pl.col("avg_clustering").cast(pl.Float64),
        pl.col("modularity").cast(pl.Float64),
        pl.col("nmi").cast(pl.Float64),
    ]).sort(["year_month", "threshold"])
