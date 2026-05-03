"""
Script 11 — Centrality analysis of rebel MPs.

Tests whether rebel MPs (who vote against their party) are structural bridges.

Reads:  data/networks/*.npy, data/parquet/term{N}/mps.parquet
        data/analysis/rebels.parquet
Writes: data/analysis/centrality_rebels.parquet
        data/analysis/centrality_correlations.parquet

Usage:
    uv run python src/scripts/11_centrality_rebels.py --term 10
"""

import logging
import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.centrality import (
    compute_centrality_measures,
    compute_spearman_correlations,
    join_with_rebels,
)
from src.config import ANALYSIS_DIR, NETWORKS_DIR
from src.data.store import load_mps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--term", "-t", type=int, default=10)
@click.option("--threshold", type=float, default=0.50, show_default=True)
def main(term: int, threshold: float) -> None:
    """Compute centrality measures and correlate with rebel scores."""
    import polars as pl

    agreement_frac = np.load(NETWORKS_DIR / "agreement_matrix.npy")
    copresence = np.load(NETWORKS_DIR / "copresence_matrix.npy")
    mp_ids = np.load(NETWORKS_DIR / "mp_ids.npy").tolist()
    mps_df = load_mps(term)

    rebels_path = ANALYSIS_DIR / "rebels.parquet"
    if not rebels_path.exists():
        logger.error("rebels.parquet not found — run 06_rebels.py first")
        return
    rebels_df = pl.read_parquet(rebels_path)

    logger.info("Computing centrality measures at threshold %.2f …", threshold)
    centrality_df = compute_centrality_measures(agreement_frac, copresence, mp_ids, threshold)

    joined_df = join_with_rebels(centrality_df, rebels_df, mps_df, term)
    correlations_df = compute_spearman_correlations(joined_df, term)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    joined_df.write_parquet(ANALYSIS_DIR / "centrality_rebels.parquet")
    correlations_df.write_parquet(ANALYSIS_DIR / "centrality_correlations.parquet")

    logger.info("Saved centrality_rebels.parquet (%d rows)", len(joined_df))

    print("\nSpearman correlations (rebel_rate vs centrality):")
    print(correlations_df)

    print("\nTop-10 by betweenness centrality:")
    print(
        joined_df
        .sort("betweenness", descending=True)
        .select(["first_name", "last_name", "club", "betweenness", "rebel_rate"])
        .head(10)
    )


if __name__ == "__main__":
    main()
