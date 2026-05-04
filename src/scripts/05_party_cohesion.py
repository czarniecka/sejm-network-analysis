"""
Script 05 — Intra-party cohesion over time.

Reads:  data/parquet/term{N}/votes.parquet, votings.parquet, mps.parquet
        data/networks/*.npy
Writes: data/analysis/party_cohesion_by_sitting.parquet
        data/analysis/party_cohesion_by_month.parquet

Usage:
    uv run python src/scripts/05_party_cohesion.py --term 10
"""

import logging
import sys
from pathlib import Path

import click
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.cohesion import aggregate_monthly, compute_party_cohesion
from src.config import ANALYSIS_DIR, NETWORKS_DIR
from src.data.store import build_vote_matrix, load_mps, load_votes, load_votings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--term", "-t", type=int, default=10)
def main(term: int) -> None:
    """Compute intra-party cohesion per sitting and per month."""
    votes_df = load_votes(term)
    votings_df = load_votings(term)
    mps_df = load_mps(term)

    vote_matrix, presence_matrix, mp_ids, voting_keys = build_vote_matrix(votes_df, mps_df)

    logger.info("Computing party cohesion per sitting …")
    cohesion_df = compute_party_cohesion(
        vote_matrix, presence_matrix, mp_ids, voting_keys, votes_df, votings_df, term
    )

    monthly_df = aggregate_monthly(cohesion_df)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    cohesion_df.write_parquet(ANALYSIS_DIR / "party_cohesion_by_sitting.parquet")
    monthly_df.write_parquet(ANALYSIS_DIR / "party_cohesion_by_month.parquet")

    logger.info("Saved party_cohesion_by_sitting.parquet (%d rows)", len(cohesion_df))
    logger.info("Saved party_cohesion_by_month.parquet (%d rows)", len(monthly_df))

    print("\nTop cohesion by club (mean across sittings):")
    print(
        cohesion_df
        .group_by("club")
        .agg(pl.col("cohesion_score").mean())
        .sort("cohesion_score", descending=True)
    )


if __name__ == "__main__":
    main()
