"""
Script 06 — Rebel MP detection.

Reads:  data/parquet/term{N}/votes.parquet, mps.parquet
        data/networks/mp_ids.npy, agreement_matrix.npy, copresence_matrix.npy
Writes: data/analysis/rebels.parquet

Usage:
    uv run python src/scripts/06_rebels.py --term 10
"""

import logging
import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.rebels import compute_rebel_scores
from src.config import ANALYSIS_DIR, NETWORKS_DIR
from src.data.store import build_vote_matrix, load_mps, load_votes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--term", "-t", type=int, default=10)
@click.option("--min-votes", type=int, default=100, show_default=True)
def main(term: int, min_votes: int) -> None:
    """Compute rebel scores for all MPs."""
    votes_df = load_votes(term)
    mps_df = load_mps(term)

    vote_matrix, presence_matrix, mp_ids, voting_keys = build_vote_matrix(votes_df, mps_df)

    rebels_df = compute_rebel_scores(
        vote_matrix, presence_matrix, mp_ids, voting_keys, votes_df, term, min_votes
    )

    # Join MP names
    rebels_df = rebels_df.join(
        mps_df.select(["mp_id", "first_name", "last_name"]),
        on="mp_id",
        how="left",
    )

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    rebels_df.write_parquet(ANALYSIS_DIR / "rebels.parquet")
    logger.info("Saved rebels.parquet (%d MPs)", len(rebels_df))

    print("\nTop-20 rebels:")
    print(
        rebels_df
        .select(["first_name", "last_name", "club", "rebel_rate", "rebel_count", "total_votes"])
        .head(20)
    )


if __name__ == "__main__":
    main()
