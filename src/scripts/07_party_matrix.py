"""
Script 07 — Inter-party majority agreement matrix.

Reads:  data/parquet/term{N}/votes.parquet, mps.parquet
        data/networks/mp_ids.npy
Writes: data/analysis/party_correlation_matrix.parquet

Usage:
    uv run python src/scripts/07_party_matrix.py --term 10
"""

import logging
import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.party_matrix import compute_party_agreement_matrix
from src.config import ANALYSIS_DIR
from src.data.store import build_vote_matrix, load_mps, load_votes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--term", "-t", type=int, default=10)
def main(term: int) -> None:
    """Compute inter-party majority agreement matrix."""
    votes_df = load_votes(term)
    mps_df = load_mps(term)

    vote_matrix, presence_matrix, mp_ids, voting_keys = build_vote_matrix(votes_df, mps_df)

    # Build mp_id -> club from mps.parquet (static, unlike votes.parquet)
    mp_club = dict(zip(mps_df["mp_id"].to_list(), mps_df["club"].to_list()))
    mp_club_str = {k: str(v) for k, v in mp_club.items()}

    matrix_df = compute_party_agreement_matrix(vote_matrix, presence_matrix, mp_ids, mp_club_str, term)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    matrix_df.write_parquet(ANALYSIS_DIR / "party_correlation_matrix.parquet")
    logger.info("Saved party_correlation_matrix.parquet")

    # Print wide-format table
    wide = matrix_df.pivot(values="agreement_rate", index="club1", on="club2")
    print("\nInter-party agreement matrix:")
    print(wide)


if __name__ == "__main__":
    main()
