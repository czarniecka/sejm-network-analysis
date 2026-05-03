"""
Script 02 — Build the pairwise MP agreement matrix and save to data/networks/.

Reads: data/parquet/term{N}/votes.parquet, mps.parquet
Writes:
  data/networks/agreement_matrix.npy   (N, N) float32
  data/networks/copresence_matrix.npy  (N, N) int32
  data/networks/mp_ids.npy             (N,)   int32

Usage:
    uv run python src/scripts/02_build_agreement.py --term 10
"""

import logging
import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.agreement import compute_agreement_matrix
from src.config import NETWORKS_DIR
from src.data.store import build_vote_matrix, load_mps, load_votes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--term", "-t", type=int, default=10)
def main(term: int) -> None:
    """Build and save the MP agreement matrix."""
    logger.info("Loading data for term %d …", term)
    votes_df = load_votes(term)
    mps_df = load_mps(term)

    logger.info("Building vote matrix …")
    vote_matrix, presence_matrix, mp_ids, voting_keys = build_vote_matrix(votes_df, mps_df)

    logger.info("Computing agreement matrix …")
    agreement_frac, copresence = compute_agreement_matrix(vote_matrix, presence_matrix)

    NETWORKS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(NETWORKS_DIR / "agreement_matrix.npy", agreement_frac)
    np.save(NETWORKS_DIR / "copresence_matrix.npy", copresence)
    np.save(NETWORKS_DIR / "mp_ids.npy", np.array(mp_ids, dtype=np.int32))

    logger.info("Saved agreement matrix (%d × %d)", *agreement_frac.shape)
    valid_pairs = int(np.sum(~np.isnan(agreement_frac))) // 2
    logger.info("Valid MP pairs: %d", valid_pairs)
    logger.info("Mean agreement rate (valid pairs): %.4f", float(np.nanmean(agreement_frac)))


if __name__ == "__main__":
    main()
