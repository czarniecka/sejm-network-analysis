"""
Script 12 — Cross-party voting bloc detection per BERTopic topic.

Reads:  data/networks/agreement_matrix.npy, copresence_matrix.npy, mp_ids.npy
        data/parquet/term{N}/mps.parquet, votes.parquet
        data/analysis/voting_topics.parquet, topic_summary.parquet
Writes: data/analysis/voting_blocs.parquet
        data/analysis/bloc_affinity.parquet
        data/analysis/bloc_summary.parquet

Usage:
    uv run python src/scripts/12_voting_blocs.py --term 10
"""

import logging
import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.blocs import compute_topic_blocs
from src.config import ANALYSIS_DIR, NETWORKS_DIR
from src.data.store import build_vote_matrix, load_mps, load_votes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--term", "-t", type=int, default=10)
def main(term: int) -> None:
    """Detect cross-party voting blocs per topic."""
    import polars as pl

    # Load base data
    votes_df = load_votes(term)
    mps_df = load_mps(term)
    vote_matrix, presence_matrix, mp_ids, voting_keys = build_vote_matrix(votes_df, mps_df)

    # Load topic data
    topics_path = ANALYSIS_DIR / "voting_topics.parquet"
    summary_path = ANALYSIS_DIR / "topic_summary.parquet"
    if not topics_path.exists() or not summary_path.exists():
        logger.error("Run 09_topic_modeling.py first")
        return

    voting_topics_df = pl.read_parquet(topics_path)
    topic_summary_df = pl.read_parquet(summary_path)

    # Align voting_topics rows with vote_matrix row order (voting_keys)
    # voting_topics_df may have different row order than vote_matrix
    vkey_to_row = {k: i for i, k in enumerate(voting_keys)}
    aligned_topics = pl.Series(
        [int(voting_topics_df.filter(pl.col("voting_key") == k)["topic_id"][0])
         if k in voting_topics_df["voting_key"].to_list() else -1
         for k in voting_keys],
        dtype=pl.Int32,
    )

    # Also need to rebuild voting_topics_df in vote_matrix row order for bloc analysis
    # Build a lookup and pass the full df
    mp_id_to_club = dict(zip(mps_df["mp_id"].to_list(), mps_df["club"].to_list()))
    club_labels = [str(mp_id_to_club.get(mp_id, "UNKNOWN")) for mp_id in mp_ids]

    logger.info("Detecting voting blocs across topics …")
    blocs_df, affinity_df, summary_df = compute_topic_blocs(
        vote_matrix, presence_matrix, mp_ids, club_labels,
        voting_topics_df, topic_summary_df, term
    )

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if not blocs_df.is_empty():
        blocs_df.write_parquet(ANALYSIS_DIR / "voting_blocs.parquet")
    if not affinity_df.is_empty():
        affinity_df.write_parquet(ANALYSIS_DIR / "bloc_affinity.parquet")
    if not summary_df.is_empty():
        summary_df.write_parquet(ANALYSIS_DIR / "bloc_summary.parquet")

    logger.info("Saved voting blocs outputs")

    if not affinity_df.is_empty():
        print("\nTop club-pair affinities:")
        print(affinity_df.sort("bloc_affinity", descending=True).head(10))

    if not summary_df.is_empty():
        print("\nTopics with most cross-party communities:")
        print(summary_df.head(10))


if __name__ == "__main__":
    main()
