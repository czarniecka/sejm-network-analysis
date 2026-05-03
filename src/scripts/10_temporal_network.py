"""
Script 10 — Temporal network evolution (monthly snapshots + cross-term).

Reads:  data/parquet/term{N}/votes.parquet, votings.parquet, mps.parquet
Writes: data/analysis/temporal_metrics.parquet

Usage:
    uv run python src/scripts/10_temporal_network.py --term 10
    uv run python src/scripts/10_temporal_network.py --term 9 --term 10
"""

import logging
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.temporal import compute_temporal_metrics
from src.config import ANALYSIS_DIR
from src.data.store import build_vote_matrix, load_mps, load_votes, load_votings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--term", "-t", multiple=True, type=int, default=[10])
def main(term: tuple[int, ...]) -> None:
    """Compute temporal network metrics (monthly snapshots per term)."""
    import polars as pl

    all_dfs = []

    for t in term:
        logger.info("Processing term %d …", t)
        votes_df = load_votes(t)
        votings_df = load_votings(t)
        mps_df = load_mps(t)

        vote_matrix, presence_matrix, mp_ids, voting_keys = build_vote_matrix(votes_df, mps_df)
        mp_id_to_club = dict(zip(mps_df["mp_id"].to_list(), mps_df["club"].to_list()))
        club_labels = [str(mp_id_to_club.get(mp_id, "UNKNOWN")) for mp_id in mp_ids]

        df = compute_temporal_metrics(
            vote_matrix, presence_matrix, voting_keys, mp_ids,
            club_labels, votings_df, t
        )
        if not df.is_empty():
            all_dfs.append(df)

    if all_dfs:
        combined = pl.concat(all_dfs)
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        combined.write_parquet(ANALYSIS_DIR / "temporal_metrics.parquet")
        logger.info("Saved temporal_metrics.parquet (%d rows)", len(combined))
        print("\nSample:")
        print(combined.head(20))
    else:
        logger.warning("No temporal metrics generated")


if __name__ == "__main__":
    main()
