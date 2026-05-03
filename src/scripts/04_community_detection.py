"""
Script 04 — Leiden community detection at multiple thresholds.

Reads:  data/networks/*.npy, data/parquet/term{N}/mps.parquet
Writes: data/networks/communities_threshold_{T}.parquet
        data/networks/community_metrics.parquet

Usage:
    uv run python src/scripts/04_community_detection.py --term 10
"""

import logging
import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.communities import detect_communities
from src.config import NETWORKS_DIR
from src.data.store import load_mps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--term", "-t", type=int, default=10)
def main(term: int) -> None:
    """Run Leiden community detection and compare to official clubs."""
    agreement_frac = np.load(NETWORKS_DIR / "agreement_matrix.npy")
    copresence = np.load(NETWORKS_DIR / "copresence_matrix.npy")
    mp_ids = np.load(NETWORKS_DIR / "mp_ids.npy").tolist()

    mps_df = load_mps(term)
    mp_id_to_club = dict(zip(mps_df["mp_id"].to_list(), mps_df["club"].to_list()))
    club_labels = [str(mp_id_to_club.get(mp_id, "UNKNOWN")) for mp_id in mp_ids]

    communities_df, metrics_df = detect_communities(
        agreement_frac, copresence, mp_ids, club_labels, term
    )

    NETWORKS_DIR.mkdir(parents=True, exist_ok=True)

    # Save per-threshold community assignments
    for t in communities_df["threshold"].unique().sort().to_list():
        sub = communities_df.filter(communities_df["threshold"] == t)
        fname = f"communities_threshold_{t:.2f}.parquet".replace(".", "_")
        sub.write_parquet(NETWORKS_DIR / fname)

    metrics_df.write_parquet(NETWORKS_DIR / "community_metrics.parquet")

    print("\nCommunity metrics:")
    print(str(metrics_df))


if __name__ == "__main__":
    main()
