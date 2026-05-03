"""
Script 03 — Compute network metrics at multiple agreement thresholds.

Reads:  data/networks/agreement_matrix.npy, copresence_matrix.npy, mp_ids.npy
Writes: data/networks/network_metrics.parquet

Usage:
    uv run python src/scripts/03_network_analysis.py --term 10
"""

import logging
import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.network import compute_all_thresholds
from src.config import AGREEMENT_THRESHOLDS, NETWORKS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--term", "-t", type=int, default=10)
def main(term: int) -> None:
    """Compute network metrics at all agreement thresholds."""
    agreement_frac = np.load(NETWORKS_DIR / "agreement_matrix.npy")
    copresence = np.load(NETWORKS_DIR / "copresence_matrix.npy")
    mp_ids = np.load(NETWORKS_DIR / "mp_ids.npy").tolist()

    logger.info("Computing network metrics for %d thresholds …", len(AGREEMENT_THRESHOLDS))
    metrics_df = compute_all_thresholds(agreement_frac, copresence, mp_ids, term)

    out_path = NETWORKS_DIR / "network_metrics.parquet"
    metrics_df.write_parquet(out_path)
    logger.info("Saved network_metrics.parquet")

    print("\n" + str(metrics_df))


if __name__ == "__main__":
    main()
