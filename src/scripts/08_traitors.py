"""
Script 08 — Party switcher (traitor) detection.

Reads:  data/parquet/term{N}/votes.parquet, mps.parquet
        data/raw/term{N}/clubs.json
Writes: data/analysis/party_switchers.parquet
        data/analysis/switch_summary.parquet

Usage:
    uv run python src/scripts/08_traitors.py --term 10
"""

import logging
import sys
from pathlib import Path

import click
import orjson

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.traitors import build_switch_summary, detect_club_switches
from src.config import ANALYSIS_DIR, RAW_DIR
from src.data.store import load_mps, load_votes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--term", "-t", type=int, default=10)
def main(term: int) -> None:
    """Detect MPs who switched political clubs during the term."""
    votes_df = load_votes(term)
    mps_df = load_mps(term)

    clubs_path = RAW_DIR / f"term{term}" / "clubs.json"
    clubs_raw: list[dict] = []
    if clubs_path.exists():
        clubs_raw = orjson.loads(clubs_path.read_bytes())

    switchers_df = detect_club_switches(votes_df, mps_df, clubs_raw, term)
    summary_df = build_switch_summary(switchers_df, term)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    switchers_df.write_parquet(ANALYSIS_DIR / "party_switchers.parquet")
    if not summary_df.is_empty():
        summary_df.write_parquet(ANALYSIS_DIR / "switch_summary.parquet")

    print(f"\nDetected {len(switchers_df)} club switches:")
    if not switchers_df.is_empty():
        print(switchers_df.select(["first_name", "last_name", "from_club", "to_club", "switch_date", "switch_type"]))


if __name__ == "__main__":
    main()
