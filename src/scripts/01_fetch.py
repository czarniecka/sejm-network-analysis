"""
Script 01 — Fetch all voting data from the Sejm API and assemble parquet files.

Usage:
    uv run python src/scripts/01_fetch.py --term 10
    uv run python src/scripts/01_fetch.py --term 10 --no-resume
    uv run python src/scripts/01_fetch.py --term 9 --term 10
"""

import asyncio
import logging
import sys
from pathlib import Path

import click

# Ensure src/ is importable when running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fetch.pipeline import assemble_parquet, run_fetch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


@click.command()
@click.option("--term", "-t", multiple=True, type=int, default=[10], show_default=True,
              help="Parliamentary term(s) to fetch. Can repeat: --term 9 --term 10")
@click.option("--resume/--no-resume", default=True, show_default=True,
              help="Skip already-fetched JSON files.")
@click.option("--assemble/--no-assemble", "do_assemble", default=True, show_default=True,
              help="After fetching, assemble raw JSON into parquet files.")
def main(term: tuple[int, ...], resume: bool, do_assemble: bool) -> None:
    """Fetch Sejm voting data and write parquet files."""
    for t in term:
        click.echo(f"\n=== Fetching term {t} ===")
        asyncio.run(run_fetch(term=t, resume=resume))
        if do_assemble:
            click.echo(f"\n=== Assembling parquet for term {t} ===")
            assemble_parquet(term=t)

    click.echo("\nDone.")


if __name__ == "__main__":
    main()
