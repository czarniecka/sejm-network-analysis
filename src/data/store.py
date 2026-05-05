"""
Data loading utilities. All analysis modules import data through these functions.
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from src.config import EXCLUDED_VOTES, PARQUET_DIR, VOTE_ENCODING

logger = logging.getLogger(__name__)


def _term_dir(term: int) -> Path:
    d = PARQUET_DIR / f"term{term}"
    if not d.exists():
        raise FileNotFoundError(
            f"Parquet directory {d} not found. Run 01_fetch.py first."
        )
    return d


def load_mps(term: int) -> pl.DataFrame:
    """Load mps.parquet for the given term (only active MPs)."""
    return pl.read_parquet(_term_dir(term) / "mps.parquet").filter(pl.col("active"))


def load_votings(term: int) -> pl.DataFrame:
    """Load votings.parquet for the given term."""
    return pl.read_parquet(_term_dir(term) / "votings.parquet")


def load_votes(term: int) -> pl.DataFrame:
    """Load votes.parquet for the given term."""
    return pl.read_parquet(_term_dir(term) / "votes.parquet")


def build_vote_matrix(
    votes_df: pl.DataFrame,
    mps_df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """
    Build dense numpy arrays from the votes DataFrame.

    Returns:
        vote_matrix:    (V, N) float32 — 1=YES, -1=NO, 0=ABSTAIN, NaN=excluded
        presence_matrix:(V, N) bool    — True where MP was present (not excluded)
        mp_ids:         list[int]      — MP IDs in column order
        voting_keys:    list[str]      — voting keys in row order

    Only ELECTRONIC and TRADITIONAL votes (not VOTE_VALID or ABSENT) count as present.
    MPs with no valid votes are still included if they appear in mps_df.
    """
    # Unique MPs in stable order
    mp_ids: list[int] = mps_df["mp_id"].sort().to_list()
    mp_index = {mp_id: i for i, mp_id in enumerate(mp_ids)}
    N = len(mp_ids)

    # Unique votings in stable order
    voting_keys: list[str] = (
        votes_df.select("voting_key")
        .unique()
        .sort("voting_key")
        ["voting_key"]
        .to_list()
    )
    voting_index = {k: i for i, k in enumerate(voting_keys)}
    V = len(voting_keys)

    vote_matrix = np.full((V, N), np.nan, dtype=np.float32)
    presence_matrix = np.zeros((V, N), dtype=bool)

    # Filter out excluded vote types once, then fill matrix in vectorised chunks
    valid_votes = votes_df.filter(~pl.col("vote").cast(pl.Utf8).is_in(list(EXCLUDED_VOTES)))

    # Map voting_key and mp_id to integer indices using polars (no Python loops)
    vkey_series = pl.Series("voting_key", list(voting_index.keys()))
    vkey_idx_series = pl.Series("vi", list(voting_index.values()), dtype=pl.Int32)
    vkey_map = pl.DataFrame({"voting_key": vkey_series, "vi": vkey_idx_series})

    mp_id_series = pl.Series("mp_id", list(mp_index.keys()), dtype=pl.Int32)
    mp_idx_series = pl.Series("mi", list(mp_index.values()), dtype=pl.Int32)
    mp_map = pl.DataFrame({"mp_id": mp_id_series, "mi": mp_idx_series})

    vote_enc_map = pl.DataFrame({
        "vote": pl.Series(list(VOTE_ENCODING.keys())),
        "vote_val": pl.Series(list(VOTE_ENCODING.values()), dtype=pl.Float32),
    })

    mapped = (
        valid_votes
        .with_columns([
            pl.col("vote").cast(pl.Utf8),
            pl.col("voting_key").cast(pl.Utf8),
            pl.col("mp_id").cast(pl.Int32),
        ])
        .join(vkey_map, on="voting_key", how="inner")
        .join(mp_map, on="mp_id", how="inner")
        .join(vote_enc_map, on="vote", how="inner")
        .select(["vi", "mi", "vote_val"])
    )

    logger.info("Filling vote matrix (%d valid vote records) …", len(mapped))

    # Fill in chunks with a progress bar
    chunk_size = 200_000
    n_chunks = (len(mapped) + chunk_size - 1) // chunk_size
    vi_arr = mapped["vi"].to_numpy()
    mi_arr = mapped["mi"].to_numpy()
    val_arr = mapped["vote_val"].to_numpy()

    with Progress(SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn()) as progress:
        task = progress.add_task("Building vote matrix", total=n_chunks)
        for i in range(n_chunks):
            sl = slice(i * chunk_size, (i + 1) * chunk_size)
            vote_matrix[vi_arr[sl], mi_arr[sl]] = val_arr[sl]
            presence_matrix[vi_arr[sl], mi_arr[sl]] = True
            progress.advance(task)

    logger.info("Built vote matrix (%d votings × %d MPs)", V, N)
    return vote_matrix, presence_matrix, mp_ids, voting_keys
