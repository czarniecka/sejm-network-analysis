"""
Intra-party cohesion over time (per sitting and per month).
"""

import logging
from datetime import date

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def compute_party_cohesion(
    vote_matrix: np.ndarray,
    presence_matrix: np.ndarray,
    mp_ids: list[int],
    voting_keys: list[str],
    votes_df: pl.DataFrame,
    votings_df: pl.DataFrame,
    term: int,
) -> pl.DataFrame:
    """
    Compute intra-party cohesion score per (club, sitting).

    cohesion(C, S) = fraction of (voting v in S, MP pair i<j in C both present at v)
                     where i and j cast the same vote.

    Returns a DataFrame with columns: club, sitting, date, cohesion_score,
    n_votings, n_mp_pairs, term.
    """
    # Build lookup: voting_key -> (sitting, date)
    key_to_sitting: dict[str, int] = {}
    key_to_date: dict[str, date] = {}
    for row in votings_df.iter_rows(named=True):
        k = row["voting_key"]
        key_to_sitting[k] = row["sitting"]
        dt = row["date"]
        key_to_date[k] = dt.date() if hasattr(dt, "date") else dt

    # Build lookup: voting_key -> row index in vote_matrix
    vkey_list = voting_keys  # same order as vote_matrix rows
    vkey_index = {k: i for i, k in enumerate(vkey_list)}

    # mp_id -> column index
    mp_index = {mp_id: i for i, mp_id in enumerate(mp_ids)}

    # Get clubs from votes_df: mp_id -> most recent club
    mp_club: dict[int, str] = {}
    for row in (
        votes_df.sort("date")
        .group_by("mp_id")
        .agg(pl.col("club").last())
        .iter_rows(named=True)
    ):
        mp_club[row["mp_id"]] = row["club"]

    # Group voting keys by sitting
    sitting_to_vkeys: dict[int, list[str]] = {}
    for vkey in vkey_list:
        s = key_to_sitting.get(vkey)
        if s is not None:
            sitting_to_vkeys.setdefault(s, []).append(vkey)

    # Get unique clubs
    clubs = sorted(set(mp_club.values()))

    rows: list[dict] = []

    for sitting, vkeys in sitting_to_vkeys.items():
        # Get row indices for this sitting's votings
        row_indices = [vkey_index[k] for k in vkeys if k in vkey_index]
        if not row_indices:
            continue
        v_mat = vote_matrix[row_indices]       # (V_s, N)
        v_pres = presence_matrix[row_indices]  # (V_s, N)

        # Get date for this sitting (from first voting)
        s_date = key_to_date.get(vkeys[0])

        for club in clubs:
            # Column indices for this club's MPs
            club_mp_ids = [mp_id for mp_id, c in mp_club.items() if c == club]
            col_indices = [mp_index[mp_id] for mp_id in club_mp_ids if mp_id in mp_index]
            if len(col_indices) < 2:
                continue

            club_votes = v_mat[:, col_indices]   # (V_s, N_c)
            club_pres = v_pres[:, col_indices]   # (V_s, N_c)

            total_agree = 0
            total_pairs = 0

            for vi in range(club_votes.shape[0]):
                pres_v = club_pres[vi]
                present_indices = np.where(pres_v)[0]
                n_present = len(present_indices)
                if n_present < 2:
                    continue
                vv = club_votes[vi, present_indices]
                # Count agreeing pairs: sum_k C(count_k, 2)
                for val in (1.0, -1.0, 0.0):
                    cnt = int(np.sum(vv == val))
                    total_agree += cnt * (cnt - 1)
                total_pairs += n_present * (n_present - 1)

            if total_pairs == 0:
                continue

            rows.append({
                "club": club,
                "sitting": sitting,
                "date": s_date,
                "cohesion_score": total_agree / total_pairs,
                "n_votings": len(row_indices),
                "n_mp_pairs": total_pairs // 2,
                "term": term,
            })

    if not rows:
        return pl.DataFrame(schema={
            "club": pl.Categorical, "sitting": pl.Int16, "date": pl.Date,
            "cohesion_score": pl.Float32, "n_votings": pl.Int16,
            "n_mp_pairs": pl.Int32, "term": pl.Int8,
        })

    return pl.DataFrame(rows).with_columns([
        pl.col("club").cast(pl.Categorical),
        pl.col("sitting").cast(pl.Int16),
        pl.col("date").cast(pl.Date),
        pl.col("cohesion_score").cast(pl.Float32),
        pl.col("n_votings").cast(pl.Int16),
        pl.col("n_mp_pairs").cast(pl.Int32),
        pl.col("term").cast(pl.Int8),
    ])


def aggregate_monthly(cohesion_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-sitting cohesion to per-month (weighted mean)."""
    return (
        cohesion_df
        .with_columns(
            pl.col("date").dt.strftime("%Y-%m").alias("year_month")
        )
        .group_by(["club", "year_month", "term"])
        .agg([
            (pl.col("cohesion_score") * pl.col("n_mp_pairs")).sum().alias("_weighted_sum"),
            pl.col("n_mp_pairs").sum().alias("_total_pairs"),
            pl.col("sitting").n_unique().alias("n_sittings"),
        ])
        .with_columns(
            (pl.col("_weighted_sum") / pl.col("_total_pairs")).cast(pl.Float32).alias("cohesion_score")
        )
        .drop(["_weighted_sum", "_total_pairs"])
        .sort(["club", "year_month"])
    )
