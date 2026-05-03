"""
Rebel MP detection: MPs who frequently vote against their party majority.
"""

import logging

import numpy as np
import polars as pl

from src.config import MIN_REBEL_VOTES

logger = logging.getLogger(__name__)


def compute_rebel_scores(
    vote_matrix: np.ndarray,
    presence_matrix: np.ndarray,
    mp_ids: list[int],
    voting_keys: list[str],
    votes_df: pl.DataFrame,
    term: int,
    min_votes: int = MIN_REBEL_VOTES,
) -> pl.DataFrame:
    """
    Compute rebel score for each MP.

    For each voting where an MP was present, compare their vote to their club's
    majority vote (computed excluding the MP themselves). Count deviations.

    Returns a DataFrame sorted by rebel_rate descending.
    Only includes MPs with total_votes >= min_votes.
    """
    # mp_id -> column index
    mp_index = {mp_id: i for i, mp_id in enumerate(mp_ids)}
    N = len(mp_ids)
    V = len(voting_keys)

    # Build per-voting club map: for each voting, the club of each MP
    # Use the club from votes_df (club at time of vote)
    vkey_idx = {k: i for i, k in enumerate(voting_keys)}

    # Build a club matrix: (V, N) — club string per cell (or "" if absent)
    # We'll use a simpler approach: build mp_id -> club from votes_df per voting
    # But for efficiency, use the most common club per MP across all votings
    mp_club_series = (
        votes_df
        .sort("date")
        .group_by("mp_id")
        .agg(pl.col("club").last().alias("club"))
    )
    mp_club: dict[int, str] = {
        row["mp_id"]: row["club"]
        for row in mp_club_series.iter_rows(named=True)
    }

    # Group MPs by club
    club_to_cols: dict[str, list[int]] = {}
    for mp_id in mp_ids:
        club = mp_club.get(mp_id, "")
        if club:
            idx = mp_index[mp_id]
            club_to_cols.setdefault(club, []).append(idx)

    # Output arrays
    rebel_counts = np.zeros(N, dtype=np.int32)
    total_votes = np.zeros(N, dtype=np.int32)
    ambiguous_votes = np.zeros(N, dtype=np.int32)

    for club, col_indices in club_to_cols.items():
        if len(col_indices) < 2:
            continue
        col_arr = np.array(col_indices)

        club_votes = vote_matrix[:, col_arr]    # (V, N_c)
        club_pres = presence_matrix[:, col_arr]  # (V, N_c)

        # Sum of each vote type per voting (for the whole club)
        yes_total = ((club_votes == 1.0) & club_pres).sum(axis=1)   # (V,)
        no_total  = ((club_votes == -1.0) & club_pres).sum(axis=1)
        abs_total = ((club_votes == 0.0) & club_pres).sum(axis=1)

        # For each MP in club (column index ci, club array index k)
        for k, ci in enumerate(col_arr):
            mp_pres = club_pres[:, k]   # (V,) bool
            mp_vote = club_votes[:, k]  # (V,) float

            present_rows = np.where(mp_pres)[0]
            if len(present_rows) == 0:
                continue

            for vi in present_rows:
                # Subtract MP's own contribution from club totals
                my_vote = mp_vote[vi]
                y = yes_total[vi] - (1 if my_vote == 1.0 else 0)
                n = no_total[vi]  - (1 if my_vote == -1.0 else 0)
                a = abs_total[vi] - (1 if my_vote == 0.0 else 0)

                # Find majority (mode), skipping ties
                counts = {1.0: y, -1.0: n, 0.0: a}
                max_count = max(counts.values())
                winners = [v for v, c in counts.items() if c == max_count]

                if len(winners) > 1 or max_count == 0:
                    ambiguous_votes[ci] += 1
                    continue

                majority = winners[0]
                total_votes[ci] += 1
                if my_vote != majority:
                    rebel_counts[ci] += 1

    # Build output DataFrame
    rows = []
    for i, mp_id in enumerate(mp_ids):
        tv = int(total_votes[i])
        if tv < min_votes:
            continue
        rc = int(rebel_counts[i])
        rows.append({
            "mp_id": mp_id,
            "club": mp_club.get(mp_id, ""),
            "rebel_rate": rc / tv if tv > 0 else 0.0,
            "rebel_count": rc,
            "total_votes": tv,
            "ambiguous_votes": int(ambiguous_votes[i]),
            "term": term,
        })

    # Join names from votes_df
    name_map = (
        votes_df
        .group_by("mp_id")
        .agg([pl.col("club").first()])
        .select("mp_id")
    )
    # Get names from mps parquet if available (joined later in script)
    df = (
        pl.DataFrame(rows)
        .with_columns([
            pl.col("mp_id").cast(pl.Int32),
            pl.col("club").cast(pl.Categorical),
            pl.col("rebel_rate").cast(pl.Float32),
            pl.col("rebel_count").cast(pl.Int32),
            pl.col("total_votes").cast(pl.Int32),
            pl.col("ambiguous_votes").cast(pl.Int32),
            pl.col("term").cast(pl.Int8),
        ])
        .sort("rebel_rate", descending=True)
    )
    logger.info("Computed rebel scores for %d MPs (min_votes=%d)", len(df), min_votes)
    return df
