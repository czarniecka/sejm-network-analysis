"""
Inter-party agreement matrix: how often do the majority votes of each party pair align?
"""

import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def compute_club_majority_votes(
    vote_matrix: np.ndarray,
    presence_matrix: np.ndarray,
    mp_ids: list[int],
    mp_club: dict[int, str],
) -> tuple[np.ndarray, list[str]]:
    """
    Compute the majority vote for each club at each voting.

    Returns:
        majority_matrix: (V, n_clubs) float32 — 1=YES, -1=NO, 0=ABSTAIN, NaN=absent/ambiguous
        clubs:           list of club names in column order
    """
    clubs = sorted(set(mp_club.values()))
    club_index = {c: i for i, c in enumerate(clubs)}
    mp_index = {mp_id: i for i, mp_id in enumerate(mp_ids)}
    V = vote_matrix.shape[0]
    n_clubs = len(clubs)

    majority_matrix = np.full((V, n_clubs), np.nan, dtype=np.float32)

    # Build club -> column indices array
    club_cols: dict[str, np.ndarray] = {}
    for mp_id, club in mp_club.items():
        if mp_id in mp_index:
            club_cols.setdefault(club, []).append(mp_index[mp_id])
    club_cols = {c: np.array(idxs) for c, idxs in club_cols.items()}

    for club, col_arr in club_cols.items():
        ci = club_index[club]
        club_votes = vote_matrix[:, col_arr]    # (V, N_c)
        club_pres = presence_matrix[:, col_arr]  # (V, N_c)

        yes_count = ((club_votes == 1.0) & club_pres).sum(axis=1).astype(np.float32)
        no_count  = ((club_votes == -1.0) & club_pres).sum(axis=1).astype(np.float32)
        abs_count = ((club_votes == 0.0) & club_pres).sum(axis=1).astype(np.float32)
        total_pres = club_pres.sum(axis=1).astype(np.float32)

        for vi in range(V):
            if total_pres[vi] < 1:
                continue  # NaN (all absent)
            y, n, a = yes_count[vi], no_count[vi], abs_count[vi]
            max_cnt = max(y, n, a)
            winners = []
            if y == max_cnt: winners.append(1.0)
            if n == max_cnt: winners.append(-1.0)
            if a == max_cnt: winners.append(0.0)
            if len(winners) == 1:
                majority_matrix[vi, ci] = winners[0]
            # else NaN (ambiguous)

    return majority_matrix, clubs


def compute_party_agreement_matrix(
    vote_matrix: np.ndarray,
    presence_matrix: np.ndarray,
    mp_ids: list[int],
    mp_club: dict[int, str],
    term: int,
) -> pl.DataFrame:
    """
    Compute the inter-party majority agreement matrix.

    Returns a long-format DataFrame: club1, club2, agreement_rate, n_valid_votings, term.
    """
    majority_matrix, clubs = compute_club_majority_votes(
        vote_matrix, presence_matrix, mp_ids, mp_club
    )
    n_clubs = len(clubs)
    V = majority_matrix.shape[0]

    rows = []
    for i, c1 in enumerate(clubs):
        for j, c2 in enumerate(clubs):
            m1 = majority_matrix[:, i]
            m2 = majority_matrix[:, j]
            valid = ~np.isnan(m1) & ~np.isnan(m2)
            n_valid = int(valid.sum())
            if n_valid == 0:
                agreement = float("nan")
            else:
                agreement = float((m1[valid] == m2[valid]).sum() / n_valid)
            rows.append({
                "club1": c1,
                "club2": c2,
                "agreement_rate": agreement,
                "n_valid_votings": n_valid,
                "term": term,
            })

    logger.info("Computed inter-party agreement matrix for %d clubs", n_clubs)
    return pl.DataFrame(rows).with_columns([
        pl.col("club1").cast(pl.Categorical),
        pl.col("club2").cast(pl.Categorical),
        pl.col("agreement_rate").cast(pl.Float32),
        pl.col("n_valid_votings").cast(pl.Int32),
        pl.col("term").cast(pl.Int8),
    ])
