"""
Party-switcher detection: MPs who changed club affiliation during a term.
Uses the club field embedded in individual vote records as the authoritative source.
"""

import logging

import polars as pl

from src.config import MIN_SWITCH_SITTINGS

logger = logging.getLogger(__name__)


def detect_club_switches(
    votes_df: pl.DataFrame,
    mps_df: pl.DataFrame,
    clubs_raw: list[dict],
    term: int,
    min_sittings: int = MIN_SWITCH_SITTINGS,
) -> pl.DataFrame:
    """
    Detect MPs who switched political clubs during the term.

    Algorithm:
    1. Sort votes by date per MP.
    2. Detect rows where club != previous club (using polars shift).
    3. Verify persistence: new club must appear in at least min_sittings consecutive
       sittings after the switch.
    4. Classify: PERSONAL switch vs CLUB_DISSOLUTION (if old club disappears entirely).

    Returns a DataFrame with one row per detected switch.
    """
    # Set of club codes that existed during the term
    known_clubs = {c.get("id") or c.get("club") or c.get("clubId", "") for c in clubs_raw}

    # Sort votes chronologically per MP; keep one row per (mp_id, sitting) with the club
    # (take the first club seen in that sitting to avoid within-sitting noise)
    per_sitting = (
        votes_df
        .filter(pl.col("vote") != "VOTE_VALID")
        .sort(["mp_id", "date", "sitting"])
        .group_by(["mp_id", "sitting"])
        .agg([
            pl.col("club").first().alias("club"),
            pl.col("date").first().alias("date"),
        ])
        .sort(["mp_id", "date"])
    )

    # Detect club changes using shift within mp_id groups
    with_prev = per_sitting.with_columns([
        pl.col("club").shift(1).over("mp_id").alias("prev_club"),
        pl.col("date").shift(1).over("mp_id").alias("prev_date"),
        pl.col("sitting").shift(1).over("mp_id").alias("prev_sitting"),
    ])

    # Candidate switches: club changed from previous sitting
    candidates = with_prev.filter(
        (pl.col("club") != pl.col("prev_club")) & pl.col("prev_club").is_not_null()
    )

    # Verify persistence: new club must appear in the NEXT sitting too
    # Build a set of (mp_id, sitting, club) for quick lookup
    sitting_club_set = set(
        zip(
            per_sitting["mp_id"].to_list(),
            per_sitting["sitting"].to_list(),
            per_sitting["club"].to_list(),
        )
    )

    rows = []
    for row in candidates.iter_rows(named=True):
        mp_id = row["mp_id"]
        new_club = row["club"]
        switch_sitting = row["sitting"]

        # Check if new club appears in the next sitting
        next_sittings = (
            per_sitting
            .filter(
                (pl.col("mp_id") == mp_id) &
                (pl.col("sitting") > switch_sitting)
            )
            .sort("sitting")
            .head(min_sittings)
        )
        if len(next_sittings) == 0:
            continue  # no subsequent votes, can't confirm persistence
        persists = all(
            r["club"] == new_club
            for r in next_sittings.iter_rows(named=True)
        )
        if not persists and len(next_sittings) >= min_sittings:
            continue  # reverted to old club — not a lasting switch

        # Classify switch type
        old_club = row["prev_club"]
        switch_type = "PERSONAL" if old_club in known_clubs else "CLUB_DISSOLUTION"

        # Count votings before and after switch
        n_before = int(
            votes_df
            .filter((pl.col("mp_id") == mp_id) & (pl.col("club") == old_club))
            .height
        )
        n_after = int(
            votes_df
            .filter((pl.col("mp_id") == mp_id) & (pl.col("club") == new_club))
            .height
        )

        rows.append({
            "mp_id": mp_id,
            "from_club": old_club,
            "to_club": new_club,
            "switch_date": row["date"],
            "switch_sitting": switch_sitting,
            "switch_type": switch_type,
            "n_votings_before": n_before,
            "n_votings_after": n_after,
            "term": term,
        })

    if not rows:
        logger.info("No club switches detected for term %d", term)
        return pl.DataFrame(schema={
            "mp_id": pl.Int32, "first_name": pl.Utf8, "last_name": pl.Utf8,
            "from_club": pl.Categorical, "to_club": pl.Categorical,
            "switch_date": pl.Date, "switch_sitting": pl.Int16,
            "switch_type": pl.Categorical,
            "n_votings_before": pl.Int32, "n_votings_after": pl.Int32,
            "term": pl.Int8,
        })

    switches_df = pl.DataFrame(rows).with_columns([
        pl.col("mp_id").cast(pl.Int32),
        pl.col("from_club").cast(pl.Categorical),
        pl.col("to_club").cast(pl.Categorical),
        pl.col("switch_date").cast(pl.Date),
        pl.col("switch_sitting").cast(pl.Int16),
        pl.col("switch_type").cast(pl.Categorical),
        pl.col("n_votings_before").cast(pl.Int32),
        pl.col("n_votings_after").cast(pl.Int32),
        pl.col("term").cast(pl.Int8),
    ])

    # Join MP names
    switches_df = switches_df.join(
        mps_df.select(["mp_id", "first_name", "last_name"]),
        on="mp_id",
        how="left",
    )

    logger.info("Detected %d club switches for term %d", len(switches_df), term)
    return switches_df.sort("switch_date")


def build_switch_summary(switches_df: pl.DataFrame, term: int) -> pl.DataFrame:
    """Aggregate switches by (from_club, to_club) pair."""
    if switches_df.is_empty():
        return pl.DataFrame()
    return (
        switches_df
        .group_by(["from_club", "to_club", "switch_type", "term"])
        .agg(pl.len().alias("n_switches"))
        .sort("n_switches", descending=True)
    )
