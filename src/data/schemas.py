"""
Parquet schema definitions for all three core tables.
Used as reference documentation and for schema validation during assembly.
"""

import polars as pl

# ---------------------------------------------------------------------------
# Schema: mps.parquet
# ---------------------------------------------------------------------------
MPS_SCHEMA: dict[str, pl.DataType] = {
    "mp_id": pl.Int32,
    "first_name": pl.Utf8,
    "last_name": pl.Utf8,
    "club": pl.Categorical,
    "active": pl.Boolean,
    "birth_date": pl.Date,
    "birth_location": pl.Utf8,
    "voivodeship": pl.Categorical,
    "district_name": pl.Utf8,
    "district_num": pl.Int16,
    "education_level": pl.Categorical,
    "profession": pl.Utf8,
    "number_of_votes": pl.Int32,
}

# ---------------------------------------------------------------------------
# Schema: votings.parquet
# ---------------------------------------------------------------------------
VOTINGS_SCHEMA: dict[str, pl.DataType] = {
    "voting_key": pl.Utf8,
    "sitting": pl.Int16,
    "voting_num": pl.Int16,
    "date": pl.Datetime,
    "title": pl.Utf8,
    "topic": pl.Utf8,
    "description": pl.Utf8,
    "kind": pl.Categorical,
    "majority_type": pl.Categorical,
    "yes_count": pl.Int16,
    "no_count": pl.Int16,
    "abstain_count": pl.Int16,
    "not_participating": pl.Int16,
    "total_voted": pl.Int16,
    "term": pl.Int8,
}

# ---------------------------------------------------------------------------
# Schema: votes.parquet
# ---------------------------------------------------------------------------
VOTES_SCHEMA: dict[str, pl.DataType] = {
    "voting_key": pl.Utf8,
    "sitting": pl.Int16,
    "voting_num": pl.Int16,
    "date": pl.Date,
    "mp_id": pl.Int32,
    "club": pl.Categorical,
    "vote": pl.Categorical,
    "term": pl.Int8,
}
