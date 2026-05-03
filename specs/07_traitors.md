# Spec 07 — Party Switchers (Traitors)

## Purpose

Detect MPs who changed their club affiliation during a parliamentary term. The source of
truth is the `club` field embedded in each individual vote record, which reflects the
club the MP belonged to at the time of each vote.

---

## Detection Algorithm

For each MP (grouped by `mp_id`):

1. Sort vote records by `date` ascending
2. Extract the time-ordered sequence of `(date, sitting, club)` tuples
3. Detect transitions: `club[t] != club[t-1]` using polars `shift()`
4. A transition is flagged as a **switch** only if it persists for at least
   `MIN_SWITCH_SITTINGS = 2` consecutive sittings (reduces false positives from
   data entry errors or temporary re-labelling)

```python
# In polars:
switches = (
    votes_df
    .sort("date")
    .with_columns([
        pl.col("club").shift(1).over("mp_id").alias("prev_club"),
        pl.col("sitting").shift(1).over("mp_id").alias("prev_sitting"),
    ])
    .filter(pl.col("club") != pl.col("prev_club"))
    .filter(pl.col("prev_club").is_not_null())
)
```

Then verify persistence: for each detected switch at sitting `s`, check that the new
club appears in sitting `s+1` as well.

---

## Distinguishing Genuine Switches from Club Renames

Cross-reference with `clubs.json` for the term. If a club name disappears entirely from
the clubs list between two sittings, the transition is a **club dissolution/rename**, not
a personal switch. Flag these differently in the output.

---

## Output Files

### `data/analysis/party_switchers.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `mp_id` | `Int32` | |
| `first_name` | `Utf8` | |
| `last_name` | `Utf8` | |
| `from_club` | `Categorical` | club before switch |
| `to_club` | `Categorical` | club after switch |
| `switch_date` | `Date` | date of first voting under new club |
| `switch_sitting` | `Int16` | sitting number of first new-club vote |
| `switch_type` | `Categorical` | PERSONAL / CLUB_DISSOLUTION |
| `n_votings_before` | `Int32` | votings under `from_club` |
| `n_votings_after` | `Int32` | votings under `to_club` |
| `term` | `Int8` | |

### `data/analysis/switch_summary.parquet`

Summary by (from_club, to_club) pair.

| Column | Type |
|--------|------|
| `from_club` | `Categorical` |
| `to_club` | `Categorical` |
| `n_switches` | `Int32` |
| `switch_type` | `Categorical` |
| `term` | `Int8` |

---

## Visualisation

- Sankey/alluvial diagram: flows between clubs
- Timeline: scatter plot of switch events (x=date, y=club, colour=from_club)

Save to `data/analysis/traitors_*.png`.

---

## Edge Cases

- MP with only 1 unique club across all votes: not a switcher
- Club code changes (e.g. "KO" → "KKO"): treat as club rename, not personal switch
- MP who returns to original club: record two switches (A→B and B→A)
- MPs in mps.parquet marked as `active=False`: include if they have vote records
