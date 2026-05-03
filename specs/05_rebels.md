# Spec 05 — Rebel MPs

## Purpose

Identify MPs who frequently vote against the majority of their own party. Answers: "Which
MPs are the most rebellious, and how does this distribute across parties?"

---

## Rebel Score Definition

For each MP `m` in club `C`:

1. For each voting `v` where MP `m` was present and vote ≠ VOTE_VALID:
   a. Compute the **club majority vote**: the mode of {YES, NO, ABSTAIN} among all
      present members of `C` (excluding `m` themselves to avoid self-influence).
   b. If `m`'s vote ≠ club majority vote → increment `rebel_count`
   c. Increment `total_votes`

2. `rebel_rate(m) = rebel_count / total_votes`

**Minimum threshold**: MP must have `total_votes >= MIN_REBEL_VOTES` (default 100).
MPs below threshold are excluded from the output.

**Tie-breaking for majority vote**: if two vote values tie (e.g. 50 YES, 50 NO), the
voting is considered ambiguous — skip it (do not count as rebel or non-rebel).

---

## Efficient Computation

Vectorise over votings using the vote matrix:

```python
# For each club C with member indices club_mask (bool, shape N):
# vote_matrix shape: (V, N), presence_matrix shape: (V, N)

club_votes = vote_matrix[:, club_mask]         # (V, N_c)
club_pres  = presence_matrix[:, club_mask]     # (V, N_c)

# For each MP i in club (column i of club_votes):
# majority = mode of club_votes[v, j≠i] for present j
# Achieved by: sum_yes, sum_no, sum_abs per row, then subtract MP i's contribution
sum_yes = ((club_votes == 1.0) & club_pres).sum(axis=1)   # (V,)
sum_no  = ((club_votes == -1.0) & club_pres).sum(axis=1)
sum_abs = ((club_votes == 0.0) & club_pres).sum(axis=1)

# For MP at column index idx:
# majority_yes[v] = sum_yes[v] - (club_votes[v, idx] == 1.0 & club_pres[v, idx])
# etc. → find argmax → compare to mp's vote
```

This avoids a Python loop over votings for each MP.

---

## Output Files

### `data/analysis/rebels.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `mp_id` | `Int32` | |
| `first_name` | `Utf8` | |
| `last_name` | `Utf8` | |
| `club` | `Categorical` | final/most recent club |
| `rebel_rate` | `Float32` | 0–1 (higher = more rebellious) |
| `rebel_count` | `Int32` | |
| `total_votes` | `Int32` | |
| `ambiguous_votes` | `Int32` | votings skipped due to tie |
| `term` | `Int8` | |

Sorted by `rebel_rate` descending.

---

## Visualisation

- Bar chart: top-20 rebels coloured by club
- Box plot: rebel_rate distribution per club
- Scatter: rebel_rate vs total_votes (to check if high rebels are also low participants)

Save to `data/analysis/rebels_*.png`.

---

## Edge Cases

- MP who switches club mid-term: use the club from each individual vote record (from votes.parquet)
- Club with 1 member: cannot compute majority, skip all votings for that MP/club
- All club members absent for a voting: skip that voting
