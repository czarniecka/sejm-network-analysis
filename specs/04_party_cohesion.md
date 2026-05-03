# Spec 04 — Party Cohesion Over Time

## Purpose

Measure how tightly each party votes together, and track how this changes over time
(per sitting and per month). Answers: "Which parties are most internally unified, and
did any party fragment or consolidate during the term?"

---

## Cohesion Score Definition

For a club `C` at sitting `S`:

```
cohesion(C, S) = fraction of (voting, MP_pair) combinations where both MPs voted the same
               = mean over all votings v in S, all pairs (i,j) in C present at v, of agree(i,j,v)
```

More precisely:
```
cohesion(C, S) = sum_{v in S} sum_{i<j in C, both present at v} agree(i,j,v)
               / sum_{v in S} sum_{i<j in C, both present at v} 1
```

where `agree(i,j,v) = 1` if both cast the same vote, else 0.

**Excluded**: ABSENT votes and VOTE_VALID votes.

---

## Efficient Computation

For each sitting `S` and each club `C`:

1. Filter `vote_matrix` rows to sitting `S` votings: shape `(V_s, N_c)` where `N_c` = club size
2. Filter `presence_matrix` rows to same sitting
3. For each voting `v` in sitting:
   - `pres_v = presence_matrix[v, :]`  — bool (N_c,)
   - `vote_v = vote_matrix[v, :]`      — float32 (N_c,)
   - `n_present = pres_v.sum()`
   - If `n_present < 2`: skip
   - `agree_v = (vote_v[pres_v][:, None] == vote_v[pres_v][None, :]).sum() - n_present`
     → count of agreeing pairs (exclude diagonal)
   - `pairs_v = n_present * (n_present - 1)` → total pairs
   - Accumulate `agree_v` and `pairs_v`
4. `cohesion(C, S) = total_agree / total_pairs`

To avoid nested Python loops over votings within a sitting, vectorise over all votings at
once using numpy broadcasting on the (V_s, N_c) vote matrix.

---

## Output Files

### `data/analysis/party_cohesion_by_sitting.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `club` | `Categorical` | |
| `sitting` | `Int16` | |
| `date` | `Date` | sitting date |
| `cohesion_score` | `Float32` | 0–1 |
| `n_votings` | `Int16` | votings in sitting |
| `n_mp_pairs` | `Int32` | total MP pairs considered |
| `term` | `Int8` | |

### `data/analysis/party_cohesion_by_month.parquet`

Aggregated by year-month (mean cohesion, weighted by n_mp_pairs).

| Column | Type |
|--------|------|
| `club` | `Categorical` |
| `year_month` | `Utf8` | e.g. "2024-03" |
| `cohesion_score` | `Float32` |
| `n_sittings` | `Int16` |
| `term` | `Int8` |

---

## Visualisation

- Line plot: cohesion over time per club (colour = club)
- Heatmap: club × sitting matrix of cohesion scores
- Box plot: distribution of sitting-level cohesion per club

Save to `data/analysis/party_cohesion_*.png`.

---

## Edge Cases

- Club with < 2 MPs present at a voting: skip that voting for that club
- Sitting with 0 valid votings for a club: set `cohesion_score = NaN`
- Newly formed clubs mid-term: include from their first appearance
