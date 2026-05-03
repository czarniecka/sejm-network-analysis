# Spec 06 — Inter-Party Agreement Matrix

## Purpose

Measure how often the majority votes of different parties align. Produces a clubs × clubs
correlation matrix. Answers: "Which parties tend to vote together, and which are most
opposed?"

---

## Definition

For each pair of clubs (C1, C2):

```
party_agreement(C1, C2) = fraction of votings where majority_vote(C1) == majority_vote(C2)
```

where `majority_vote(C, v)` = mode of {YES, NO, ABSTAIN} among present members of `C` at
voting `v`.

**Excluded from denominator**: votings where either club has an ambiguous majority (tie)
or all members absent.

The matrix is symmetric: `party_agreement(C1, C2) == party_agreement(C2, C1)`.
Diagonal is 1.0 (a club always agrees with itself).

---

## Computation

```python
# majority_votes shape: (V, n_clubs) — majority vote per voting per club
# encoded as 1=YES, -1=NO, 0=ABSTAIN, NaN=absent/ambiguous

# For each pair (c1, c2):
# valid = ~isnan(majority[v,c1]) & ~isnan(majority[v,c2])  for all v
# agree = (majority[v,c1] == majority[v,c2]) & valid
# agreement(c1,c2) = agree.sum() / valid.sum()
```

Compute `majority_votes` once per voting using `np.nanargmax` on club-level vote counts.

---

## Output Files

### `data/analysis/party_correlation_matrix.parquet`

Wide format: rows = clubs, columns = clubs, values = agreement rate.

| Column | Type |
|--------|------|
| `club` | `Utf8` (row label) |
| `{club_name}` | `Float32` (one column per club) |
| `term` | `Int8` |

Also save as `data/analysis/party_correlation_matrix.npy` for easy numpy loading.

---

## Visualisation

- Heatmap with annotation: clubs × clubs, colour = agreement rate (0–1)
  - Use diverging colormap centred at 0.5 (chance level for binary vote)
  - Annotate cells with agreement percentage
- Dendrogram clustered heatmap (seaborn clustermap)

Save to `data/analysis/party_correlation_heatmap.png`.

---

## Edge Cases

- Club with 0 valid votings (all absent or all ambiguous): set row/col to NaN
- Very small clubs (< 3 members): include but flag in output
- Club dissolution mid-term: include all votings where that club had members
