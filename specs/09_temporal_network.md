# Spec 09 — Temporal Network Evolution

## Purpose

Track how the MP voting agreement network changes over time — both across parliamentary
terms (term 9 vs 10) and within a term (month-by-month snapshots). Answers: "Has the
parliament become more polarised or more unified over time?"

---

## Two Levels of Temporal Analysis

### Level 1: Cross-Term (term 9 vs term 10)

Fetch and process both terms. For each term, compute the full agreement matrix and all
network metrics (spec 02). Compare metrics across terms.

Note: MP rosters differ between terms. Build separate matrices for each term; do not
attempt to merge MP IDs across terms.

### Level 2: Within-Term Monthly Snapshots

Within a single term, group votings by year-month. For each month `m`:
1. Filter `vote_matrix` to votings in month `m`
2. Require minimum co-presence of `MIN_COPRESENCE_MONTHLY = 10` (lower than global
   threshold since fewer votings per month)
3. Compute agreement matrix for that month's votings
4. Apply threshold sweep and compute network metrics
5. Run Leiden community detection on the monthly snapshot

---

## Metrics to Track Over Time

| Metric | Interpretation |
|--------|---------------|
| `density` | Overall connectivity |
| `n_components` | Fragmentation |
| `avg_clustering` | Local cohesion |
| `modularity` | Partisan structure strength |
| `nmi` | Alignment of communities with official clubs |
| `largest_component_frac` | Fraction of MPs in main connected component |

At threshold 0.50 (primary analysis threshold).

---

## Output Files

### `data/analysis/temporal_metrics.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `term` | `Int8` | |
| `year_month` | `Utf8` | e.g. "2024-03" or "FULL" for whole-term |
| `n_votings` | `Int32` | votings in this window |
| `threshold` | `Float32` | |
| `n_nodes` | `Int32` | |
| `n_edges` | `Int32` | |
| `density` | `Float64` | |
| `n_components` | `Int32` | |
| `largest_component_frac` | `Float32` | |
| `avg_clustering` | `Float64` | |
| `modularity` | `Float64` | Leiden modularity |
| `nmi` | `Float64` | vs official clubs |

### `data/analysis/temporal_party_cohesion.parquet`

Month × club cohesion matrix. (Separate from spec 04 which is per-sitting; this is
aggregated monthly for easier plotting.)

| Column | Type |
|--------|------|
| `term` | `Int8` |
| `year_month` | `Utf8` |
| `club` | `Categorical` |
| `cohesion_score` | `Float32` |
| `n_votings` | `Int32` |

---

## Visualisation

- Line plot: density, modularity, NMI over months (within term 10)
- Side-by-side bar: term 9 vs term 10 full-term metrics
- Animated network (optional, if time permits): monthly snapshots as frames

Save to `data/analysis/temporal_*.png`.

---

## Edge Cases

- Months with < 20 votings: include but flag `low_data = True`
- MPs who are new mid-term (by-elections): include from their first vote
- MPs who left mid-term: exclude from months after their last vote
