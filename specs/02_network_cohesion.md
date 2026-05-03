# Spec 02 — Network Cohesion Analysis

## Purpose

Build pairwise MP agreement networks at multiple thresholds and measure how network
properties change as the threshold is varied. This answers: "How tightly connected is the
parliament if we only connect MPs who agree at least X% of the time?"

---

## Agreement Definition

Two MPs **agree** on a voting if:
- Both were **present** (vote ≠ ABSENT and vote ≠ VOTE_VALID)
- Both cast the **same** vote (YES=YES, NO=NO, or ABSTAIN=ABSTAIN)

**Agreement rate** for pair (i, j):
```
agreement_rate(i, j) = count(agree) / count(co-present)
```

If `count(co-present) < MIN_COPRESENCE` (default 50), the pair is excluded (set to NaN).

---

## Agreement Matrix Computation

Input:
- `vote_matrix`: shape `(V, N)` float32 — encoded votes per (voting, MP).
  Values: 1.0=YES, -1.0=NO, 0.0=ABSTAIN, NaN=ABSENT/excluded
- `presence_matrix`: shape `(V, N)` bool — True where MP was present

Output:
- `agreement_frac`: shape `(N, N)` float32 — pairwise agreement rates (NaN if below min copresence)
- `copresence`: shape `(N, N)` int32 — pairwise co-presence counts

### Vectorized Implementation (BLAS matmul trick)

```python
pres = presence_matrix.astype(np.float32)           # (V, N)
v_yes  = ((vote_matrix == 1.0) & presence_matrix).astype(np.float32)
v_no   = ((vote_matrix == -1.0) & presence_matrix).astype(np.float32)
v_abs  = ((vote_matrix == 0.0) & presence_matrix).astype(np.float32)

copresence = pres.T @ pres                          # (N, N)
agree_raw  = v_yes.T @ v_yes + v_no.T @ v_no + v_abs.T @ v_abs  # (N, N)

agreement_frac = np.where(
    copresence >= MIN_COPRESENCE,
    agree_raw / np.maximum(copresence, 1),
    np.nan
)
np.fill_diagonal(agreement_frac, np.nan)            # exclude self-pairs
```

Three float32 matmuls on (3873 × 460) → < 5 s on Apple Silicon (Accelerate BLAS).
Memory: 3 × 3873 × 460 × 4 bytes ≈ 21 MB.

---

## Threshold Sweep

Thresholds: `[0.30, 0.50, 0.70, 0.90]` (configurable in `config.py`).

For each threshold `t`:
1. Build binary adjacency matrix: `adj = (agreement_frac >= t).astype(np.uint8)`
   (NaN → 0, i.e. no edge)
2. Convert to networkx `Graph`
3. Compute metrics (see below)

---

## Network Metrics

For each threshold, compute and record:

| Metric | networkx function | Notes |
|--------|------------------|-------|
| `n_nodes` | `G.number_of_nodes()` | MPs with ≥1 edge |
| `n_edges` | `G.number_of_edges()` | |
| `density` | `nx.density(G)` | |
| `n_components` | `nx.number_connected_components(G)` | |
| `largest_component_size` | `len(max(..., key=len))` | |
| `largest_component_frac` | `largest / N` | fraction of all MPs |
| `avg_clustering` | `nx.average_clustering(G)` | |
| `diameter_lcc` | `nx.diameter(lcc)` | on largest connected component |
| `avg_path_length_lcc` | `nx.average_shortest_path_length(lcc)` | on LCC |

For large networks (high threshold → dense graph), diameter and avg path length are
computed on the LCC only to keep runtime tractable.

---

## Output Files

### `data/networks/agreement_matrix.npz`

Scipy sparse? No — store as dense numpy `.npy` file (460×460 float32 ≈ 0.8 MB).
```
data/networks/agreement_matrix.npy   # float32, shape (N, N)
data/networks/copresence_matrix.npy  # int32,   shape (N, N)
data/networks/mp_ids.npy             # int32,   shape (N,) — MP id order
```

### `data/networks/network_metrics.parquet`

| Column | Type |
|--------|------|
| `threshold` | `Float32` |
| `n_nodes` | `Int32` |
| `n_edges` | `Int32` |
| `density` | `Float64` |
| `n_components` | `Int32` |
| `largest_component_size` | `Int32` |
| `largest_component_frac` | `Float32` |
| `avg_clustering` | `Float64` |
| `diameter_lcc` | `Int32` |
| `avg_path_length_lcc` | `Float64` |
| `term` | `Int8` |

---

## Visualisation (script output)

For each threshold, save:
- `data/networks/threshold_{T}_adjacency.png` — heatmap of agreement matrix (MPs ordered by club)
- `data/networks/metrics_plot.png` — 4-panel plot: density, n_components, avg_clustering, diameter vs threshold
