# Spec 03 — Community Detection

## Purpose

Run Leiden community detection on the MP agreement network at multiple thresholds and
compare detected communities to official party/club membership. This answers: "Do MPs
naturally cluster by party, and does the answer change with the threshold?"

---

## Algorithm Choice

Use **Leiden** algorithm via `leidenalg` + `python-igraph`.

Reasons over Louvain (python-louvain):
- Deterministic with fixed random seed
- Guarantees well-connected communities (no disconnected communities)
- Higher modularity in practice

---

## Input

For each threshold `t` in `AGREEMENT_THRESHOLDS`:
- `agreement_frac`: `(N, N)` float32 agreement matrix from spec 02
- `mp_ids`: `(N,)` int32 array of MP IDs (same order as matrix rows/cols)
- `mps.parquet`: provides `club` labels per MP

---

## Graph Construction

Build a **weighted** igraph graph:
```python
import igraph as ig
import leidenalg

# mask below threshold
mask = agreement_frac >= threshold  # (N, N) bool
np.fill_diagonal(mask, False)

# upper triangle only (undirected)
rows, cols = np.where(np.triu(mask, k=1))
weights = agreement_frac[rows, cols].tolist()

g = ig.Graph(n=N, edges=list(zip(rows.tolist(), cols.tolist())), directed=False)
g.es["weight"] = weights
g.vs["mp_id"] = mp_ids.tolist()
```

---

## Running Leiden

```python
partition = leidenalg.find_partition(
    g,
    leidenalg.ModularityVertexPartition,
    weights="weight",
    seed=42,
    n_iterations=10,
)
```

`seed=42` ensures reproducibility.

---

## Comparison to Official Club Labels

For each detected partition, compute:

| Metric | Function | Interpretation |
|--------|----------|---------------|
| Modularity | `partition.modularity` | Quality of community structure (0-1) |
| NMI | `sklearn.metrics.normalized_mutual_info_score` | 1 = perfect alignment with clubs |
| ARI | `sklearn.metrics.adjusted_rand_score` | 1 = perfect, 0 = random |
| Homogeneity | `sklearn.metrics.homogeneity_score` | Are communities pure (one club per community)? |
| Completeness | `sklearn.metrics.completeness_score` | Is each club in one community? |

Club label encoding: map club strings to integers for sklearn metrics.

---

## Output Files

### `data/networks/communities_threshold_{T}.parquet`

One row per MP.

| Column | Type | Notes |
|--------|------|-------|
| `mp_id` | `Int32` | |
| `first_name` | `Utf8` | joined from mps.parquet |
| `last_name` | `Utf8` | |
| `club` | `Categorical` | official club |
| `community_id` | `Int32` | Leiden community index |
| `threshold` | `Float32` | |
| `term` | `Int8` | |

### `data/networks/community_metrics.parquet`

One row per threshold.

| Column | Type |
|--------|------|
| `threshold` | `Float32` |
| `n_communities` | `Int32` |
| `modularity` | `Float64` |
| `nmi` | `Float64` |
| `ari` | `Float64` |
| `homogeneity` | `Float64` |
| `completeness` | `Float64` |
| `term` | `Int8` |

---

## Visualisation

For each threshold, save:
- `data/networks/communities_threshold_{T}.png` — network layout coloured by community,
  with official club shown as node shape/border

Console output: per-threshold table of metrics + confusion matrix (community × club).

---

## Edge Cases

- MPs with no edges at a given threshold: assign `community_id = -1` (isolated nodes)
- Single-MP communities: valid output, do not merge
- If graph is empty (threshold too high): skip, record NaN metrics
