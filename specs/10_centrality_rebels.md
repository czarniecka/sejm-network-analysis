# Spec 10 — Centrality Analysis of Rebel MPs

## Purpose

Test the hypothesis that MPs who frequently vote against their party ("rebels") occupy
structurally important bridge positions in the voting network (high betweenness centrality).
Answers: "Are rebels bridges between party clusters, or are they simply marginal nodes?"

---

## Centrality Measures

Compute on the agreement network at threshold 0.50 (primary threshold):

| Measure | Function | Interpretation |
|---------|----------|---------------|
| Betweenness centrality | `nx.betweenness_centrality(G, normalized=True)` | Bridges between communities |
| PageRank | `nx.pagerank(G, weight="weight")` | Influence/importance |
| Degree centrality | `nx.degree_centrality(G)` | Raw connectivity |
| Eigenvector centrality | `nx.eigenvector_centrality(G, max_iter=500)` | Connected to highly connected MPs |
| Clustering coefficient | `nx.clustering(G)` | Local cohesion (low = bridge) |

Use the **weighted** network (edge weight = agreement rate) for PageRank and
eigenvector centrality. Use the **binary** network for betweenness (unweighted
shortest paths) as it is more interpretable for bridge detection.

---

## Joining with Rebel Scores

Join centrality measures with `rebels.parquet` on `mp_id`.

For MPs not in `rebels.parquet` (below MIN_REBEL_VOTES), set `rebel_rate = NaN`.

---

## Statistical Analysis

Compute Spearman rank correlation between:
- `rebel_rate` and `betweenness_centrality`
- `rebel_rate` and `clustering_coefficient` (hypothesis: rebels have lower clustering)
- `rebel_rate` and `pagerank`

Report p-values (scipy.stats.spearmanr).

Also compute per-club: mean centrality vs mean rebel rate to check if pattern holds
within parties.

---

## Output Files

### `data/analysis/centrality_rebels.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `mp_id` | `Int32` | |
| `first_name` | `Utf8` | |
| `last_name` | `Utf8` | |
| `club` | `Categorical` | |
| `rebel_rate` | `Float32` | NaN if below threshold |
| `betweenness` | `Float64` | |
| `pagerank` | `Float64` | |
| `degree_centrality` | `Float64` | |
| `eigenvector` | `Float64` | |
| `clustering_coeff` | `Float64` | |
| `threshold` | `Float32` | network threshold used |
| `term` | `Int8` | |

### `data/analysis/centrality_correlations.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `x_var` | `Utf8` | "rebel_rate" |
| `y_var` | `Utf8` | centrality measure name |
| `spearman_rho` | `Float64` | |
| `p_value` | `Float64` | |
| `n_samples` | `Int32` | |
| `threshold` | `Float32` | |
| `term` | `Int8` | |

---

## Visualisation

- Scatter plot: rebel_rate vs betweenness, coloured by club
- Scatter plot: rebel_rate vs clustering_coefficient, coloured by club
- Network plot: node size = betweenness, node colour = rebel_rate (gradient)
- Bar chart: top-10 highest betweenness MPs with their rebel_rate annotated

Save to `data/analysis/centrality_*.png`.

---

## Edge Cases

- Isolated nodes (no edges at threshold 0.50): betweenness = 0, exclude from correlations
- Eigenvector centrality may not converge: catch `nx.PowerIterationFailedConvergence`,
  set to NaN and log warning
- If rebel_rate NaN for many MPs: compute correlations only on non-NaN pairs
