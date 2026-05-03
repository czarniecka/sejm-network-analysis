# Spec 11 — Cross-Party Voting Blocs

## Purpose

Detect groups of MPs who vote together on specific topics, even if they belong to
different official parties. Answers: "Do PiS and Konfederacja form a voting bloc on
certain topics? Which club pairs co-occur most in the same topic-specific community?"

---

## Approach

For each BERTopic topic `t` (from spec 08) with at least `MIN_TOPIC_VOTINGS = 5` votings:

1. **Build topic sub-network**:
   - Filter `vote_matrix` to votings assigned to topic `t`
   - Compute agreement matrix on this subset (with `MIN_COPRESENCE_TOPIC = 3`)
   - Apply threshold 0.60 (higher than global, since topic networks are denser)
   - Convert to igraph weighted graph

2. **Run Leiden** on the topic sub-network (same parameters as spec 03, seed=42)

3. **Characterise communities**:
   - For each community: count MPs per club
   - A community is **cross-party** if it contains MPs from ≥ 2 clubs AND no single club
     holds > 70% of members
   - Compute `cross_party_score = 1 - max(club_fractions)` ∈ [0, 1]

4. **Identify recurring cross-party pairs**:
   - For each pair of clubs (C1, C2), count topics where they appear in the same community
   - `bloc_affinity(C1, C2) = n_topics_same_community / n_topics_both_present`

---

## Output Files

### `data/analysis/voting_blocs.parquet`

One row per (topic, community, MP).

| Column | Type | Notes |
|--------|------|-------|
| `topic_id` | `Int32` | from BERTopic |
| `topic_label` | `Utf8` | top words |
| `community_id` | `Int32` | Leiden community within topic |
| `mp_id` | `Int32` | |
| `first_name` | `Utf8` | |
| `last_name` | `Utf8` | |
| `club` | `Categorical` | |
| `community_size` | `Int32` | |
| `cross_party_score` | `Float32` | 0=pure, 1=maximally mixed |
| `is_cross_party` | `Boolean` | cross_party_score threshold |
| `term` | `Int8` | |

### `data/analysis/bloc_affinity.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `club1` | `Categorical` | |
| `club2` | `Categorical` | |
| `bloc_affinity` | `Float32` | fraction of topics they co-bloc |
| `n_topics_same_community` | `Int32` | |
| `n_topics_both_present` | `Int32` | |
| `term` | `Int8` | |

### `data/analysis/bloc_summary.parquet`

Top cross-party blocs per topic (for easy lookup).

| Column | Type | Notes |
|--------|------|-------|
| `topic_id` | `Int32` | |
| `topic_label` | `Utf8` | |
| `mean_pair_agreement` | `Float32` | from spec 08 |
| `n_cross_party_communities` | `Int32` | |
| `dominant_bloc_clubs` | `Utf8` | comma-separated club names in largest cross-party community |
| `term` | `Int8` | |

---

## Visualisation

- Heatmap: clubs × clubs bloc_affinity matrix
- Network plot per high-interest topic: nodes = MPs, coloured by club, communities shown
  as clusters
- Bar chart: top-10 topics with highest cross_party_score in largest community

Save to `data/analysis/blocs_*.png`.

---

## Edge Cases

- Topic with < MIN_TOPIC_VOTINGS: skip bloc analysis
- Topic sub-network with no edges at threshold 0.60: all MPs in separate communities
- Single-club topic (all MPs from one party in that topic): cross_party_score = 0
- Topics where outlier cluster (-1) is largest: skip that community
