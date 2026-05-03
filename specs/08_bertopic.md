# Spec 08 — BERTopic Voting Topic Modelling

## Purpose

Discover latent topics in voting titles using BERTopic, then characterise each topic by
its consensus level (all MPs agree) or controversy level (MPs split ~50/50). Answers:
"What kinds of issues produce universal agreement vs. deep partisan splits?"

---

## Input Text Preparation

For each voting, concatenate:
```
text = f"{title} {topic} {description}"
```

Where `topic` and `description` may be null → replace with empty string.
Strip leading/trailing whitespace. Filter out texts shorter than 10 characters.

---

## Embedding Model

Model: `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers)
- Supports Polish natively
- ~117 MB download
- 384-dimensional embeddings

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="mps")
embeddings = model.encode(texts, batch_size=256, show_progress_bar=True)
```

Cache embeddings to `data/analysis/voting_embeddings.npy`. On re-run, load from cache
if file exists and shape matches number of texts.

---

## BERTopic Configuration

```python
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

umap_model = UMAP(
    n_components=5,
    n_neighbors=15,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)
hdbscan_model = HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)
vectorizer = CountVectorizer(
    ngram_range=(1, 2),
    stop_words=None,       # Polish stopwords not needed; BERTopic uses c-TF-IDF
    min_df=2,
)
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    nr_topics="auto",
    calculate_probabilities=False,
    verbose=True,
)
topics, _ = topic_model.fit_transform(texts, embeddings)
```

Topic `-1` = outlier/noise class. Keep but mark separately.

---

## Agreement Metric per Topic

For each topic `t`, compute the **mean pairwise agreement rate** across all votings
assigned to that topic:

```
topic_agreement(t) = mean over v in topic(t) of mean_pair_agreement(v)
```

where `mean_pair_agreement(v)` = fraction of present MP pairs that agreed on voting `v`.

This scalar can be computed from the `yes_count`, `no_count`, `abstain_count` columns
without rebuilding the full pair matrix:

```python
# For voting v with y YES, n NO, a ABSTAIN, total = y+n+a present:
# agree_pairs = C(y,2) + C(n,2) + C(a,2)
# total_pairs = C(total, 2)
# pair_agreement(v) = agree_pairs / total_pairs
```

where C(k,2) = k*(k-1)/2.

---

## Cross-Party Analysis per Topic

For each topic `t` and each club `C`:
- Compute the majority vote of club `C` across votings in topic `t`
- Produce a `(n_clubs, n_topics)` majority-vote matrix
- Identify topic clusters where:
  - All clubs agree (consensus topic)
  - Only opposition clubs disagree (partisan topic)
  - Cross-party split (e.g. PiS+Konfederacja vs KO+TD)

---

## Output Files

### `data/analysis/voting_embeddings.npy`

Shape `(V, 384)` float32. One row per voting (same order as `votings.parquet`).

### `data/analysis/voting_topics.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `voting_key` | `Utf8` | |
| `sitting` | `Int16` | |
| `voting_num` | `Int16` | |
| `date` | `Date` | |
| `title` | `Utf8` | |
| `topic_id` | `Int32` | -1 = outlier |
| `pair_agreement` | `Float32` | per-voting agreement rate |
| `term` | `Int8` | |

### `data/analysis/topic_summary.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `topic_id` | `Int32` | |
| `top_words` | `Utf8` | comma-separated top 5 words |
| `n_votings` | `Int32` | |
| `mean_pair_agreement` | `Float32` | 1=full consensus, 0.5=random |
| `std_pair_agreement` | `Float32` | variance within topic |
| `is_outlier` | `Boolean` | topic_id == -1 |
| `term` | `Int8` | |

### `data/analysis/topic_club_majority.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `topic_id` | `Int32` | |
| `club` | `Categorical` | |
| `majority_yes_frac` | `Float32` | fraction of topic votings where club majority=YES |
| `majority_no_frac` | `Float32` | |
| `majority_abs_frac` | `Float32` | |
| `n_votings` | `Int32` | |
| `term` | `Int8` | |

---

## Visualisation

- Bar chart: topics ranked by `mean_pair_agreement` (most consensus → most controversy)
- Heatmap: clubs × topics (majority vote direction)
- UMAP 2D scatter coloured by topic (for all votings)

Save to `data/analysis/topics_*.png`.

---

## Edge Cases

- Texts shorter than 10 chars: assign `topic_id = -1` without modelling
- Duplicate texts: keep all, they get the same topic naturally
- Topics with < 5 votings: keep but flag with low confidence
