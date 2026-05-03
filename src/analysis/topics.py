"""
BERTopic topic modelling on voting titles and descriptions.
Computes per-topic agreement metrics and cross-party majority vote analysis.
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl

from src.config import (
    ANALYSIS_DIR,
    BERTOPIC_MIN_TOPIC_SIZE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
)

logger = logging.getLogger(__name__)

EMBEDDINGS_CACHE = ANALYSIS_DIR / "voting_embeddings.npy"


def prepare_texts(votings_df: pl.DataFrame) -> list[str]:
    """Concatenate title + topic + description into one text per voting."""
    texts = []
    for row in votings_df.iter_rows(named=True):
        parts = [row.get("title") or "", row.get("topic") or "", row.get("description") or ""]
        text = " ".join(p for p in parts if p).strip()
        texts.append(text if len(text) >= 10 else "")
    return texts


def get_or_compute_embeddings(texts: list[str]) -> np.ndarray:
    """Load embeddings from cache or compute and cache them."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    if EMBEDDINGS_CACHE.exists():
        cached = np.load(EMBEDDINGS_CACHE)
        if cached.shape[0] == len(texts):
            logger.info("Loaded %d embeddings from cache", len(texts))
            return cached
        logger.warning("Cache shape mismatch (%d vs %d), recomputing", cached.shape[0], len(texts))

    from sentence_transformers import SentenceTransformer
    logger.info("Computing embeddings with %s on %s …", EMBEDDING_MODEL, EMBEDDING_DEVICE)
    model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    np.save(EMBEDDINGS_CACHE, embeddings)
    logger.info("Saved embeddings to %s", EMBEDDINGS_CACHE)
    return embeddings


def run_bertopic(texts: list[str], embeddings: np.ndarray) -> tuple:
    """
    Fit BERTopic model.

    Returns:
        topic_model: fitted BERTopic instance
        topics:      (V,) int array of topic IDs (-1 = outlier)
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    umap_model = UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=2)

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        min_topic_size=BERTOPIC_MIN_TOPIC_SIZE,
        nr_topics="auto",
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(texts, embeddings)
    logger.info("BERTopic: %d topics found", len(set(topics)) - (1 if -1 in topics else 0))
    return topic_model, np.array(topics)


def compute_pair_agreement_per_voting(votings_df: pl.DataFrame) -> np.ndarray:
    """
    Compute the pairwise agreement rate for each voting from aggregate counts.

    pair_agreement(v) = (C(y,2) + C(n,2) + C(a,2)) / C(total,2)
    where C(k,2) = k*(k-1)//2.

    Returns (V,) float32 array, one per row in votings_df.
    """
    yes = votings_df["yes_count"].to_numpy().astype(np.int64)
    no  = votings_df["no_count"].to_numpy().astype(np.int64)
    ab  = votings_df["abstain_count"].to_numpy().astype(np.int64)
    total = yes + no + ab

    def comb2(k: np.ndarray) -> np.ndarray:
        return k * (k - 1) // 2

    agree_pairs = comb2(yes) + comb2(no) + comb2(ab)
    total_pairs = comb2(total)

    pair_agreement = np.where(total_pairs > 0, agree_pairs / total_pairs, np.nan)
    return pair_agreement.astype(np.float32)


def build_topic_outputs(
    votings_df: pl.DataFrame,
    topics: np.ndarray,
    topic_model,
    term: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Build voting_topics.parquet and topic_summary.parquet.
    """
    pair_agreement = compute_pair_agreement_per_voting(votings_df)

    # voting_topics
    voting_rows = []
    for i, row in enumerate(votings_df.iter_rows(named=True)):
        dt = row["date"]
        voting_rows.append({
            "voting_key": row["voting_key"],
            "sitting": row["sitting"],
            "voting_num": row["voting_num"],
            "date": dt.date() if hasattr(dt, "date") else dt,
            "title": row["title"],
            "topic_id": int(topics[i]),
            "pair_agreement": float(pair_agreement[i]) if not np.isnan(pair_agreement[i]) else None,
            "term": term,
        })

    voting_topics_df = pl.DataFrame(voting_rows).with_columns([
        pl.col("sitting").cast(pl.Int16),
        pl.col("voting_num").cast(pl.Int16),
        pl.col("topic_id").cast(pl.Int32),
        pl.col("pair_agreement").cast(pl.Float32),
        pl.col("term").cast(pl.Int8),
    ])

    # topic_summary
    topic_info = topic_model.get_topic_info()
    summary_rows = []
    for _, trow in topic_info.iterrows():
        tid = int(trow["Topic"])
        mask = topics == tid
        n_votings = int(mask.sum())
        pa = pair_agreement[mask]
        pa_valid = pa[~np.isnan(pa)]

        words = topic_model.get_topic(tid)
        top_words = ", ".join([w for w, _ in words[:5]]) if words else ""

        summary_rows.append({
            "topic_id": tid,
            "top_words": top_words,
            "n_votings": n_votings,
            "mean_pair_agreement": float(np.mean(pa_valid)) if len(pa_valid) > 0 else None,
            "std_pair_agreement": float(np.std(pa_valid)) if len(pa_valid) > 0 else None,
            "is_outlier": tid == -1,
            "term": term,
        })

    topic_summary_df = pl.DataFrame(summary_rows).with_columns([
        pl.col("topic_id").cast(pl.Int32),
        pl.col("n_votings").cast(pl.Int32),
        pl.col("mean_pair_agreement").cast(pl.Float32),
        pl.col("std_pair_agreement").cast(pl.Float32),
        pl.col("term").cast(pl.Int8),
    ]).sort("mean_pair_agreement", descending=True)

    return voting_topics_df, topic_summary_df
