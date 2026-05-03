"""
Script 09 — BERTopic topic modelling on voting titles.

Reads:  data/parquet/term{N}/votings.parquet
Writes: data/analysis/voting_embeddings.npy    (cached embeddings)
        data/analysis/voting_topics.parquet
        data/analysis/topic_summary.parquet

Usage:
    uv run python src/scripts/09_topic_modeling.py --term 10
"""

import logging
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.topics import (
    build_topic_outputs,
    get_or_compute_embeddings,
    prepare_texts,
    run_bertopic,
)
from src.config import ANALYSIS_DIR
from src.data.store import load_votings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--term", "-t", type=int, default=10)
def main(term: int) -> None:
    """Run BERTopic on voting titles and compute per-topic agreement metrics."""
    votings_df = load_votings(term)
    logger.info("Loaded %d votings", len(votings_df))

    texts = prepare_texts(votings_df)
    logger.info("Prepared %d texts (non-empty: %d)", len(texts), sum(1 for t in texts if t))

    embeddings = get_or_compute_embeddings(texts)

    non_empty_mask = [len(t) >= 10 for t in texts]
    texts_for_model = [t if ok else "" for t, ok in zip(texts, non_empty_mask)]

    topic_model, topics = run_bertopic(texts_for_model, embeddings)

    voting_topics_df, topic_summary_df = build_topic_outputs(votings_df, topics, topic_model, term)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    voting_topics_df.write_parquet(ANALYSIS_DIR / "voting_topics.parquet")
    topic_summary_df.write_parquet(ANALYSIS_DIR / "topic_summary.parquet")

    logger.info("Saved voting_topics.parquet (%d rows)", len(voting_topics_df))
    logger.info("Saved topic_summary.parquet (%d topics)", len(topic_summary_df))

    print("\nTop topics by consensus (mean pair agreement):")
    print(topic_summary_df.filter(~topic_summary_df["is_outlier"]).head(10))

    print("\nMost controversial topics (lowest agreement):")
    print(
        topic_summary_df
        .filter(~topic_summary_df["is_outlier"])
        .sort("mean_pair_agreement")
        .head(10)
    )


if __name__ == "__main__":
    main()
