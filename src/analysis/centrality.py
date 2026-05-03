"""
Centrality analysis of rebel MPs.
Tests whether MPs who vote against their party are structural bridges in the network.
"""

import logging

import networkx as nx
import numpy as np
import polars as pl
from scipy.stats import spearmanr

from src.config import MIN_COPRESENCE

logger = logging.getLogger(__name__)


def compute_centrality_measures(
    agreement_frac: np.ndarray,
    copresence: np.ndarray,
    mp_ids: list[int],
    threshold: float = 0.50,
    min_copresence: int = MIN_COPRESENCE,
) -> pl.DataFrame:
    """
    Compute betweenness, pagerank, degree, eigenvector, and clustering for each MP.

    Returns a DataFrame with one row per MP.
    """
    from src.analysis.network import adjacency_to_networkx

    G = adjacency_to_networkx(agreement_frac, threshold, mp_ids, min_copresence, copresence)

    logger.info("Computing betweenness centrality (may take a moment) …")
    betweenness = nx.betweenness_centrality(G, normalized=True)

    logger.info("Computing PageRank …")
    pagerank = nx.pagerank(G, weight="weight")

    degree_cent = nx.degree_centrality(G)

    logger.info("Computing eigenvector centrality …")
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=500, weight="weight")
    except nx.PowerIterationFailedConvergence:
        logger.warning("Eigenvector centrality did not converge — setting to NaN")
        eigenvector = {n: float("nan") for n in G.nodes()}

    clustering = nx.clustering(G)

    rows = []
    for i, mp_id in enumerate(mp_ids):
        rows.append({
            "mp_id": mp_id,
            "betweenness": betweenness.get(i, 0.0),
            "pagerank": pagerank.get(i, 0.0),
            "degree_centrality": degree_cent.get(i, 0.0),
            "eigenvector": eigenvector.get(i, float("nan")),
            "clustering_coeff": clustering.get(i, 0.0),
            "threshold": threshold,
        })

    return pl.DataFrame(rows).with_columns([
        pl.col("mp_id").cast(pl.Int32),
        pl.col("betweenness").cast(pl.Float64),
        pl.col("pagerank").cast(pl.Float64),
        pl.col("degree_centrality").cast(pl.Float64),
        pl.col("eigenvector").cast(pl.Float64),
        pl.col("clustering_coeff").cast(pl.Float64),
        pl.col("threshold").cast(pl.Float32),
    ])


def join_with_rebels(
    centrality_df: pl.DataFrame,
    rebels_df: pl.DataFrame,
    mps_df: pl.DataFrame,
    term: int,
) -> pl.DataFrame:
    """Join centrality measures with rebel scores and MP names."""
    result = (
        centrality_df
        .join(
            mps_df.select(["mp_id", "first_name", "last_name", "club"]),
            on="mp_id",
            how="left",
        )
        .join(
            rebels_df.select(["mp_id", "rebel_rate", "rebel_count", "total_votes"]),
            on="mp_id",
            how="left",
        )
        .with_columns(pl.lit(term).cast(pl.Int8).alias("term"))
    )
    return result


def compute_spearman_correlations(
    joined_df: pl.DataFrame,
    term: int,
) -> pl.DataFrame:
    """
    Compute Spearman correlations between rebel_rate and each centrality measure.
    Only uses rows where rebel_rate is not null.
    """
    df = joined_df.filter(pl.col("rebel_rate").is_not_null())
    rebel_rates = df["rebel_rate"].to_numpy()

    centrality_vars = ["betweenness", "pagerank", "degree_centrality", "eigenvector", "clustering_coeff"]
    rows = []
    for var in centrality_vars:
        values = df[var].to_numpy()
        valid = ~np.isnan(rebel_rates) & ~np.isnan(values)
        if valid.sum() < 10:
            continue
        rho, pval = spearmanr(rebel_rates[valid], values[valid])
        rows.append({
            "x_var": "rebel_rate",
            "y_var": var,
            "spearman_rho": float(rho),
            "p_value": float(pval),
            "n_samples": int(valid.sum()),
            "threshold": float(df["threshold"][0]),
            "term": term,
        })

    return pl.DataFrame(rows).with_columns([
        pl.col("threshold").cast(pl.Float32),
        pl.col("term").cast(pl.Int8),
    ])
