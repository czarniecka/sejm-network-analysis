"""
Networkx-based network metric computation for MP agreement networks.
"""

import logging

import networkx as nx
import numpy as np
import polars as pl

from src.config import AGREEMENT_THRESHOLDS, MIN_COPRESENCE

logger = logging.getLogger(__name__)


def adjacency_to_networkx(
    agreement_frac: np.ndarray,
    threshold: float,
    mp_ids: list[int],
    min_copresence: int = MIN_COPRESENCE,
    copresence: np.ndarray | None = None,
) -> nx.Graph:
    """Build a networkx Graph from the agreement matrix at a given threshold."""
    N = agreement_frac.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))

    for i in range(N):
        for j in range(i + 1, N):
            w = agreement_frac[i, j]
            if np.isnan(w):
                continue
            if copresence is not None and copresence[i, j] < min_copresence:
                continue
            if w >= threshold:
                G.add_edge(i, j, weight=float(w))

    # Store mp_id as node attribute
    nx.set_node_attributes(G, {i: mp_ids[i] for i in range(N)}, "mp_id")
    return G


def compute_network_metrics(
    agreement_frac: np.ndarray,
    copresence: np.ndarray,
    threshold: float,
    mp_ids: list[int],
    term: int,
) -> dict:
    """
    Compute all network metrics for one threshold level.

    Returns a dict matching the network_metrics.parquet schema.
    """
    G = adjacency_to_networkx(agreement_frac, threshold, mp_ids, copresence=copresence)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    components = list(nx.connected_components(G))
    n_components = len(components)
    largest = max(components, key=len) if components else set()
    lcc_size = len(largest)
    lcc_frac = lcc_size / n_nodes if n_nodes > 0 else 0.0
    avg_clustering = nx.average_clustering(G)

    # Diameter and avg path length on LCC only
    diameter_lcc: int | None = None
    avg_path_lcc: float | None = None
    if lcc_size > 1:
        lcc_subgraph = G.subgraph(largest)
        try:
            diameter_lcc = nx.diameter(lcc_subgraph)
            avg_path_lcc = nx.average_shortest_path_length(lcc_subgraph)
        except nx.NetworkXError:
            pass

    metrics = {
        "threshold": threshold,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": density,
        "n_components": n_components,
        "largest_component_size": lcc_size,
        "largest_component_frac": lcc_frac,
        "avg_clustering": avg_clustering,
        "diameter_lcc": diameter_lcc,
        "avg_path_length_lcc": avg_path_lcc,
        "term": term,
    }
    logger.info(
        "Threshold %.2f: %d edges, density=%.4f, components=%d, clustering=%.4f",
        threshold, n_edges, density, n_components, avg_clustering,
    )
    return metrics


def compute_all_thresholds(
    agreement_frac: np.ndarray,
    copresence: np.ndarray,
    mp_ids: list[int],
    term: int,
    thresholds: list[float] = AGREEMENT_THRESHOLDS,
) -> pl.DataFrame:
    """Run compute_network_metrics for all thresholds and return a DataFrame."""
    rows = [
        compute_network_metrics(agreement_frac, copresence, t, mp_ids, term)
        for t in thresholds
    ]
    return pl.DataFrame(rows).with_columns([
        pl.col("threshold").cast(pl.Float32),
        pl.col("term").cast(pl.Int8),
    ])
