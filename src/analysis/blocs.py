"""
Cross-party voting bloc detection.
For each BERTopic topic, builds a sub-agreement network and runs Leiden to find
groups that cross official party lines.
"""

import logging

import numpy as np
import polars as pl

from src.config import (
    BLOC_THRESHOLD,
    CROSS_PARTY_MAX_FRACTION,
    LEIDEN_SEED,
    MIN_COPRESENCE_TOPIC,
    MIN_TOPIC_VOTINGS,
)

logger = logging.getLogger(__name__)


def compute_topic_blocs(
    vote_matrix: np.ndarray,
    presence_matrix: np.ndarray,
    mp_ids: list[int],
    club_labels: list[str],
    voting_topics_df: pl.DataFrame,
    topic_summary_df: pl.DataFrame,
    term: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Run bloc detection for each BERTopic topic.

    Returns:
        voting_blocs_df:  one row per (topic, community, MP)
        bloc_affinity_df: club × club affinity matrix (long format)
        bloc_summary_df:  one row per topic with cross-party summary
    """
    from src.analysis.agreement import compute_agreement_matrix
    from src.analysis.communities import build_igraph, run_leiden

    # voting_topics_df has columns: voting_key, topic_id, sitting, voting_num
    # Build voting_key -> row index in vote_matrix
    vkey_list = voting_topics_df["voting_key"].to_list()
    vkey_to_idx: dict[str, int] = {}
    for i, vkey in enumerate(vkey_list):
        vkey_to_idx[vkey] = i

    unique_topics = (
        topic_summary_df
        .filter(
            (pl.col("topic_id") != -1) &
            (pl.col("n_votings") >= MIN_TOPIC_VOTINGS)
        )
        ["topic_id"]
        .to_list()
    )

    # Build topic_id -> top_words for labels
    topic_labels: dict[int, str] = {
        row["topic_id"]: row["top_words"]
        for row in topic_summary_df.iter_rows(named=True)
    }
    topic_agreement: dict[int, float] = {
        row["topic_id"]: row.get("mean_pair_agreement") or float("nan")
        for row in topic_summary_df.iter_rows(named=True)
    }

    clubs = sorted(set(club_labels))
    club_idx = {c: i for i, c in enumerate(clubs)}
    N = len(mp_ids)
    mp_club = {mp_id: club for mp_id, club in zip(mp_ids, club_labels)}

    # Track club co-memberships for affinity
    # affinity_counts[c1][c2] = {topics_same: int, topics_both: int}
    from collections import defaultdict
    co_topic: dict[tuple[str, str], int] = defaultdict(int)
    both_topic: dict[tuple[str, str], int] = defaultdict(int)

    all_bloc_rows: list[dict] = []
    all_summary_rows: list[dict] = []

    for topic_id in unique_topics:
        # Get voting row indices for this topic
        topic_vkeys = (
            voting_topics_df
            .filter(pl.col("topic_id") == topic_id)
            ["voting_key"]
            .to_list()
        )
        row_indices = [vkey_to_idx[k] for k in topic_vkeys if k in vkey_to_idx]
        if len(row_indices) < MIN_TOPIC_VOTINGS:
            continue

        idx_arr = np.array(row_indices)
        vm_sub = vote_matrix[idx_arr]
        pm_sub = presence_matrix[idx_arr]

        # Build sub-agreement matrix
        agreement_sub, copresence_sub = compute_agreement_matrix(
            vm_sub, pm_sub, min_copresence=MIN_COPRESENCE_TOPIC
        )

        # Build igraph and run Leiden
        try:
            g = build_igraph(agreement_sub, BLOC_THRESHOLD, mp_ids, copresence_sub, MIN_COPRESENCE_TOPIC)
            if g.ecount() == 0:
                continue
            partition = run_leiden(g)
            membership = partition.membership
        except Exception as exc:
            logger.debug("Bloc detection failed for topic %d: %s", topic_id, exc)
            continue

        n_communities = len(set(membership))
        topic_label = topic_labels.get(topic_id, "")
        mean_agree = topic_agreement.get(topic_id, float("nan"))

        n_cross_party = 0
        dominant_bloc_clubs = ""
        max_cross_score = 0.0

        # Analyse each community
        community_clubs: dict[int, list[str]] = {}
        for i, comm_id in enumerate(membership):
            community_clubs.setdefault(comm_id, []).append(club_labels[i])

        for comm_id, comm_club_list in community_clubs.items():
            comm_size = len(comm_club_list)
            if comm_size < 2:
                continue

            # Club fractions in this community
            club_counts: dict[str, int] = {}
            for c in comm_club_list:
                club_counts[c] = club_counts.get(c, 0) + 1
            max_frac = max(cnt / comm_size for cnt in club_counts.values())
            cross_party_score = 1.0 - max_frac
            is_cross_party = max_frac < CROSS_PARTY_MAX_FRACTION

            if is_cross_party:
                n_cross_party += 1

            if cross_party_score > max_cross_score:
                max_cross_score = cross_party_score
                dominant_clubs = sorted(club_counts.keys(), key=lambda c: -club_counts[c])
                dominant_bloc_clubs = ", ".join(dominant_clubs[:3])

            for i, mp_id in enumerate(mp_ids):
                if membership[i] != comm_id:
                    continue
                all_bloc_rows.append({
                    "topic_id": topic_id,
                    "topic_label": topic_label,
                    "community_id": comm_id,
                    "mp_id": mp_id,
                    "club": mp_club.get(mp_id, ""),
                    "community_size": comm_size,
                    "cross_party_score": cross_party_score,
                    "is_cross_party": is_cross_party,
                    "term": term,
                })

        # Update co-topic affinity
        present_clubs = set(club_labels[i] for i in range(N) if not np.isnan(agreement_sub[i, :]).all())
        for c1 in present_clubs:
            for c2 in present_clubs:
                if c1 >= c2:
                    continue
                both_topic[(c1, c2)] += 1

        for comm_id, comm_club_list in community_clubs.items():
            comm_club_set = set(comm_club_list)
            for c1 in comm_club_set:
                for c2 in comm_club_set:
                    if c1 >= c2:
                        co_topic[(c1, c2)] += 1

        all_summary_rows.append({
            "topic_id": topic_id,
            "topic_label": topic_label,
            "mean_pair_agreement": mean_agree,
            "n_cross_party_communities": n_cross_party,
            "dominant_bloc_clubs": dominant_bloc_clubs,
            "term": term,
        })

    # Build affinity DataFrame
    affinity_rows = []
    for (c1, c2), n_both in both_topic.items():
        n_same = co_topic.get((c1, c2), 0)
        affinity_rows.append({
            "club1": c1,
            "club2": c2,
            "bloc_affinity": n_same / n_both if n_both > 0 else float("nan"),
            "n_topics_same_community": n_same,
            "n_topics_both_present": n_both,
            "term": term,
        })

    def _safe_df(rows, name):
        if not rows:
            logger.warning("No %s rows generated", name)
            return pl.DataFrame()
        return pl.DataFrame(rows)

    blocs_df = _safe_df(all_bloc_rows, "blocs").with_columns([
        pl.col("topic_id").cast(pl.Int32),
        pl.col("community_id").cast(pl.Int32),
        pl.col("mp_id").cast(pl.Int32),
        pl.col("club").cast(pl.Categorical),
        pl.col("community_size").cast(pl.Int32),
        pl.col("cross_party_score").cast(pl.Float32),
        pl.col("term").cast(pl.Int8),
    ]) if all_bloc_rows else pl.DataFrame()

    affinity_df = _safe_df(affinity_rows, "affinity").with_columns([
        pl.col("club1").cast(pl.Categorical),
        pl.col("club2").cast(pl.Categorical),
        pl.col("bloc_affinity").cast(pl.Float32),
        pl.col("n_topics_same_community").cast(pl.Int32),
        pl.col("n_topics_both_present").cast(pl.Int32),
        pl.col("term").cast(pl.Int8),
    ]) if affinity_rows else pl.DataFrame()

    summary_df = _safe_df(all_summary_rows, "bloc_summary").with_columns([
        pl.col("topic_id").cast(pl.Int32),
        pl.col("mean_pair_agreement").cast(pl.Float32),
        pl.col("n_cross_party_communities").cast(pl.Int32),
        pl.col("term").cast(pl.Int8),
    ]).sort("n_cross_party_communities", descending=True) if all_summary_rows else pl.DataFrame()

    return blocs_df, affinity_df, summary_df
