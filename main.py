"""
Main pipeline orchestrator.

Runs all analysis steps in dependency order in-process (no subprocesses),
so progress bars and logging stream directly to the terminal.

Each step is skipped if its output files already exist. Use --force or --from
to re-run specific steps.

Usage:
    uv run python main.py
    uv run python main.py --term 10
    uv run python main.py --list
    uv run python main.py --force fetch
    uv run python main.py --force agreement --force communities
    uv run python main.py --from rebels
"""

import asyncio
import logging
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent))

from src.config import ANALYSIS_DIR, NETWORKS_DIR, PARQUET_DIR, RAW_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _exists(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


# ---------------------------------------------------------------------------
# Step runners — each calls the underlying function directly
# ---------------------------------------------------------------------------

def step_fetch(term: int) -> None:
    from src.fetch.pipeline import assemble_parquet, run_fetch
    asyncio.run(run_fetch(term=term, resume=True))
    assemble_parquet(term=term)


def step_agreement(term: int) -> None:
    import numpy as np
    from src.analysis.agreement import compute_agreement_matrix
    from src.data.store import build_vote_matrix, load_mps, load_votes
    NETWORKS_DIR.mkdir(parents=True, exist_ok=True)
    votes_df = load_votes(term)
    mps_df = load_mps(term)
    vote_matrix, presence_matrix, mp_ids, _ = build_vote_matrix(votes_df, mps_df)
    agreement_frac, copresence = compute_agreement_matrix(vote_matrix, presence_matrix)
    np.save(NETWORKS_DIR / "agreement_matrix.npy", agreement_frac)
    np.save(NETWORKS_DIR / "copresence_matrix.npy", copresence)
    np.save(NETWORKS_DIR / "mp_ids.npy", np.array(mp_ids, dtype=np.int32))
    logger.info("Saved agreement matrix (%d × %d)", *agreement_frac.shape)


def step_network(term: int) -> None:
    import numpy as np
    from src.analysis.network import compute_all_thresholds
    agreement_frac = np.load(NETWORKS_DIR / "agreement_matrix.npy")
    copresence = np.load(NETWORKS_DIR / "copresence_matrix.npy")
    mp_ids = np.load(NETWORKS_DIR / "mp_ids.npy").tolist()
    df = compute_all_thresholds(agreement_frac, copresence, mp_ids, term)
    df.write_parquet(NETWORKS_DIR / "network_metrics.parquet")
    logger.info("Saved network_metrics.parquet")
    print(df)


def step_communities(term: int) -> None:
    import numpy as np
    from src.analysis.communities import detect_communities
    from src.data.store import load_mps
    agreement_frac = np.load(NETWORKS_DIR / "agreement_matrix.npy")
    copresence = np.load(NETWORKS_DIR / "copresence_matrix.npy")
    mp_ids = np.load(NETWORKS_DIR / "mp_ids.npy").tolist()
    mps_df = load_mps(term)
    mp_id_to_club = dict(zip(mps_df["mp_id"].to_list(), mps_df["club"].to_list()))
    club_labels = [str(mp_id_to_club.get(mid, "UNKNOWN")) for mid in mp_ids]
    communities_df, metrics_df = detect_communities(agreement_frac, copresence, mp_ids, club_labels, term)
    for t in communities_df["threshold"].unique().sort().to_list():
        sub = communities_df.filter(communities_df["threshold"] == t)
        fname = f"communities_threshold_{t:.2f}.parquet".replace(".", "_")
        sub.write_parquet(NETWORKS_DIR / fname)
    metrics_df.write_parquet(NETWORKS_DIR / "community_metrics.parquet")
    print(metrics_df)


def step_cohesion(term: int) -> None:
    import numpy as np
    from src.analysis.cohesion import aggregate_monthly, compute_party_cohesion
    from src.data.store import build_vote_matrix, load_mps, load_votes, load_votings
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    votes_df = load_votes(term)
    votings_df = load_votings(term)
    mps_df = load_mps(term)
    vote_matrix, presence_matrix, mp_ids, voting_keys = build_vote_matrix(votes_df, mps_df)
    df = compute_party_cohesion(vote_matrix, presence_matrix, mp_ids, voting_keys, votes_df, votings_df, term)
    df.write_parquet(ANALYSIS_DIR / "party_cohesion_by_sitting.parquet")
    aggregate_monthly(df).write_parquet(ANALYSIS_DIR / "party_cohesion_by_month.parquet")
    logger.info("Saved party cohesion (%d rows)", len(df))


def step_rebels(term: int) -> None:
    import numpy as np
    from src.analysis.rebels import compute_rebel_scores
    from src.data.store import build_vote_matrix, load_mps, load_votes
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    votes_df = load_votes(term)
    mps_df = load_mps(term)
    vote_matrix, presence_matrix, mp_ids, voting_keys = build_vote_matrix(votes_df, mps_df)
    df = compute_rebel_scores(vote_matrix, presence_matrix, mp_ids, voting_keys, votes_df, term)
    df = df.join(mps_df.select(["mp_id", "first_name", "last_name"]), on="mp_id", how="left")
    df.write_parquet(ANALYSIS_DIR / "rebels.parquet")
    logger.info("Saved rebels.parquet (%d MPs)", len(df))
    print(df.select(["first_name", "last_name", "club", "rebel_rate"]).head(10))


def step_party_matrix(term: int) -> None:
    from src.analysis.party_matrix import compute_party_agreement_matrix
    from src.data.store import build_vote_matrix, load_mps, load_votes
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    votes_df = load_votes(term)
    mps_df = load_mps(term)
    vote_matrix, presence_matrix, mp_ids, _ = build_vote_matrix(votes_df, mps_df)
    mp_club = {k: str(v) for k, v in zip(mps_df["mp_id"].to_list(), mps_df["club"].to_list())}
    df = compute_party_agreement_matrix(vote_matrix, presence_matrix, mp_ids, mp_club, term)
    df.write_parquet(ANALYSIS_DIR / "party_correlation_matrix.parquet")
    logger.info("Saved party_correlation_matrix.parquet")


def step_traitors(term: int) -> None:
    import orjson
    from src.analysis.traitors import build_switch_summary, detect_club_switches
    from src.data.store import load_mps, load_votes
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    votes_df = load_votes(term)
    mps_df = load_mps(term)
    clubs_path = RAW_DIR / f"term{term}" / "clubs.json"
    clubs_raw = orjson.loads(clubs_path.read_bytes()) if clubs_path.exists() else []
    df = detect_club_switches(votes_df, mps_df, clubs_raw, term)
    df.write_parquet(ANALYSIS_DIR / "party_switchers.parquet")
    summary = build_switch_summary(df, term)
    if not summary.is_empty():
        summary.write_parquet(ANALYSIS_DIR / "switch_summary.parquet")
    logger.info("Detected %d switches", len(df))
    if not df.is_empty():
        print(df.select(["first_name", "last_name", "from_club", "to_club", "switch_date"]))


def step_topics(term: int) -> None:
    from src.analysis.topics import (
        build_topic_outputs, get_or_compute_embeddings,
        prepare_texts, run_bertopic,
    )
    from src.data.store import load_votings
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    votings_df = load_votings(term)
    texts = prepare_texts(votings_df)
    embeddings = get_or_compute_embeddings(texts)
    texts_for_model = [t if len(t) >= 10 else "" for t in texts]
    topic_model, topics = run_bertopic(texts_for_model, embeddings)
    vt_df, ts_df = build_topic_outputs(votings_df, topics, topic_model, term)
    vt_df.write_parquet(ANALYSIS_DIR / "voting_topics.parquet")
    ts_df.write_parquet(ANALYSIS_DIR / "topic_summary.parquet")
    logger.info("Saved topic outputs (%d topics)", len(ts_df))


def step_temporal(term: int) -> None:
    import polars as pl
    from src.analysis.temporal import compute_temporal_metrics
    from src.data.store import build_vote_matrix, load_mps, load_votes, load_votings
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    votes_df = load_votes(term)
    votings_df = load_votings(term)
    mps_df = load_mps(term)
    vote_matrix, presence_matrix, mp_ids, voting_keys = build_vote_matrix(votes_df, mps_df)
    mp_id_to_club = dict(zip(mps_df["mp_id"].to_list(), mps_df["club"].to_list()))
    club_labels = [str(mp_id_to_club.get(mid, "UNKNOWN")) for mid in mp_ids]
    df = compute_temporal_metrics(vote_matrix, presence_matrix, voting_keys, mp_ids, club_labels, votings_df, term)
    if not df.is_empty():
        df.write_parquet(ANALYSIS_DIR / "temporal_metrics.parquet")
        logger.info("Saved temporal_metrics.parquet (%d rows)", len(df))


def step_centrality(term: int) -> None:
    import numpy as np
    import polars as pl
    from src.analysis.centrality import (
        compute_centrality_measures, compute_spearman_correlations, join_with_rebels,
    )
    from src.data.store import load_mps
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    agreement_frac = np.load(NETWORKS_DIR / "agreement_matrix.npy")
    copresence = np.load(NETWORKS_DIR / "copresence_matrix.npy")
    mp_ids = np.load(NETWORKS_DIR / "mp_ids.npy").tolist()
    mps_df = load_mps(term)
    rebels_df = pl.read_parquet(ANALYSIS_DIR / "rebels.parquet")
    cent_df = compute_centrality_measures(agreement_frac, copresence, mp_ids)
    joined = join_with_rebels(cent_df, rebels_df, mps_df, term)
    corr = compute_spearman_correlations(joined, term)
    joined.write_parquet(ANALYSIS_DIR / "centrality_rebels.parquet")
    corr.write_parquet(ANALYSIS_DIR / "centrality_correlations.parquet")
    print(corr)


def step_blocs(term: int) -> None:
    import numpy as np
    import polars as pl
    from src.analysis.blocs import compute_topic_blocs
    from src.data.store import build_vote_matrix, load_mps, load_votes
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    votes_df = load_votes(term)
    mps_df = load_mps(term)
    vote_matrix, presence_matrix, mp_ids, _ = build_vote_matrix(votes_df, mps_df)
    mp_id_to_club = dict(zip(mps_df["mp_id"].to_list(), mps_df["club"].to_list()))
    club_labels = [str(mp_id_to_club.get(mid, "UNKNOWN")) for mid in mp_ids]
    voting_topics_df = pl.read_parquet(ANALYSIS_DIR / "voting_topics.parquet")
    topic_summary_df = pl.read_parquet(ANALYSIS_DIR / "topic_summary.parquet")
    blocs, affinity, summary = compute_topic_blocs(
        vote_matrix, presence_matrix, mp_ids, club_labels, voting_topics_df, topic_summary_df, term
    )
    if not blocs.is_empty():    blocs.write_parquet(ANALYSIS_DIR / "voting_blocs.parquet")
    if not affinity.is_empty(): affinity.write_parquet(ANALYSIS_DIR / "bloc_affinity.parquet")
    if not summary.is_empty():  summary.write_parquet(ANALYSIS_DIR / "bloc_summary.parquet")
    logger.info("Saved voting blocs")


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

STEPS = [
    {
        "name": "fetch",
        "label": "01 — Fetch raw data + assemble parquet",
        "outputs": lambda t: [
            PARQUET_DIR / f"term{t}" / "votes.parquet",
            PARQUET_DIR / f"term{t}" / "votings.parquet",
            PARQUET_DIR / f"term{t}" / "mps.parquet",
        ],
        "fn": step_fetch,
    },
    {
        "name": "agreement",
        "label": "02 — Build agreement matrix",
        "outputs": lambda t: [
            NETWORKS_DIR / "agreement_matrix.npy",
            NETWORKS_DIR / "copresence_matrix.npy",
            NETWORKS_DIR / "mp_ids.npy",
        ],
        "fn": step_agreement,
    },
    {
        "name": "network",
        "label": "03 — Network metrics",
        "outputs": lambda t: [NETWORKS_DIR / "network_metrics.parquet"],
        "fn": step_network,
    },
    {
        "name": "communities",
        "label": "04 — Community detection (Leiden)",
        "outputs": lambda t: [NETWORKS_DIR / "community_metrics.parquet"],
        "fn": step_communities,
    },
    {
        "name": "cohesion",
        "label": "05 — Party cohesion over time",
        "outputs": lambda t: [ANALYSIS_DIR / "party_cohesion_by_sitting.parquet"],
        "fn": step_cohesion,
    },
    {
        "name": "rebels",
        "label": "06 — Rebel MPs",
        "outputs": lambda t: [ANALYSIS_DIR / "rebels.parquet"],
        "fn": step_rebels,
    },
    {
        "name": "party_matrix",
        "label": "07 — Inter-party agreement matrix",
        "outputs": lambda t: [ANALYSIS_DIR / "party_correlation_matrix.parquet"],
        "fn": step_party_matrix,
    },
    {
        "name": "traitors",
        "label": "08 — Party switchers",
        "outputs": lambda t: [ANALYSIS_DIR / "party_switchers.parquet"],
        "fn": step_traitors,
    },
    {
        "name": "topics",
        "label": "09 — BERTopic topic modelling",
        "outputs": lambda t: [
            ANALYSIS_DIR / "voting_topics.parquet",
            ANALYSIS_DIR / "topic_summary.parquet",
        ],
        "fn": step_topics,
    },
    {
        "name": "temporal",
        "label": "10 — Temporal network evolution",
        "outputs": lambda t: [ANALYSIS_DIR / "temporal_metrics.parquet"],
        "fn": step_temporal,
    },
    {
        "name": "centrality",
        "label": "11 — Centrality vs rebel scores",
        "outputs": lambda t: [ANALYSIS_DIR / "centrality_rebels.parquet"],
        "fn": step_centrality,
    },
    {
        "name": "blocs",
        "label": "12 — Cross-party voting blocs",
        "outputs": lambda t: [ANALYSIS_DIR / "voting_blocs.parquet"],
        "fn": step_blocs,
    },
]

STEP_NAMES = [s["name"] for s in STEPS]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--term", "-t", type=int, default=10, show_default=True)
@click.option(
    "--force", "-f", multiple=True,
    type=click.Choice(STEP_NAMES),
    help="Force re-run of a step even if outputs exist. Repeatable.",
)
@click.option(
    "--from", "from_step",
    type=click.Choice(STEP_NAMES), default=None,
    help="Re-run from this step onwards.",
)
@click.option("--list", "list_steps", is_flag=True, help="List all steps and exit.")
def main(term: int, force: tuple[str, ...], from_step: str | None, list_steps: bool) -> None:
    """Run the full Sejm analysis pipeline, skipping already-completed steps."""
    if list_steps:
        for step in STEPS:
            done = _exists(*step["outputs"](term))
            status = "done" if done else "pending"
            click.echo(f"  [{status:>7}] {step['name']:15} — {step['label']}")
        return

    forced: set[str] = set(force)
    if from_step:
        idx = STEP_NAMES.index(from_step)
        forced |= set(STEP_NAMES[idx:])

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    NETWORKS_DIR.mkdir(parents=True, exist_ok=True)

    for step in STEPS:
        name = step["name"]
        done = _exists(*step["outputs"](term))

        if done and name not in forced:
            click.echo(f"  [skip]  {step['label']}")
            continue

        click.echo(f"\n  ── {step['label']}")
        try:
            step["fn"](term)
        except Exception as exc:
            logger.exception("Step '%s' failed: %s", name, exc)
            click.echo(f"\nPipeline stopped at '{name}'.", err=True)
            sys.exit(1)

    click.echo("\nAll steps complete.")


if __name__ == "__main__":
    main()
