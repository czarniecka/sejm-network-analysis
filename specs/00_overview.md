# Spec 00 — Project Overview

## Goals

Analyze the voting behaviour of Polish Members of Parliament (MPs) using network science,
community detection, and NLP topic modelling. The project answers the following research
questions:

1. How does network cohesion (connectivity, clustering, components) change as the required
   pairwise voting-agreement threshold is varied?
2. Do Leiden-detected communities match official party lines, and how does this change with
   the threshold?
3. How does intra-party cohesion evolve over time (per sitting, per month)?
4. Which MPs most frequently vote against their party ("rebels")?
5. Which parties are most correlated with each other (inter-party agreement matrix)?
6. Which MPs switched parties ("traitors"), and when?
7. What topics do MPs vote on (BERTopic), and which topics produce the strongest consensus
   or the most controversy?
8. How does the voting network evolve across parliamentary terms (term 9 vs 10)?
9. Do rebel MPs occupy high-betweenness-centrality positions (bridge nodes between parties)?
10. Do cross-party voting blocs form on specific topics (e.g. PiS + Konfederacja)?

---

## Glossary

| Term | Definition |
|------|-----------|
| MP | Member of Parliament (poseł) |
| Club | Official political club/party fraction in the Sejm |
| Term | Parliamentary term (kadencja); currently term 10 (2023–) |
| Sitting | Plenary session (posiedzenie); identified by an integer |
| Voting | Single roll-call vote within a sitting; identified by (sitting, voting_num) |
| Agreement | Two MPs cast the same vote (YES/YES, NO/NO, or ABSTAIN/ABSTAIN) |
| Co-presence | Both MPs were present (not ABSENT) for a given voting |
| Agreement rate | agreements / co-presence count for an MP pair |
| Rebel | MP whose agreement rate with own-club majority is below average |
| Traitor | MP whose `club` field in vote records changes during a term |
| Voting bloc | A cross-party group of MPs who vote together on a topic sub-network |

---

## Directory Layout

```
sejm-network-analysis/
├── pyproject.toml          # uv-managed dependencies
├── specs/                  # this folder — one spec per analysis area
├── src/
│   ├── config.py           # all tunable constants
│   ├── fetch/              # async API client and pipeline
│   ├── data/               # parquet schemas and loading utilities
│   ├── analysis/           # analysis modules (one per spec)
│   └── scripts/            # entry-point scripts (01_fetch.py … 12_voting_blocs.py)
└── data/
    ├── raw/term{N}/        # raw JSON files: {sitting}_{num}.json
    ├── parquet/term{N}/    # mps.parquet, votings.parquet, votes.parquet
    ├── networks/           # agreement_matrix.npz, copresence_matrix.npz, metrics
    └── analysis/           # output parquet files for each analysis
```

---

## Technology Choices

| Concern | Choice | Reason |
|---------|--------|--------|
| HTTP client | `aiohttp` | Best throughput for many small async requests |
| JSON parsing | `orjson` | 3-5× faster than stdlib json |
| DataFrame | `polars` | Lazy evaluation, zero-copy parquet, fast groupby |
| Numeric arrays | `numpy` (float32) | BLAS matmul for agreement matrix; avoids Python loops |
| Network | `networkx` + `python-igraph` | networkx for metrics, igraph for Leiden |
| Community detection | `leidenalg` | Deterministic with fixed seed, higher quality than Louvain |
| Topic modelling | `BERTopic` | State-of-the-art; handles Polish via multilingual transformer |
| Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` | Polish support, ~117 MB, fast on Apple MPS |
| Package manager | `uv` | Fast, reproducible, supports `pyproject.toml` |

---

## Language

All source code, comments, docstrings, specs, and output file names are written in **English**.
Polish appears only in raw data strings (MP names, voting titles, etc.).

---

## Running Order

```
uv run python src/scripts/01_fetch.py --term 10
uv run python src/scripts/02_build_agreement.py --term 10
uv run python src/scripts/03_network_analysis.py --term 10
uv run python src/scripts/04_community_detection.py --term 10
uv run python src/scripts/05_party_cohesion.py --term 10
uv run python src/scripts/06_rebels.py --term 10
uv run python src/scripts/07_party_matrix.py --term 10
uv run python src/scripts/08_traitors.py --term 10
uv run python src/scripts/09_topic_modeling.py --term 10
uv run python src/scripts/10_temporal_network.py --terms 9 10
uv run python src/scripts/11_centrality_rebels.py --term 10
uv run python src/scripts/12_voting_blocs.py --term 10
```
