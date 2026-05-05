"""
Script 19 — Topics × clubs analysis.

Analyses:
  1. Per-topic mean agreement within each club (heatmap)
  2. Most divisive topics within clubs (high intra-club disagreement)
  3. Topics that unite / divide coalition vs opposition

Figures:
  fig21_topic_clubs_heatmap.png
  fig22_topic_divisiveness.png
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import ANALYSIS_DIR, PARQUET_DIR, PROJECT_ROOT
from src.data.store import load_mps
from src.scripts.poster_style import apply_style, CLUB_COLOURS, cc, MAIN_CLUBS, PALETTE, COALITION, OPPOSITION, club_en

apply_style()

FIG_DIR = PROJECT_ROOT / "data" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def shorten_topic(words: str, n: int = 4) -> str:
    parts = [w.strip().rstrip(",") for w in words.split(",")][:n]
    return ", ".join(parts)


def build_topic_club_matrix(term: int = 10) -> tuple[pl.DataFrame, np.ndarray, list, list]:
    """
    For each topic × club, compute the mean agreement (Rice-like index).
    Per voting within a topic: club_agreement = 1 - |frac_yes - frac_no|
    Then average over all votings in the topic.
    """
    votes   = pl.read_parquet(PARQUET_DIR / f"term{term}/votes.parquet")
    mps     = load_mps(term)
    topics_df  = pl.read_parquet(ANALYSIS_DIR / "voting_topics.parquet")
    topic_sum  = (
        pl.read_parquet(ANALYSIS_DIR / "topic_summary.parquet")
        .filter(~pl.col("is_outlier"))
        .filter(pl.col("n_votings") >= 8)
    )

    # Join votes with topic
    votes_with_topic = (
        votes
        .join(topics_df.select(["voting_key", "topic_id"]), on="voting_key", how="inner")
        .join(
            topic_sum.select(["topic_id", "top_words"]),
            on="topic_id", how="inner"
        )
        .with_columns(pl.col("club").cast(pl.Utf8).alias("club_str"))
        .filter(pl.col("club_str").is_in(MAIN_CLUBS))
        .filter(pl.col("vote").cast(pl.Utf8).is_in(["YES", "NO", "ABSTAIN"]))
        .with_columns(
            pl.when(pl.col("vote").cast(pl.Utf8) == "YES").then(pl.lit(1.0))
            .when(pl.col("vote").cast(pl.Utf8) == "NO").then(pl.lit(-1.0))
            .otherwise(pl.lit(0.0)).alias("vote_val")
        )
    )

    # Per voting × club: fraction YES and NO
    per_voting_club = (
        votes_with_topic
        .group_by(["topic_id", "top_words", "voting_key", "club_str"])
        .agg([
            (pl.col("vote_val") == 1.0).sum().alias("n_yes"),
            (pl.col("vote_val") == -1.0).sum().alias("n_no"),
            pl.len().alias("n_total"),
        ])
        .filter(pl.col("n_total") >= 3)
        .with_columns(
            (1 - (pl.col("n_yes") - pl.col("n_no")).abs() / pl.col("n_total")).alias("cohesion")
        )
    )

    # Average over votings per topic × club
    topic_club_avg = (
        per_voting_club
        .group_by(["topic_id", "top_words", "club_str"])
        .agg([
            pl.col("cohesion").mean().alias("mean_cohesion"),
            pl.col("voting_key").n_unique().alias("n_votings"),
        ])
        .filter(pl.col("n_votings") >= 3)
        .sort(["topic_id", "club_str"])
    )

    # Build matrix: topics × clubs
    topics_list = (
        topic_club_avg
        .group_by(["topic_id", "top_words"])
        .agg(pl.col("n_votings").sum().alias("total_votings"))
        .sort("total_votings", descending=True)
        .head(25)
    )
    topic_ids   = topics_list["topic_id"].to_list()
    topic_words = [shorten_topic(w) for w in topics_list["top_words"].to_list()]

    mat = np.full((len(topic_ids), len(MAIN_CLUBS)), np.nan)
    id_to_col = {c: i for i, c in enumerate(MAIN_CLUBS)}

    for row in topic_club_avg.filter(pl.col("topic_id").is_in(topic_ids)).iter_rows(named=True):
        ti = topic_ids.index(row["topic_id"]) if row["topic_id"] in topic_ids else -1
        ci = id_to_col.get(row["club_str"], -1)
        if ti >= 0 and ci >= 0:
            mat[ti, ci] = row["mean_cohesion"]

    return topic_club_avg, mat, topic_words, MAIN_CLUBS


def build_coalition_topic_split(term: int = 10) -> pl.DataFrame:
    """
    Per topic: mean agreement within coalition, within opposition, and between blocs.
    Divisiveness = between_blocs low AND within blocs high.
    """
    votes   = pl.read_parquet(PARQUET_DIR / f"term{term}/votes.parquet")
    mps     = load_mps(term)
    topics_df  = pl.read_parquet(ANALYSIS_DIR / "voting_topics.parquet")
    topic_sum  = (
        pl.read_parquet(ANALYSIS_DIR / "topic_summary.parquet")
        .filter(~pl.col("is_outlier"))
        .filter(pl.col("n_votings") >= 8)
    )

    mps = mps.with_columns(
        pl.col("club").cast(pl.Utf8).map_elements(
            lambda c: "coalition" if c in COALITION else ("opposition" if c in OPPOSITION else None),
            return_dtype=pl.Utf8,
        ).alias("bloc")
    )
    bloc_map = dict(zip(mps["mp_id"].to_list(), mps["bloc"].to_list()))

    votes_t = (
        votes
        .join(topics_df.select(["voting_key", "topic_id"]), on="voting_key", how="inner")
        .join(topic_sum.select(["topic_id", "top_words"]), on="topic_id", how="inner")
        .with_columns([
            pl.col("mp_id").map_elements(lambda x: bloc_map.get(x), return_dtype=pl.Utf8).alias("bloc"),
            pl.col("vote").cast(pl.Utf8).alias("vote_str"),
        ])
        .filter(pl.col("bloc").is_not_null())
        .filter(pl.col("vote_str").is_in(["YES", "NO", "ABSTAIN"]))
        .with_columns(
            pl.when(pl.col("vote_str") == "YES").then(pl.lit(1.0))
            .when(pl.col("vote_str") == "NO").then(pl.lit(-1.0))
            .otherwise(pl.lit(0.0)).alias("vote_val")
        )
    )

    # Per voting × bloc: mean vote
    vb = (
        votes_t
        .group_by(["topic_id", "top_words", "voting_key", "bloc"])
        .agg([
            pl.col("vote_val").mean().alias("mean_vote"),
            pl.len().alias("n"),
        ])
        .filter(pl.col("n") >= 5)
    )

    wide = (
        vb.pivot(on="bloc", index=["topic_id", "top_words", "voting_key"],
                 values="mean_vote", aggregate_function="mean")
        .drop_nulls()
    )
    if "coalition" not in wide.columns or "opposition" not in wide.columns:
        return pl.DataFrame()

    wide = wide.with_columns(
        (1 - (pl.col("coalition") - pl.col("opposition")).abs() / 2).alias("agreement_coal_oppo")
    )

    per_topic = (
        wide.group_by(["topic_id", "top_words"])
        .agg([
            pl.col("agreement_coal_oppo").mean().alias("mean_coal_oppo_agree"),
            pl.col("voting_key").n_unique().alias("n_votings"),
        ])
        .filter(pl.col("n_votings") >= 5)
        .sort("mean_coal_oppo_agree")
    )
    return per_topic


# ── figures ───────────────────────────────────────────────────────────────────

def fig21_topic_clubs_heatmap(mat: np.ndarray, topic_words: list, clubs: list) -> None:
    print("  fig21: topic × club heatmap …")
    cmap = LinearSegmentedColormap.from_list(
        "rc", ["#FFE8E8", "#FF9999", PALETTE["accent"], PALETTE["accent"], "#8B0000", "#3d0000"]
    )

    fig, ax = plt.subplots(figsize=(12, max(8, len(topic_words) * 0.38 + 1.5)), facecolor="white")
    ax.set_facecolor("white")

    im = ax.imshow(mat, cmap=cmap, vmin=0.3, vmax=1.0, aspect="auto")

    # Club colour header strip
    for j, club in enumerate(clubs):
        ax.add_patch(mpl.patches.Rectangle(
            (j - 0.5, -1.5), 1, 1,
            facecolor=CLUB_COLOURS.get(club, PALETTE["neutral"]), edgecolor="white", linewidth=1,
            clip_on=False, transform=ax.transData,
        ))

    ax.set_xticks(range(len(clubs)))
    ax.set_xticklabels(clubs, rotation=35, ha="right", fontsize=9.5)
    ax.set_yticks(range(len(topic_words)))
    ax.set_yticklabels(topic_words, fontsize=8.5)

    for i in range(len(topic_words)):
        for j in range(len(clubs)):
            v = mat[i, j]
            if not np.isnan(v):
                col = "white" if v > 0.65 else PALETTE["dark"]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=col)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Intra-club voting cohesion", fontsize=9)

    ax.set_title(
        "Voting cohesion by topic and club  |  Term X\n"
        "(Rice index: 1 = full unity, 0 = 50/50 split)",
        fontsize=12, pad=14,
    )
    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig21_topic_clubs_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out}")


def fig22_topic_divisiveness(split_df: pl.DataFrame) -> None:
    print("  fig22: topic divisiveness coalition vs opposition …")
    if split_df.is_empty():
        print("    no data")
        return

    fig = plt.figure(figsize=(14, 7), facecolor="white")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.45)

    cmap = LinearSegmentedColormap.from_list(
        "rc2", ["#FFE8E8", "#FF9999", PALETTE["accent"], PALETTE["accent"], "#8B0000"]
    )
    norm = mpl.colors.Normalize(vmin=split_df["mean_coal_oppo_agree"].min(),
                                 vmax=split_df["mean_coal_oppo_agree"].max())

    # Left: most divisive (coalition vs opposition)
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("white")
    div = split_df.sort("mean_coal_oppo_agree").head(15)
    labels = [shorten_topic(r["top_words"]) for r in div.iter_rows(named=True)]
    vals   = div["mean_coal_oppo_agree"].to_numpy()
    n_v    = div["n_votings"].to_list()
    cols   = [cmap(norm(v)) for v in vals]
    y      = np.arange(len(labels))
    bars   = ax1.barh(y, vals, color=cols, alpha=0.88, edgecolor="#222", linewidth=0.3, height=0.65)
    for bar, v, nv in zip(bars, vals, n_v):
        ax1.text(v + 0.008, bar.get_y() + bar.get_height()/2,
                 f"{v:.2f}  (n={nv})", va="center", fontsize=8.5, color=PALETTE["dark"])
    ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel("Coalition–opposition agreement", fontsize=10)
    ax1.set_title("Most polarising topics", fontsize=12)
    ax1.set_xlim(0, 1.2); ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.25); ax1.spines[["top", "right"]].set_visible(False)

    # Right: most uniting
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("white")
    uni = split_df.sort("mean_coal_oppo_agree", descending=True).head(15)
    labels2 = [shorten_topic(r["top_words"]) for r in uni.iter_rows(named=True)]
    vals2   = uni["mean_coal_oppo_agree"].to_numpy()
    n_v2    = uni["n_votings"].to_list()
    cols2   = [cmap(norm(v)) for v in vals2]
    y2      = np.arange(len(labels2))
    bars2   = ax2.barh(y2, vals2, color=cols2, alpha=0.88, edgecolor="#222", linewidth=0.3, height=0.65)
    for bar, v, nv in zip(bars2, vals2, n_v2):
        ax2.text(v + 0.008, bar.get_y() + bar.get_height()/2,
                 f"{v:.2f}  (n={nv})", va="center", fontsize=8.5, color=PALETTE["dark"])
    ax2.set_yticks(y2); ax2.set_yticklabels(labels2, fontsize=9)
    ax2.set_xlabel("Coalition–opposition agreement", fontsize=10)
    ax2.set_title("Topics uniting the Sejm", fontsize=12)
    ax2.set_xlim(0, 1.2); ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.25); ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Which topics divide vs. unite coalition and opposition?  |  Term X",
                 fontsize=14, color=PALETTE["dark"], y=1.01)
    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig22_topic_divisiveness.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out}")


if __name__ == "__main__":
    print("Script 19 — Topics × clubs analysis …")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    topic_club_df, mat, topic_words, clubs = build_topic_club_matrix()
    topic_club_df.write_parquet(ANALYSIS_DIR / "topic_club_cohesion.parquet")

    split_df = build_coalition_topic_split()
    if not split_df.is_empty():
        split_df.write_parquet(ANALYSIS_DIR / "topic_coalition_split.parquet")

    fig21_topic_clubs_heatmap(mat, topic_words, clubs)
    fig22_topic_divisiveness(split_df)
    print(f"\nDone. Figures in {FIG_DIR}")
