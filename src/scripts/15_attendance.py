"""
Script 15 — Attendance analysis.

Analyses:
  1. Attendance rate per MP (% non-absent) by club — box plot
  2. Attendance over time per main club — monthly line chart
  3. Strategic absence — is absence rate higher on controversial votes?

Figures:
  fig11_attendance_by_club.png
  fig12_attendance_over_time.png
  fig13_strategic_absence.png
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import ANALYSIS_DIR, PARQUET_DIR, PROJECT_ROOT
from src.data.store import load_mps
from src.scripts.poster_style import apply_style, CLUB_COLOURS, cc, MAIN_CLUBS, PALETTE, COALITION, OPPOSITION, club_en

apply_style()

FIG_DIR = PROJECT_ROOT / "data" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def build_attendance(term: int = 10) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-MP attendance rates and monthly per-club attendance."""
    votes = pl.read_parquet(PARQUET_DIR / f"term{term}/votes.parquet")
    mps   = load_mps(term)

    votes = votes.with_columns(
        pl.col("vote").cast(pl.Utf8).alias("vote_str")
    )

    # Per MP: count absent vs total
    per_mp = (
        votes
        .group_by(["mp_id", "vote_str"])
        .agg(pl.len().alias("n"))
        .pivot(on="vote_str", index="mp_id", values="n", aggregate_function="sum")
        .fill_null(0)
    )
    # Ensure ABSENT column exists
    if "ABSENT" not in per_mp.columns:
        per_mp = per_mp.with_columns(pl.lit(0).alias("ABSENT"))
    vote_cols = [c for c in per_mp.columns if c != "mp_id"]
    per_mp = per_mp.with_columns(
        pl.sum_horizontal(*[pl.col(c) for c in vote_cols]).alias("total")
    ).with_columns(
        (1 - pl.col("ABSENT") / pl.col("total")).alias("attendance_rate")
    )
    per_mp = per_mp.join(mps.select(["mp_id", "first_name", "last_name", "club"]), on="mp_id", how="left")

    # Monthly per-club attendance
    monthly = (
        votes
        .with_columns(
            pl.col("date").dt.strftime("%Y-%m").alias("year_month"),
            (pl.col("vote").cast(pl.Utf8) == "ABSENT").alias("is_absent"),
        )
        .group_by(["year_month", "club"])
        .agg(
            pl.col("is_absent").sum().alias("n_absent"),
            pl.len().alias("total"),
        )
        .with_columns(
            (1 - pl.col("n_absent") / pl.col("total")).alias("attendance_rate")
        )
        .sort(["club", "year_month"])
    )
    return per_mp, monthly


def build_strategic_absence(term: int = 10) -> pl.DataFrame:
    """Compare absence rate on controversial vs non-controversial votes per club."""
    votes    = pl.read_parquet(PARQUET_DIR / f"term{term}/votes.parquet")
    votings  = pl.read_parquet(PARQUET_DIR / f"term{term}/votings.parquet")

    votings = votings.with_columns(
        (
            pl.min_horizontal(pl.col("yes_count").cast(pl.Float64),
                              pl.col("no_count").cast(pl.Float64))
            / pl.col("total_voted").cast(pl.Float64).clip(lower_bound=1)
        ).alias("controversy")
    ).select(["voting_key", "controversy"])

    votes = (
        votes
        .with_columns(pl.col("vote").cast(pl.Utf8).alias("vote_str"))
        .join(votings, on="voting_key", how="left")
        .with_columns([
            (pl.col("controversy") >= 0.30).alias("is_controversial"),
            (pl.col("vote_str") == "ABSENT").alias("is_absent"),
        ])
        .filter(pl.col("club").cast(pl.Utf8).is_in(MAIN_CLUBS))
    )

    result = (
        votes
        .group_by(["club", "is_controversial"])
        .agg(
            pl.col("is_absent").mean().alias("absence_rate"),
            pl.len().alias("n_votes"),
        )
        .sort(["club", "is_controversial"])
    )
    return result


# ── figures ───────────────────────────────────────────────────────────────────

def fig11_attendance_by_club(per_mp: pl.DataFrame) -> None:
    print("  fig11: attendance by club …")
    df = per_mp.filter(pl.col("club").cast(pl.Utf8).is_in(MAIN_CLUBS))
    clubs_ordered = (
        df.group_by("club")
        .agg(pl.col("attendance_rate").median().alias("med"))
        .sort("med", descending=True)
        ["club"].cast(pl.Utf8).to_list()
    )

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    ax.set_facecolor("white")

    for i, club in enumerate(clubs_ordered):
        rates = df.filter(pl.col("club").cast(pl.Utf8) == club)["attendance_rate"].to_numpy()
        col = cc(club)
        bp = ax.boxplot(rates, positions=[i], widths=0.55,
                        patch_artist=True, notch=False,
                        boxprops=dict(facecolor=col, alpha=0.8, linewidth=0.8),
                        medianprops=dict(color=PALETTE["dark"], linewidth=2),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        flierprops=dict(marker="o", markersize=3, alpha=0.4,
                                        markerfacecolor=col, markeredgewidth=0))

    ax.set_xticks(range(len(clubs_ordered)))
    ax.set_xticklabels(clubs_ordered, fontsize=10)
    ax.set_ylabel("Attendance rate", fontsize=10)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.set_ylim(0.5, 1.02)
    ax.set_title("MP attendance by club  |  Term X", fontsize=13, pad=12)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig11_attendance_by_club.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out}")


def fig12_attendance_over_time(monthly: pl.DataFrame) -> None:
    print("  fig12: attendance over time …")
    df = monthly.filter(pl.col("club").cast(pl.Utf8).is_in(MAIN_CLUBS))
    all_months = sorted(df["year_month"].unique().to_list())

    fig, ax = plt.subplots(figsize=(13, 6), facecolor="white")
    ax.set_facecolor("white")

    ls = ["-", "--", "-.", ":", "-", "--", "-."]
    for i, club in enumerate(MAIN_CLUBS):
        sub = df.filter(pl.col("club").cast(pl.Utf8) == club).sort("year_month")
        if len(sub) == 0:
            continue
        months = sub["year_month"].to_list()
        vals   = sub["attendance_rate"].to_list()
        x_pos  = [all_months.index(m) for m in months]
        ax.plot(x_pos, vals, color=cc(club), linewidth=2.0,
                label=club, marker="o", markersize=3.5, alpha=0.9,
                linestyle=ls[i % len(ls)])

    ax.set_xticks(range(0, len(all_months), 3))
    ax.set_xticklabels(all_months[::3], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Attendance rate", fontsize=10)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.set_ylim(0.55, 1.02)
    ax.set_title("Club attendance over time  |  Term X", fontsize=13, pad=12)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, ncol=2, framealpha=0.5)

    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig12_attendance_over_time.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out}")


def fig13_strategic_absence(strat: pl.DataFrame) -> None:
    print("  fig13: strategic absence …")
    pivot = (
        strat
        .with_columns(
            pl.when(pl.col("is_controversial")).then(pl.lit("controversial"))
            .otherwise(pl.lit("routine")).alias("type")
        )
        .pivot(on="type", index="club", values="absence_rate", aggregate_function="mean")
        .fill_null(0)
    )
    clubs     = pivot["club"].cast(pl.Utf8).to_list()
    routine   = pivot["routine"].to_list()
    controver = pivot["controversial"].to_list()
    diff      = [c - r for c, r in zip(controver, routine)]

    order = np.argsort(diff)[::-1]
    clubs_s    = [clubs[i] for i in order]
    routine_s  = [routine[i] for i in order]
    controver_s= [controver[i] for i in order]
    diff_s     = [diff[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")
    fig.suptitle("Strategic absence — do MPs skip controversial votes more often?\n"
                 "Controversial = min(yes, no) / total ≥ 30%",
                 fontsize=13, color=PALETTE["dark"], y=1.01)

    ax = axes[0]
    ax.set_facecolor("white")
    x = np.arange(len(clubs_s))
    w = 0.35
    ax.bar(x - w/2, routine_s,   w, label="Routine",       color="#BBBBBB", alpha=0.85, edgecolor="#333", linewidth=0.4)
    ax.bar(x + w/2, controver_s, w, label="Controversial", color=PALETTE["accent"], alpha=0.85, edgecolor="#333", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(clubs_s, fontsize=9.5, rotation=20, ha="right")
    ax.set_ylabel("Absence rate", fontsize=10)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.set_title("Absence by vote type", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.5)

    ax2 = axes[1]
    ax2.set_facecolor("white")
    bar_cols = [PALETTE["accent"] if d > 0 else "#AAAAAA" for d in diff_s]
    ax2.barh(range(len(clubs_s)), diff_s, color=bar_cols, alpha=0.85, edgecolor="#333", linewidth=0.4)
    for i, d in enumerate(diff_s):
        ax2.text(d + (0.001 if d >= 0 else -0.001), i,
                 f"{d:+.1%}", va="center", ha="left" if d >= 0 else "right", fontsize=9)
    ax2.set_yticks(range(len(clubs_s)))
    ax2.set_yticklabels(clubs_s, fontsize=9.5)
    ax2.axvline(0, color=PALETTE["dark"], linewidth=0.8)
    ax2.set_xlabel("Absence difference (contr. – routine)", fontsize=10)
    ax2.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax2.set_title("Extra absence on controversial votes", fontsize=11)
    ax2.grid(axis="x", alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig13_strategic_absence.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved {out}")


if __name__ == "__main__":
    print("Script 15 — Attendance analysis …")
    per_mp, monthly = build_attendance()
    per_mp.write_parquet(ANALYSIS_DIR / "attendance_per_mp.parquet")
    monthly.write_parquet(ANALYSIS_DIR / "attendance_monthly.parquet")
    strat = build_strategic_absence()
    strat.write_parquet(ANALYSIS_DIR / "strategic_absence.parquet")

    fig11_attendance_by_club(per_mp)
    fig12_attendance_over_time(monthly)
    fig13_strategic_absence(strat)
    print(f"\nDone. Figures in {FIG_DIR}")
