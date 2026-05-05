"""
Script 18 — Polarization evolution + wild card votes.

Analyses:
  1. Monthly polarization index: within-bloc agreement vs between-bloc agreement
  2. Wild card votes: votings where coalition lost (expected to win but didn't)

Figures:
  fig19_polarization.png
  fig20_wild_cards.png
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import ANALYSIS_DIR, PARQUET_DIR, PROJECT_ROOT
from src.data.store import load_mps

FIG_DIR = PROJECT_ROOT / "data" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BG    = "#FAFAFA"
BG2   = "#FFFFFF"
RED   = "#C0392B"
RED2  = "#E74C3C"
DARK  = "#1a1a1a"
GREY  = "#757575"
GRID  = "#E0E0E0"
BLUE  = "#3498DB"
GREEN = "#27AE60"

mpl.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG2,
    "axes.edgecolor": "#CCCCCC", "axes.labelcolor": DARK,
    "text.color": DARK, "xtick.color": DARK, "ytick.color": DARK,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "font.family": "sans-serif", "font.size": 11,
})

COALITION  = {"KO", "PSL-TD", "Lewica", "Polska2050", "Polska2050-TD", "Razem"}
OPPOSITION = {"PiS", "Konfederacja", "Konfederacja_KP"}


def build_monthly_polarization(term: int = 10) -> pl.DataFrame:
    """
    For each month compute:
      - within_coalition: mean pairwise agreement between coalition MPs
      - within_opposition: mean pairwise agreement between opposition MPs
      - between: mean pairwise agreement between coalition and opposition MPs
      - polarization: (within_coal + within_oppo) / 2 - between
    Uses votes directly (fast, no full matrix needed).
    """
    votes = pl.read_parquet(PARQUET_DIR / f"term{term}/votes.parquet")
    mps   = load_mps(term)

    # Assign bloc
    mps = mps.with_columns(
        pl.col("club").cast(pl.Utf8).map_elements(
            lambda c: "coalition" if c in COALITION else ("opposition" if c in OPPOSITION else "other"),
            return_dtype=pl.Utf8,
        ).alias("bloc")
    )
    bloc_map = dict(zip(mps["mp_id"].to_list(), mps["bloc"].to_list()))

    votes = (
        votes
        .with_columns([
            pl.col("date").dt.strftime("%Y-%m").alias("year_month"),
            pl.col("vote").cast(pl.Utf8).alias("vote_str"),
            pl.col("mp_id").map_elements(lambda x: bloc_map.get(x, "other"), return_dtype=pl.Utf8).alias("bloc"),
        ])
        .filter(pl.col("bloc").is_in(["coalition", "opposition"]))
        .filter(pl.col("vote_str").is_in(["YES", "NO", "ABSTAIN"]))
        .with_columns(
            pl.when(pl.col("vote_str") == "YES").then(pl.lit(1.0))
            .when(pl.col("vote_str") == "NO").then(pl.lit(-1.0))
            .otherwise(pl.lit(0.0)).alias("vote_val")
        )
    )

    # Per voting × bloc: mean vote value
    bloc_means = (
        votes
        .group_by(["voting_key", "year_month", "bloc"])
        .agg(
            pl.col("vote_val").mean().alias("mean_vote"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= 5)
    )

    # Pivot to wide: coalition mean and opposition mean per voting
    wide = (
        bloc_means
        .pivot(on="bloc", index=["voting_key", "year_month"],
               values="mean_vote", aggregate_function="mean")
        .drop_nulls()
    )
    if "coalition" not in wide.columns or "opposition" not in wide.columns:
        return pl.DataFrame()

    # Agreement between blocs = 1 - |coal_mean - oppo_mean| / 2 (normalised to [0,1])
    wide = wide.with_columns(
        (1 - (pl.col("coalition") - pl.col("opposition")).abs() / 2).alias("between_agree"),
        (1 - (pl.col("coalition") - pl.col("coalition")).abs() / 2).alias("within_coal_agree"),
    )

    # Per-voting within-bloc agreement: use std as proxy (low std = high agreement)
    # Compute properly using pairwise approach via mean abs deviation
    within = (
        votes
        .group_by(["voting_key", "year_month", "bloc"])
        .agg([
            pl.col("vote_val").mean().alias("mean_vote"),
            pl.col("vote_val").std().alias("std_vote"),
            pl.len().alias("n"),
        ])
        .with_columns(
            (1 - pl.col("std_vote").fill_null(0) / 1.0).clip(0, 1).alias("within_agree")
        )
        .filter(pl.col("n") >= 5)
    )

    within_coal = within.filter(pl.col("bloc") == "coalition")
    within_oppo = within.filter(pl.col("bloc") == "opposition")

    monthly_coal = (
        within_coal.group_by("year_month")
        .agg(pl.col("within_agree").mean().alias("within_coalition"))
    )
    monthly_oppo = (
        within_oppo.group_by("year_month")
        .agg(pl.col("within_agree").mean().alias("within_opposition"))
    )
    monthly_between = (
        wide.group_by("year_month")
        .agg(pl.col("between_agree").mean().alias("between_blocs"))
    )

    result = (
        monthly_coal
        .join(monthly_oppo, on="year_month", how="inner")
        .join(monthly_between, on="year_month", how="inner")
        .with_columns(
            (
                (pl.col("within_coalition") + pl.col("within_opposition")) / 2
                - pl.col("between_blocs")
            ).alias("polarization")
        )
        .sort("year_month")
    )
    return result


def build_wild_cards(term: int = 10) -> pl.DataFrame:
    """
    Wild cards: votings where coalition MPs voted predominantly YES
    but the vote FAILED, or predominantly NO but the vote PASSED.
    Also finds votes where defectors decided the outcome.
    """
    votes   = pl.read_parquet(PARQUET_DIR / f"term{term}/votes.parquet")
    votings = pl.read_parquet(PARQUET_DIR / f"term{term}/votings.parquet")
    mps     = load_mps(term)

    coal_ids = set(
        mps.filter(pl.col("club").cast(pl.Utf8).is_in(COALITION))["mp_id"].to_list()
    )

    # Coalition vote intent per voting
    coal_votes = (
        votes
        .filter(pl.col("mp_id").is_in(coal_ids))
        .filter(pl.col("vote").cast(pl.Utf8).is_in(["YES", "NO"]))
        .with_columns(
            (pl.col("vote").cast(pl.Utf8) == "YES").alias("is_yes")
        )
        .group_by("voting_key")
        .agg([
            pl.col("is_yes").sum().alias("coal_yes"),
            pl.len().alias("coal_total"),
        ])
        .with_columns(
            (pl.col("coal_yes") / pl.col("coal_total")).alias("coal_yes_frac")
        )
    )

    # Voting outcome
    result_df = (
        votings.select(["voting_key", "yes_count", "no_count", "total_voted", "title", "date"])
        .join(coal_votes, on="voting_key", how="inner")
        .filter(pl.col("coal_total") >= 30)  # enough coalition MPs voted
        .with_columns([
            (pl.col("yes_count") > pl.col("no_count")).alias("passed"),
            (pl.col("coal_yes_frac") >= 0.65).alias("coal_wanted_yes"),
            (pl.col("coal_yes_frac") <= 0.35).alias("coal_wanted_no"),
        ])
        .filter(
            (pl.col("coal_wanted_yes") & ~pl.col("passed")) |   # coalition wanted YES but lost
            (pl.col("coal_wanted_no")  &  pl.col("passed"))     # coalition wanted NO but it passed
        )
        .with_columns(
            pl.when(pl.col("coal_wanted_yes") & ~pl.col("passed"))
            .then(pl.lit("Koalicja chciała TAK, przegrała"))
            .otherwise(pl.lit("Koalicja chciała NIE, przeszło"))
            .alias("wild_type")
        )
        .sort("date")
    )
    return result_df


# ── figures ───────────────────────────────────────────────────────────────────

def fig19_polarization(pol_df: pl.DataFrame) -> None:
    print("  fig19: polarization …")
    if pol_df.is_empty():
        print("    no data, skipping")
        return

    months = pol_df["year_month"].to_list()
    x      = np.arange(len(months))

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), facecolor=BG, sharex=True)
    fig.suptitle("Ewolucja polaryzacji politycznej  |  Kadencja X",
                 fontsize=14, color=DARK, y=1.01)

    ax1 = axes[0]
    ax1.set_facecolor(BG2)
    ax1.plot(x, pol_df["within_coalition"].to_list(), color=RED2, linewidth=2.2,
             label="Spójność koalicji", marker="o", markersize=3.5)
    ax1.plot(x, pol_df["within_opposition"].to_list(), color="#8B0000", linewidth=2.2,
             label="Spójność opozycji", marker="s", markersize=3.5, linestyle="--")
    ax1.plot(x, pol_df["between_blocs"].to_list(), color=GREY, linewidth=2.0,
             label="Zgodność między obozami", marker="^", markersize=3.5, linestyle=":")
    ax1.set_ylabel("Wskaźnik zgodności głosowań", fontsize=10)
    ax1.set_title("Spójność wewnętrzna obozów vs zgodność między nimi", fontsize=11)
    ax1.grid(alpha=0.25); ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(fontsize=9, framealpha=0.5)
    ax1.set_ylim(0, 1.05)

    ax2 = axes[1]
    ax2.set_facecolor(BG2)
    pol_vals = pol_df["polarization"].to_list()
    ax2.fill_between(x, pol_vals, alpha=0.25, color=RED)
    ax2.plot(x, pol_vals, color=RED, linewidth=2.5, marker="o", markersize=4)
    # Rolling average (window 3)
    if len(pol_vals) >= 3:
        roll = np.convolve(pol_vals, np.ones(3)/3, mode="valid")
        ax2.plot(x[1:-1], roll, color="#8B0000", linewidth=1.8, linestyle="--",
                 alpha=0.7, label="Średnia krocząca (3 m-ce)")
        ax2.legend(fontsize=9, framealpha=0.5)
    ax2.set_ylabel("Indeks polaryzacji", fontsize=10)
    ax2.set_title("Indeks polaryzacji = spójność wewnętrzna − zgodność między obozami", fontsize=11)
    ax2.grid(alpha=0.25); ax2.spines[["top", "right"]].set_visible(False)
    ax2.axhline(0, color=DARK, linewidth=0.8, linestyle="-", alpha=0.4)

    ax2.set_xticks(range(0, len(months), 2))
    ax2.set_xticklabels(months[::2], rotation=45, ha="right", fontsize=8.5)

    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig19_polarization.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    saved {out}")


def fig20_wild_cards(wc_df: pl.DataFrame) -> None:
    print("  fig20: wild cards …")
    if wc_df.is_empty():
        print("    no wild cards found")
        return

    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

    # Left: count per month
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(BG2)
    monthly = (
        wc_df
        .with_columns(pl.col("date").cast(pl.Date).dt.strftime("%Y-%m").alias("year_month"))
        .group_by(["year_month", "wild_type"])
        .agg(pl.len().alias("n"))
        .sort("year_month")
    )
    all_months = sorted(monthly["year_month"].unique().to_list())
    type1 = "Koalicja chciała TAK, przegrała"
    type2 = "Koalicja chciała NIE, przeszło"
    n1 = []
    n2 = []
    for m in all_months:
        sub = monthly.filter(pl.col("year_month") == m)
        n1.append(int(sub.filter(pl.col("wild_type") == type1)["n"].sum() or 0))
        n2.append(int(sub.filter(pl.col("wild_type") == type2)["n"].sum() or 0))

    x = np.arange(len(all_months))
    ax1.bar(x, n1, color=RED,   alpha=0.8, label="Koalicja chciała TAK, przegrała")
    ax1.bar(x, n2, bottom=n1, color="#8B0000", alpha=0.8, label="Koalicja chciała NIE, przeszło")
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels(all_months[::2], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Liczba głosowań", fontsize=10)
    ax1.set_title(f"Nieoczekiwane wyniki głosowań wg miesiąca\n(łącznie: {len(wc_df)})", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(fontsize=8.5, framealpha=0.5)

    # Right: table of top wild cards
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(BG2)
    ax2.axis("off")
    top20 = wc_df.sort("date", descending=True).head(20)
    titles = [
        (t[:55] + "…") if len(t) > 55 else t
        for t in top20["title"].to_list()
    ]
    col_labels = ["Data", "Wynik", "Typ"]
    rows_data = [
        [str(r["date"])[:10], f"{r['yes_count']}:{r['no_count']}", r["wild_type"][:30]]
        for r in top20.iter_rows(named=True)
    ]
    table = ax2.table(
        cellText=rows_data,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor(RED_PALE := "#FFCDD2")
            cell.set_text_props(fontweight="bold", color=DARK)
        elif r % 2 == 0:
            cell.set_facecolor("#F5F5F5")
        else:
            cell.set_facecolor(BG2)
    ax2.set_title("Ostatnie 20 nieoczekiwanych wyników", fontsize=11, pad=12)

    fig.suptitle("Dzikie karty — głosowania z nieoczekiwanym wynikiem  |  Kadencja X",
                 fontsize=14, color=DARK, y=1.01)
    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig20_wild_cards.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    saved {out}")


if __name__ == "__main__":
    print("Script 18 — Polarization + wild cards …")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    pol_df = build_monthly_polarization()
    if not pol_df.is_empty():
        pol_df.write_parquet(ANALYSIS_DIR / "monthly_polarization.parquet")

    wc_df = build_wild_cards()
    if not wc_df.is_empty():
        wc_df.write_parquet(ANALYSIS_DIR / "wild_cards.parquet")

    fig19_polarization(pol_df)
    fig20_wild_cards(wc_df)
    print(f"\nDone. Figures in {FIG_DIR}")
