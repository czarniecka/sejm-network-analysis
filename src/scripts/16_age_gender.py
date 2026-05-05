"""
Script 16 — Age, gender and profession analysis.

Analyses:
  1. Age distribution by club (violin + box)
  2. Rebel rate vs age (scatter + trend)
  3. Gender (inferred from name ending) distribution and rebel rate by gender
  4. Rebel rate vs profession (top professions)

Figures:
  fig14_age_by_club.png
  fig15_gender_analysis.png
  fig16_profession_rebel.png
"""

import sys
from pathlib import Path
from datetime import date

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import ANALYSIS_DIR, PROJECT_ROOT
from src.data.store import load_mps

FIG_DIR = PROJECT_ROOT / "data" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BG       = "#FAFAFA"
BG2      = "#FFFFFF"
RED      = "#C0392B"
RED2     = "#E74C3C"
RED_PALE = "#FFCDD2"
DARK     = "#1a1a1a"
GREY     = "#757575"
GRID     = "#E0E0E0"

CLUB_COLOURS = {
    "KO": "#E8C4C4", "PiS": "#8B0000", "PSL-TD": "#C0392B",
    "Lewica": "#FF5252", "Polska2050": "#FF8C69", "Polska2050-TD": "#FFAB91",
    "Konfederacja": "#5D0000", "Konfederacja_KP": "#7A0000",
    "Razem": "#FF7675", "PSL": "#D4826B", "Centrum": "#F0A090",
    "niez.": "#777777", "Demokracja": "#B05050",
}
MAIN_CLUBS = ["KO", "PiS", "PSL-TD", "Lewica", "Polska2050-TD", "Konfederacja", "Razem"]

mpl.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG2,
    "axes.edgecolor": "#CCCCCC", "axes.labelcolor": DARK,
    "text.color": DARK, "xtick.color": DARK, "ytick.color": DARK,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "font.family": "sans-serif", "font.size": 11,
})

REF_DATE = date(2023, 11, 13)   # start of term X


def build_enriched(term: int = 10) -> pl.DataFrame:
    mps     = load_mps(term)
    rebels  = pl.read_parquet(ANALYSIS_DIR / "rebels.parquet")

    df = mps.join(rebels.select(["mp_id", "rebel_rate", "total_votes"]), on="mp_id", how="left")

    df = df.with_columns([
        pl.col("birth_date").map_elements(
            lambda d: (REF_DATE - d).days // 365 if d is not None else None,
            return_dtype=pl.Int32,
        ).alias("age"),
        # Female heuristic: Polish female names end in 'a' (≥95% accuracy)
        pl.col("first_name").str.ends_with("a").alias("is_female"),
        pl.col("club").cast(pl.Utf8).alias("club_str"),
    ])
    return df


# ── figures ───────────────────────────────────────────────────────────────────

def fig14_age_by_club(df: pl.DataFrame) -> None:
    print("  fig14: age by club …")
    data = df.filter(
        pl.col("club_str").is_in(MAIN_CLUBS) & pl.col("age").is_not_null()
    )
    clubs_ordered = (
        data.group_by("club_str")
        .agg(pl.col("age").median().alias("med"))
        .sort("med")
        ["club_str"].to_list()
    )

    fig = plt.figure(figsize=(13, 6.5), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

    # left: violin
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(BG2)
    for i, club in enumerate(clubs_ordered):
        ages = data.filter(pl.col("club_str") == club)["age"].drop_nulls().to_numpy()
        col  = CLUB_COLOURS.get(club, GREY)
        vp   = ax1.violinplot(ages, positions=[i], widths=0.7, showmedians=True)
        for pc in vp["bodies"]:
            pc.set_facecolor(col); pc.set_alpha(0.7)
        vp["cmedians"].set_color(DARK); vp["cmedians"].set_linewidth(2)
        for part in ("cbars", "cmins", "cmaxes"):
            vp[part].set_color(DARK); vp[part].set_linewidth(0.8)

    ax1.set_xticks(range(len(clubs_ordered)))
    ax1.set_xticklabels(clubs_ordered, fontsize=9.5, rotation=20, ha="right")
    ax1.set_ylabel("Wiek (lata, stan na 13.11.2023)", fontsize=10)
    ax1.set_title("Rozkład wieku wg klubu", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    # right: rebel rate vs age scatter
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(BG2)
    valid = data.filter(pl.col("rebel_rate").is_not_null() & (pl.col("total_votes").cast(pl.Float64) >= 500))
    ages  = valid["age"].to_numpy()
    rates = valid["rebel_rate"].to_numpy()
    cols  = [CLUB_COLOURS.get(c, GREY) for c in valid["club_str"].to_list()]

    ax2.scatter(ages, rates, c=cols, s=22, alpha=0.75, linewidths=0)
    mask = ~np.isnan(ages.astype(float)) & ~np.isnan(rates)
    if mask.sum() > 2:
        coef = np.polyfit(ages[mask], rates[mask], 1)
        xfit = np.linspace(ages[mask].min(), ages[mask].max(), 100)
        ax2.plot(xfit, np.polyval(coef, xfit), color=RED, linewidth=2, linestyle="--", alpha=0.8)
        corr = np.corrcoef(ages[mask], rates[mask])[0, 1]
        ax2.text(0.97, 0.96, f"r = {corr:.3f}", transform=ax2.transAxes,
                 ha="right", va="top", fontsize=10, color=RED)

    ax2.set_xlabel("Wiek posła", fontsize=10)
    ax2.set_ylabel("Wskaźnik buntownictwa", fontsize=10)
    ax2.set_title("Buntownictwo vs wiek", fontsize=12)
    ax2.grid(alpha=0.25)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Analiza wieku posłów  |  Kadencja X", fontsize=14, y=1.02, color=DARK)
    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig14_age_by_club.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    saved {out}")


def fig15_gender_analysis(df: pl.DataFrame) -> None:
    print("  fig15: gender analysis …")
    data = df.filter(pl.col("club_str").is_in(MAIN_CLUBS))

    fig = plt.figure(figsize=(14, 6), facecolor=BG)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42)

    # left: gender share per club
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(BG2)
    gender_club = (
        data.group_by(["club_str", "is_female"])
        .agg(pl.len().alias("n"))
        .pivot(on="is_female", index="club_str", values="n", aggregate_function="sum")
        .fill_null(0)
        .rename({"true": "kobiety", "false": "mężczyźni"})
        .with_columns(
            (pl.col("kobiety") / (pl.col("kobiety") + pl.col("mężczyźni"))).alias("pct_female")
        )
        .sort("pct_female", descending=True)
    )
    clubs  = gender_club["club_str"].to_list()
    female = gender_club["pct_female"].to_list()
    male   = [1 - f for f in female]
    y = np.arange(len(clubs))
    ax1.barh(y, female, color=RED,  alpha=0.8, label="Kobiety")
    ax1.barh(y, male, left=female, color="#CCCCCC", alpha=0.7, label="Mężczyźni")
    ax1.set_yticks(y); ax1.set_yticklabels(clubs, fontsize=9.5)
    ax1.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax1.set_xlabel("Odsetek", fontsize=10)
    ax1.set_title("Udział kobiet wg klubu", fontsize=11)
    ax1.axvline(0.5, color=DARK, linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.grid(axis="x", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(fontsize=8.5, framealpha=0.5)

    # middle: rebel rate by gender per club
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(BG2)
    rebel_gender = (
        data.filter(pl.col("rebel_rate").is_not_null() & (pl.col("total_votes") >= 200))
        .group_by(["club_str", "is_female"])
        .agg(pl.col("rebel_rate").mean().alias("mean_rebel"))
        .sort(["club_str", "is_female"])
    )
    x = np.arange(len(clubs))
    w = 0.35
    female_r = {r["club_str"]: r["mean_rebel"] for r in
                rebel_gender.filter(pl.col("is_female")).iter_rows(named=True)}
    male_r   = {r["club_str"]: r["mean_rebel"] for r in
                rebel_gender.filter(~pl.col("is_female")).iter_rows(named=True)}
    fv = [female_r.get(c, 0) for c in clubs]
    mv = [male_r.get(c, 0) for c in clubs]
    ax2.bar(x - w/2, fv, w, label="Kobiety",    color=RED,       alpha=0.8, edgecolor="#333", linewidth=0.4)
    ax2.bar(x + w/2, mv, w, label="Mężczyźni",  color="#CCCCCC", alpha=0.8, edgecolor="#333", linewidth=0.4)
    ax2.set_xticks(x); ax2.set_xticklabels(clubs, fontsize=8.5, rotation=25, ha="right")
    ax2.set_ylabel("Średni wskaźnik buntownictwa", fontsize=10)
    ax2.set_title("Buntownictwo wg płci i klubu", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(fontsize=8.5, framealpha=0.5)

    # right: overall gender stats
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(BG2)
    overall = (
        data.filter(pl.col("rebel_rate").is_not_null() & (pl.col("total_votes") >= 200))
        .group_by("is_female")
        .agg(
            pl.col("rebel_rate").mean().alias("mean_rebel"),
            pl.col("rebel_rate").median().alias("median_rebel"),
            pl.len().alias("n"),
        )
    )
    labels = ["Mężczyźni", "Kobiety"]
    gender_bool = [False, True]
    means = []
    for g in gender_bool:
        row = overall.filter(pl.col("is_female") == g)
        means.append(float(row["mean_rebel"][0]) if len(row) > 0 else 0)
    bars = ax3.bar(labels, means, color=["#CCCCCC", RED], alpha=0.85,
                   edgecolor="#333", linewidth=0.6, width=0.5)
    for bar, val in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11, color=DARK)
    ax3.set_ylabel("Średni wskaźnik buntownictwa", fontsize=10)
    ax3.set_title("Ogółem: buntownictwo\nwg płci", fontsize=11)
    ax3.grid(axis="y", alpha=0.3)
    ax3.spines[["top", "right"]].set_visible(False)

    total_f = int(data["is_female"].sum())
    total_m = len(data) - total_f
    ax3.text(0, -0.08, f"n={total_m}", transform=ax3.get_xaxis_transform(), ha="center", fontsize=9, color=GREY)
    ax3.text(1, -0.08, f"n={total_f}", transform=ax3.get_xaxis_transform(), ha="center", fontsize=9, color=GREY)

    fig.suptitle("Analiza płci posłów  |  Kadencja X\n(płeć określona na podstawie końcówki imienia)",
                 fontsize=13, y=1.02, color=DARK)
    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig15_gender_analysis.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    saved {out}")


def fig16_profession_rebel(df: pl.DataFrame) -> None:
    print("  fig16: profession vs rebel rate …")
    data = df.filter(
        pl.col("rebel_rate").is_not_null()
        & (pl.col("total_votes").cast(pl.Float64) >= 200)
        & (pl.col("profession") != "")
    )
    top_profs = (
        data.group_by("profession")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= 8)
        .sort("n", descending=True)
        .head(15)
        ["profession"].to_list()
    )
    data = data.filter(pl.col("profession").is_in(top_profs))

    prof_stats = (
        data.group_by("profession")
        .agg([
            pl.col("rebel_rate").mean().alias("mean_rebel"),
            pl.col("rebel_rate").std().alias("std_rebel"),
            pl.len().alias("n"),
        ])
        .sort("mean_rebel", descending=True)
    )

    profs   = prof_stats["profession"].to_list()
    means   = prof_stats["mean_rebel"].to_numpy()
    stds    = prof_stats["std_rebel"].fill_null(0).to_numpy()
    ns      = prof_stats["n"].to_list()

    fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
    ax.set_facecolor(BG2)
    y = np.arange(len(profs))
    norm = mpl.colors.Normalize(vmin=means.min(), vmax=means.max())
    cmap_r = mpl.colormaps.get_cmap("Reds")
    cols = [cmap_r(norm(m)) for m in means]

    bars = ax.barh(y, means, xerr=stds, color=cols, alpha=0.88,
                   edgecolor="#333", linewidth=0.4,
                   error_kw=dict(elinewidth=0.8, capsize=3, ecolor="#555"))
    for bar, m, n in zip(bars, means, ns):
        ax.text(m + max(stds) + 0.003, bar.get_y() + bar.get_height()/2,
                f"{m:.3f}  (n={n})", va="center", fontsize=8.5, color=DARK)

    ax.set_yticks(y); ax.set_yticklabels(profs, fontsize=9.5)
    ax.set_xlabel("Średni wskaźnik buntownictwa (± std)", fontsize=10)
    ax.set_title("Buntownictwo wg zawodu posła  |  Kadencja X\n"
                 "(min. 8 posłów z danym zawodem, min. 200 głosowań)", fontsize=12, pad=12)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=1.2)
    out = FIG_DIR / "fig16_profession_rebel.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    saved {out}")


if __name__ == "__main__":
    print("Script 16 — Age / gender / profession analysis …")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    df = build_enriched()
    df.write_parquet(ANALYSIS_DIR / "mps_enriched.parquet")
    fig14_age_by_club(df)
    fig15_gender_analysis(df)
    fig16_profession_rebel(df)
    print(f"\nDone. Figures in {FIG_DIR}")
