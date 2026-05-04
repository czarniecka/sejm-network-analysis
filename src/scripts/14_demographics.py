"""
Script 14 — Demographics analysis: birth geography, education, migration.

Reads:  data/parquet/term{N}/mps.parquet
Writes: data/analysis/birth_geography.parquet
        data/analysis/education_by_party.parquet
        data/analysis/migration_flows.parquet
        data/figures/fig7_birth_geography.png
        data/figures/fig8_education.png
        data/figures/fig9_migration.png

Usage:
    python src/scripts/14_demographics.py --term 10
"""

import logging
import sys
from pathlib import Path

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import ANALYSIS_DIR
FIGURES_DIR = Path(__file__).parent.parent.parent / "data" / "figures"
from src.data.store import load_mps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── colour constants ────────────────────────────────────────────────────────
BG       = "#FAFAFA"
BG2      = "#FFFFFF"
RED      = "#C0392B"
RED2     = "#E74C3C"
RED_PALE = "#FFCDD2"
DARK     = "#1a1a1a"
GREY     = "#9E9E9E"
GRIDCOL  = "#E0E0E0"

CLUB_COLORS = {
    "KO":              "#C0392B",
    "PSL-TD":          "#E57373",
    "Lewica":          "#EF9A9A",
    "Razem":           "#FFCDD2",
    "Polska2050":      "#B71C1C",
    "Polska2050-TD":   "#D32F2F",
    "PiS":             "#37474F",
    "Konfederacja":    "#546E7A",
    "Konfederacja_KP": "#607D8B",
    "Centrum":         "#90A4AE",
    "Demokracja":      "#B0BEC5",
    "niezrzeszony":    "#CFD8DC",
    "niez.":           "#CFD8DC",
}

# Electoral district → voivodeship mapping
DISTRICT_TO_VOIV = {
    "Legnica": "dolnośląskie", "Wałbrzych": "dolnośląskie", "Wrocław": "dolnośląskie",
    "Bydgoszcz": "kujawsko-pomorskie", "Toruń": "kujawsko-pomorskie",
    "Chełm": "lubelskie", "Lublin": "lubelskie",
    "Zielona Góra": "lubuskie",
    "Łódź": "łódzkie", "Piotrków Trybunalski": "łódzkie", "Sieradz": "łódzkie",
    "Kraków": "małopolskie", "Nowy Sącz": "małopolskie", "Tarnów": "małopolskie",
    "Katowice": "śląskie", "Bielsko-Biała": "śląskie", "Częstochowa": "śląskie",
    "Kielce": "świętokrzyskie",
    "Olsztyn": "warmińsko-mazurskie", "Elbląg": "warmińsko-mazurskie",
    "Białystok": "podlaskie",
    "Gdańsk": "pomorskie", "Słupsk": "pomorskie",
    "Koszalin": "zachodniopomorskie", "Szczecin": "zachodniopomorskie",
    "Płock": "mazowieckie", "Radom": "mazowieckie", "Siedlce": "mazowieckie", "Warszawa": "mazowieckie",
    "Opole": "opolskie",
    "Krosno": "podkarpackie", "Rzeszów": "podkarpackie",
    "Piła": "wielkopolskie", "Kalisz": "wielkopolskie", "Konin": "wielkopolskie", "Poznań": "wielkopolskie",
}

# Birth city → voivodeship mapping (complete coverage of term10 MPs)
BIRTH_CITY_VOIV = {
    "Aalborg, Dania": "zagranica", "Chicago (Usa)": "zagranica",
    "Gańczary K. Lwowa": "zagranica", "Szwajcaria": "zagranica", "Wańkowa": "zagranica",
    "Aleksandrów Kujawski": "kujawsko-pomorskie", "Annopol": "lubelskie",
    "Barlinek": "zachodniopomorskie", "Biała": "opolskie", "Biała Podlaska": "lubelskie",
    "Białystok": "podlaskie", "Bielsko-Biała": "śląskie", "Bilcza": "świętokrzyskie",
    "Biskupiec": "warmińsko-mazurskie", "Biłgoraj": "lubelskie", "Bochnia": "małopolskie",
    "Bolesławiec": "dolnośląskie", "Brusy": "pomorskie", "Brzostek": "podkarpackie",
    "Busko Zdrój": "świętokrzyskie", "Bydgoszcz": "kujawsko-pomorskie", "Bytom": "śląskie",
    "Chełm": "lubelskie", "Chełmno": "kujawsko-pomorskie", "Chojnice": "pomorskie",
    "Chojnów": "dolnośląskie", "Chrośnica": "dolnośląskie", "Ciechanów": "mazowieckie",
    "Cieszyn": "śląskie", "Czeladź": "śląskie", "Czernikowo": "kujawsko-pomorskie",
    "Częstochowa": "śląskie", "Dobra": "zachodniopomorskie", "Domatowo": "kujawsko-pomorskie",
    "Dylągówka": "podkarpackie", "Działdowo": "warmińsko-mazurskie", "Dzierżoniów": "dolnośląskie",
    "Dąbrowa Tarnowska": "małopolskie", "Dąbrówka": "mazowieckie", "Dębica": "podkarpackie",
    "Elbląg": "warmińsko-mazurskie", "Ełk": "warmińsko-mazurskie", "Garwolin": "mazowieckie",
    "Gdańsk": "pomorskie", "Gdynia": "pomorskie", "Giżycko": "warmińsko-mazurskie",
    "Gliwice": "śląskie", "Gniezno": "wielkopolskie", "Godziesze Wielkie": "wielkopolskie",
    "Gorlice": "małopolskie", "Grabinka": "mazowieckie", "Grajewo": "podlaskie",
    "Grodzisko Dolne": "podkarpackie", "Grudziądz": "kujawsko-pomorskie",
    "Gryfice": "zachodniopomorskie", "Gryfino": "zachodniopomorskie", "Grójec": "mazowieckie",
    "Góra": "dolnośląskie", "Głogów": "dolnośląskie", "Głubczyce": "opolskie",
    "Głuchołazy": "opolskie", "Głuszyca": "dolnośląskie", "Inowrocław": "kujawsko-pomorskie",
    "Iława": "warmińsko-mazurskie", "Janów Lubelski": "lubelskie", "Jarocin": "wielkopolskie",
    "Jarosław": "podkarpackie", "Jawor": "dolnośląskie", "Jaworzno": "śląskie",
    "Jelenin": "dolnośląskie", "Jodłowa": "podkarpackie", "Józefów": "mazowieckie",
    "Jędrzejów": "świętokrzyskie", "Kalisz": "wielkopolskie", "Katowice": "śląskie",
    "Kałuszyn": "mazowieckie", "Kielce": "świętokrzyskie", "Kluczbork": "opolskie",
    "Knurów": "śląskie", "Kochcice": "śląskie", "Kocierz Rydzwałdzki": "śląskie",
    "Kolbuszowa": "podkarpackie", "Komorowo": "mazowieckie", "Koszalin": "zachodniopomorskie",
    "Koło": "wielkopolskie", "Końskie": "świętokrzyskie", "Kościan": "wielkopolskie",
    "Koźmin": "wielkopolskie", "Koźminek": "wielkopolskie", "Kraków": "małopolskie",
    "Krasnobród": "lubelskie", "Kraśnik": "lubelskie", "Krosno": "podkarpackie",
    "Krynica-Zdrój": "małopolskie", "Krzeszowice": "małopolskie", "Kutno": "łódzkie",
    "Kwidzyn": "pomorskie", "Kłobuck": "śląskie", "Kłodzko": "dolnośląskie",
    "Legionowo": "mazowieckie", "Legnica": "dolnośląskie", "Lesko": "podkarpackie",
    "Leszno": "wielkopolskie", "Libiąż": "małopolskie", "Lidzbark Warmiński": "warmińsko-mazurskie",
    "Limanowa": "małopolskie", "Lniano": "kujawsko-pomorskie", "Lubaczów": "podkarpackie",
    "Lubin": "dolnośląskie", "Lublin": "lubelskie", "Lubliniec": "śląskie",
    "Lubsko": "lubuskie", "Lędziny": "śląskie", "Malbork": "pomorskie",
    "Mielec": "podkarpackie", "Mikołów": "śląskie", "Międzyrzec": "lubelskie",
    "Miłomłyn": "warmińsko-mazurskie", "Morąg": "warmińsko-mazurskie",
    "Murowana Goślina": "wielkopolskie", "Myślenice": "małopolskie", "Mława": "mazowieckie",
    "Niechanowo": "wielkopolskie", "Nisko": "podkarpackie", "Nowa Ruda": "dolnośląskie",
    "Nowa Sól": "lubuskie", "Nowy Sącz": "małopolskie", "Nowy Targ": "małopolskie",
    "Olecko": "warmińsko-mazurskie", "Olkusz": "małopolskie", "Olsztyn": "warmińsko-mazurskie",
    "Opoczno": "łódzkie", "Opole": "opolskie", "Ostrowiec Świętokrzyski": "świętokrzyskie",
    "Ostrołęka": "mazowieckie", "Ostrów Mazowiecka": "mazowieckie",
    "Ostrów Wielkopolski": "wielkopolskie", "Oświęcim": "małopolskie",
    "Pabianice": "łódzkie", "Paczków": "opolskie", "Parczew": "lubelskie",
    "Pasłęk": "warmińsko-mazurskie", "Piekary Śląskie": "śląskie",
    "Piotrków Trybunalski": "łódzkie", "Piła": "wielkopolskie", "Poznań": "wielkopolskie",
    "Połczyn-Zdrój": "zachodniopomorskie", "Proszowice": "małopolskie", "Prudnik": "opolskie",
    "Przemków": "dolnośląskie", "Przemyśl": "podkarpackie", "Pszczyna": "śląskie",
    "Puck": "pomorskie", "Płock": "mazowieckie", "Płońsk": "mazowieckie",
    "Rabka": "małopolskie", "Rabka-Zdrój": "małopolskie", "Racibórz": "śląskie",
    "Radom": "mazowieckie", "Radomsko": "łódzkie", "Radymno": "podkarpackie",
    "Radziłów": "podlaskie", "Radzyń Podlaski": "lubelskie", "Rawa Mazowiecka": "łódzkie",
    "Resko": "zachodniopomorskie", "Ruda Śląska": "śląskie", "Rybnik": "śląskie",
    "Ryki": "lubelskie", "Rypin": "kujawsko-pomorskie", "Rzeki Wielkie": "małopolskie",
    "Rzeszów": "podkarpackie", "Rzążew": "mazowieckie", "Sanok": "podkarpackie",
    "Sawice": "lubelskie", "Sawin": "lubelskie", "Siedlce": "mazowieckie",
    "Siedlików": "łódzkie", "Siedliszcze": "lubelskie", "Sielc": "mazowieckie",
    "Siemiatycze": "podlaskie", "Sieradz": "łódzkie", "Skarżysko-Kamienna": "świętokrzyskie",
    "Skała": "małopolskie", "Skierniewice": "łódzkie", "Skrwilno": "kujawsko-pomorskie",
    "Sochaczew": "mazowieckie", "Sopot": "pomorskie", "Sosnowiec": "śląskie",
    "Stalowa Wola": "podkarpackie", "Starachowice": "świętokrzyskie",
    "Starogard Gdański": "pomorskie", "Stawiszyn": "wielkopolskie", "Stopnica": "świętokrzyskie",
    "Sucha Beskidzka": "małopolskie", "Sulechów": "lubuskie", "Suwałki": "podlaskie",
    "Szamotuły": "wielkopolskie", "Szczecin": "zachodniopomorskie",
    "Szczecinek": "zachodniopomorskie", "Szczytno": "warmińsko-mazurskie",
    "Słupsk": "pomorskie", "Tarnobrzeg": "podkarpackie", "Tarnogród": "lubelskie",
    "Tarnowskie Góry": "śląskie", "Tarnów": "małopolskie", "Tczew": "pomorskie",
    "Tomaszów Mazowiecki": "łódzkie", "Toruń": "kujawsko-pomorskie",
    "Trzcianka": "wielkopolskie", "Trzebiatów": "zachodniopomorskie",
    "Trzemeszno": "wielkopolskie", "Tuchów": "małopolskie", "Turek": "wielkopolskie",
    "Warszawa": "mazowieckie", "Wałbrzych": "dolnośląskie", "Wałcz": "zachodniopomorskie",
    "Wejherowo": "pomorskie", "Wieluń": "łódzkie", "Więcbork": "kujawsko-pomorskie",
    "Wola Obszańska": "lubelskie", "Wołczyn": "opolskie", "Wrocław": "dolnośląskie",
    "Września": "wielkopolskie", "Wyrzysk": "wielkopolskie",
    "Wysokie Mazowieckie": "podlaskie", "Włocławek": "kujawsko-pomorskie",
    "Włoszczowa": "świętokrzyskie", "Zakopane": "małopolskie", "Zambrów": "podlaskie",
    "Zamość": "lubelskie", "Zator": "małopolskie", "Zgierz": "łódzkie",
    "Zielona Góra": "lubuskie", "Ząbkowice Śl.": "dolnośląskie",
    "Złocieniec": "zachodniopomorskie", "Złotów": "wielkopolskie",
    "Łańcut": "podkarpackie", "Łomża": "podlaskie", "Łowicz": "łódzkie",
    "Łódź": "łódzkie", "Śrem": "wielkopolskie", "Świdnik": "lubelskie",
    "Świebodzice": "dolnośląskie", "Świebodzin": "lubuskie", "Żabianka": "pomorskie",
    "Żagań": "lubuskie", "Żarów": "dolnośląskie", "Żuromin": "mazowieckie",
}

# Ordered voivodeships: roughly N→S, W→E (for consistent display)
VOIV_ORDER = [
    "zachodniopomorskie", "pomorskie", "warmińsko-mazurskie", "podlaskie",
    "kujawsko-pomorskie", "mazowieckie",
    "wielkopolskie", "łódzkie", "lubelskie",
    "lubuskie", "dolnośląskie", "opolskie", "śląskie", "świętokrzyskie", "podkarpackie",
    "małopolskie",
]

# Simplified coalition labels
COALITION = {"KO", "PSL-TD", "Lewica", "Razem", "Polska2050", "Polska2050-TD"}

def coalition_label(club: str) -> str:
    return "Koalicja" if club in COALITION else "Opozycja"


# ── analysis ─────────────────────────────────────────────────────────────────

def build_birth_geography(mps_df: pl.DataFrame) -> pl.DataFrame:
    """Count MPs per birth voivodeship × club."""
    return (
        mps_df
        .with_columns(
            pl.col("birth_location")
              .map_elements(lambda c: BIRTH_CITY_VOIV.get(c, ""), return_dtype=pl.Utf8)
              .alias("birth_voiv")
        )
        .filter((pl.col("birth_voiv") != "") & (pl.col("birth_voiv") != "zagranica"))
        .group_by(["birth_voiv", "club"])
        .agg(pl.len().alias("n_mps"))
        .sort(["birth_voiv", "club"])
    )


def build_education(mps_df: pl.DataFrame) -> pl.DataFrame:
    """Count MPs per education_level × club."""
    return (
        mps_df
        .group_by(["education_level", "club"])
        .agg(pl.len().alias("n_mps"))
        .sort(["education_level", "club"])
    )


def build_migration(mps_df: pl.DataFrame) -> pl.DataFrame:
    """Build birth_voivodeship → electoral_voivodeship flow matrix."""
    df = (
        mps_df
        .with_columns(
            pl.col("birth_location")
              .map_elements(lambda c: BIRTH_CITY_VOIV.get(c, ""), return_dtype=pl.Utf8)
              .alias("birth_voiv"),
            pl.col("voivodeship").cast(pl.Utf8).alias("electoral_voiv"),
        )
        .filter((pl.col("birth_voiv") != "") & (pl.col("birth_voiv") != "zagranica"))
        .filter(pl.col("electoral_voiv") != "")
        .group_by(["birth_voiv", "electoral_voiv"])
        .agg(pl.len().alias("n_mps"))
        .sort("n_mps", descending=True)
    )
    return df


# ── figures ───────────────────────────────────────────────────────────────────

def fig7_birth_geography(geo_df: pl.DataFrame, mps_df: pl.DataFrame, out: Path) -> None:
    """Stacked bar: birth voivodeship × coalition/opposition."""
    df = (
        mps_df
        .with_columns(
            pl.col("birth_location")
              .map_elements(lambda c: BIRTH_CITY_VOIV.get(c, ""), return_dtype=pl.Utf8)
              .alias("birth_voiv"),
            pl.col("club").cast(pl.Utf8)
              .map_elements(coalition_label, return_dtype=pl.Utf8)
              .alias("bloc"),
        )
        .filter((pl.col("birth_voiv") != "") & (pl.col("birth_voiv") != "zagranica"))
        .group_by(["birth_voiv", "bloc"])
        .agg(pl.len().alias("n"))
    )

    # Pivot to wide
    pivot = (
        df
        .pivot(on="bloc", index="birth_voiv", values="n", aggregate_function="sum")
        .fill_null(0)
    )
    # Ensure both columns exist
    for col in ("Koalicja", "Opozycja"):
        if col not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(0).alias(col))
    pivot = pivot.with_columns(
        (pl.col("Koalicja") + pl.col("Opozycja")).alias("total")
    ).sort("total", descending=True)

    voivs  = pivot["birth_voiv"].to_list()
    coal   = pivot["Koalicja"].to_list()
    oppo   = pivot["Opozycja"].to_list()

    # Also compute total births per voivodeship as % of 498
    n_total = len(mps_df)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    fig.suptitle(
        "Skąd pochodzą posłowie?  |  Kadencja X",
        fontsize=16, fontweight="bold", color=DARK, y=0.98,
    )

    # ── left: stacked bar ─────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG2)
    y = np.arange(len(voivs))
    bars_c = ax.barh(y, coal, color=RED,  alpha=0.85, label="Koalicja")
    bars_o = ax.barh(y, oppo, left=coal, color=GREY, alpha=0.70, label="Opozycja")

    ax.set_yticks(y)
    ax.set_yticklabels(voivs, fontsize=9, color=DARK)
    ax.set_xlabel("Liczba posłów", fontsize=10, color=DARK)
    ax.set_title("Urodzenia wg województwa", fontsize=11, color=DARK)
    ax.tick_params(colors=DARK)
    ax.xaxis.set_tick_params(colors=DARK)
    ax.spines["left"].set_color(GRIDCOL)
    ax.spines["bottom"].set_color(GRIDCOL)
    ax.grid(axis="x", color=GRIDCOL, linewidth=0.6)
    ax.legend(fontsize=9, frameon=False, labelcolor=DARK)

    # ── right: top birth cities ───────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG2)

    city_df = (
        mps_df
        .filter(pl.col("birth_location") != "")
        .group_by("birth_location")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(20)
    )
    cities = city_df["birth_location"].to_list()
    counts = city_df["n"].to_list()

    # color by majority party of MPs born there
    city_colors = []
    for city in cities:
        party_counts = (
            mps_df
            .filter(pl.col("birth_location") == city)
            .group_by("club")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
        )
        top_club = party_counts["club"].cast(pl.Utf8)[0] if len(party_counts) > 0 else ""
        city_colors.append(CLUB_COLORS.get(top_club, GREY))

    y2 = np.arange(len(cities))
    ax2.barh(y2, counts, color=city_colors, alpha=0.85)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(cities, fontsize=9, color=DARK)
    ax2.set_xlabel("Liczba posłów", fontsize=10, color=DARK)
    ax2.set_title("Top 20 miast urodzenia", fontsize=11, color=DARK)
    ax2.tick_params(colors=DARK)
    ax2.xaxis.set_tick_params(colors=DARK)
    ax2.spines["left"].set_color(GRIDCOL)
    ax2.spines["bottom"].set_color(GRIDCOL)
    ax2.grid(axis="x", color=GRIDCOL, linewidth=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig8_education(edu_df: pl.DataFrame, mps_df: pl.DataFrame, out: Path) -> None:
    """Education level breakdown per party."""
    # Simplify to main clubs only
    main_clubs = ["KO", "PiS", "PSL-TD", "Lewica", "Konfederacja", "Polska2050-TD"]

    edu_levels = ["wyższe", "średnie ogólne", "średnie zawodowe", "średnie policealne/pomaturalne"]
    edu_short  = ["wyższe", "śr. ogólne", "śr. zawodowe", "śr. policealne"]

    # Build matrix: clubs × edu_levels (% within club)
    df_filt = (
        mps_df
        .with_columns(pl.col("club").cast(pl.Utf8))
        .filter(pl.col("club").is_in(main_clubs))
    )

    matrix = {}
    for club in main_clubs:
        sub = df_filt.filter(pl.col("club") == club)
        total = len(sub)
        row = []
        for edu in edu_levels:
            cnt = len(sub.filter(pl.col("education_level").cast(pl.Utf8) == edu))
            row.append(100 * cnt / total if total > 0 else 0)
        matrix[club] = row

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    ax.set_facecolor(BG2)
    fig.suptitle(
        "Wykształcenie posłów wg klubu  |  Kadencja X",
        fontsize=15, fontweight="bold", color=DARK,
    )

    x = np.arange(len(main_clubs))
    width = 0.18
    colors_edu = [RED, "#E57373", "#EF9A9A", RED_PALE]
    labels_edu = edu_short

    for i, (edu, color, label) in enumerate(zip(edu_levels, colors_edu, labels_edu)):
        vals = [matrix[club][i] for club in main_clubs]
        ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(main_clubs, fontsize=10, color=DARK)
    ax.set_ylabel("Odsetek posłów w klubie [%]", fontsize=10, color=DARK)
    ax.tick_params(colors=DARK)
    ax.yaxis.set_tick_params(colors=DARK)
    ax.grid(axis="y", color=GRIDCOL, linewidth=0.6)
    ax.spines["left"].set_color(GRIDCOL)
    ax.spines["bottom"].set_color(GRIDCOL)
    ax.legend(fontsize=9, frameon=False, labelcolor=DARK)

    plt.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig9_migration(mps_df: pl.DataFrame, out: Path) -> None:
    """Migration matrix: birth voivodeship → electoral voivodeship."""
    df = (
        mps_df
        .with_columns(
            pl.col("birth_location")
              .map_elements(lambda c: BIRTH_CITY_VOIV.get(c, ""), return_dtype=pl.Utf8)
              .alias("birth_voiv"),
            pl.col("voivodeship").cast(pl.Utf8).alias("electoral_voiv"),
        )
        .filter((pl.col("birth_voiv") != "") & (pl.col("birth_voiv") != "zagranica"))
        .filter(pl.col("electoral_voiv") != "")
    )

    # Build matrix
    all_voivs = sorted(set(df["birth_voiv"].to_list()) | set(df["electoral_voiv"].to_list()))
    n = len(all_voivs)
    mat = np.zeros((n, n), dtype=float)
    voiv_idx = {v: i for i, v in enumerate(all_voivs)}

    for row in df.iter_rows(named=True):
        b = row["birth_voiv"]
        d = row["electoral_voiv"]
        if b in voiv_idx and d in voiv_idx:
            mat[voiv_idx[b], voiv_idx[d]] += 1

    # Count migrants (off-diagonal)
    total_mps = int(mat.sum())
    migrants  = int(mat.sum() - np.trace(mat))
    pct_mig   = 100 * migrants / total_mps if total_mps > 0 else 0

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    fig.suptitle(
        f"Migracja posłów: miejsce urodzenia → okręg wyborczy  |  Kadencja X\n"
        f"{migrants}/{total_mps} posłów ({pct_mig:.0f}%) kandyduje poza województwem urodzenia",
        fontsize=13, fontweight="bold", color=DARK,
    )

    # ── left: heatmap ─────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG2)
    short = [v[:8] for v in all_voivs]
    cmap = LinearSegmentedColormap.from_list("redmap", [BG2, RED_PALE, RED2, RED])
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7, color=DARK)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short, fontsize=7, color=DARK)
    ax.set_xlabel("Województwo okręgu wyborczego", fontsize=9, color=DARK)
    ax.set_ylabel("Województwo urodzenia", fontsize=9, color=DARK)
    ax.set_title("Macierz przepływów", fontsize=11, color=DARK)
    plt.colorbar(im, ax=ax, fraction=0.046, label="Liczba posłów")

    # ── right: net import/export bars ────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG2)

    import_v = mat.sum(axis=0)    # by destination (electoral)
    export_v = mat.sum(axis=1)    # by origin (birth)
    net      = import_v - export_v  # positive = region imports politicians

    sorted_idx = np.argsort(net)
    y = np.arange(n)
    bar_colors = [RED if v >= 0 else GREY for v in net[sorted_idx]]
    ax2.barh(y, net[sorted_idx], color=bar_colors, alpha=0.85)
    ax2.set_yticks(y)
    ax2.set_yticklabels([all_voivs[i] for i in sorted_idx], fontsize=8, color=DARK)
    ax2.axvline(0, color=DARK, linewidth=0.8)
    ax2.set_xlabel("Import − eksport (netto)", fontsize=9, color=DARK)
    ax2.set_title("Które regiony importują / eksportują polityków?", fontsize=11, color=DARK)
    ax2.tick_params(colors=DARK)
    ax2.xaxis.set_tick_params(colors=DARK)
    ax2.grid(axis="x", color=GRIDCOL, linewidth=0.6)
    ax2.spines["left"].set_color(GRIDCOL)
    ax2.spines["bottom"].set_color(GRIDCOL)

    red_patch  = mpatches.Patch(color=RED,  alpha=0.85, label="importuje polityków")
    grey_patch = mpatches.Patch(color=GREY, alpha=0.70, label="eksportuje polityków")
    ax2.legend(handles=[red_patch, grey_patch], fontsize=9, frameon=False, labelcolor=DARK)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info("Saved %s", out)


# ── main ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--term", "-t", type=int, default=10)
def main(term: int) -> None:
    mps_df = load_mps(term)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    geo_df = build_birth_geography(mps_df)
    geo_df.write_parquet(ANALYSIS_DIR / "birth_geography.parquet")

    edu_df = build_education(mps_df)
    edu_df.write_parquet(ANALYSIS_DIR / "education_by_party.parquet")

    mig_df = build_migration(mps_df)
    mig_df.write_parquet(ANALYSIS_DIR / "migration_flows.parquet")

    fig7_birth_geography(geo_df, mps_df, FIGURES_DIR / "fig7_birth_geography.png")
    fig8_education(edu_df, mps_df, FIGURES_DIR / "fig8_education.png")
    fig9_migration(mps_df, FIGURES_DIR / "fig9_migration.png")

    logger.info("Done.")


if __name__ == "__main__":
    main()
