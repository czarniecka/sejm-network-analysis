"""
Shared poster-quality style settings for all visualization scripts.

Provides:
  - apply_style()       — call once per script to set rcParams
  - CLUB_COLOURS        — distinguishable, colorblind-friendly palette
  - COALITION_COLOUR / OPPOSITION_COLOUR
  - MAIN_CLUBS          — ordered list of main parliamentary clubs
  - PALETTE             — named accent colours for non-club data
  - club_label(name)    — short English label for a club name
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Club colour palette ────────────────────────────────────────────────────
# Uses matplotlib tab20 values — designed for categorical distinguishability.
# Each club has a unique hue; no two clubs share a similar shade.
CLUB_COLOURS: dict[str, str] = {
    "KO":              "#1f77b4",   # steel blue
    "PiS":             "#d62728",   # brick red
    "PSL-TD":          "#2ca02c",   # forest green
    "Lewica":          "#ff7f0e",   # orange
    "Polska2050-TD":   "#9467bd",   # purple
    "Konfederacja":    "#8c564b",   # brown
    "Konfederacja_KP": "#e377c2",   # pink
    "Razem":           "#17becf",   # teal
    "PSL":             "#bcbd22",   # olive
    "Polska2050":      "#aec7e8",   # light blue
    "Centrum":         "#ffbb78",   # peach
    "niez.":           "#7f7f7f",   # neutral grey
    "Demokracja":      "#98df8a",   # light green
}

def cc(name: str) -> str:
    """Return hex colour for a club name."""
    return CLUB_COLOURS.get(str(name), "#7f7f7f")

# ── Bloc colours ───────────────────────────────────────────────────────────
COALITION_COLOUR  = "#1f77b4"   # blue
OPPOSITION_COLOUR = "#d62728"   # red
NEUTRAL_COLOUR    = "#7f7f7f"   # grey

COALITION  = {"KO", "PSL-TD", "Lewica", "Polska2050", "Polska2050-TD", "Razem"}
OPPOSITION = {"PiS", "Konfederacja", "Konfederacja_KP"}

MAIN_CLUBS = ["KO", "PiS", "PSL-TD", "Lewica", "Polska2050-TD", "Konfederacja", "Razem"]

# ── English short labels ───────────────────────────────────────────────────
CLUB_LABELS_EN: dict[str, str] = {
    "KO":              "KO",
    "PiS":             "PiS",
    "PSL-TD":          "PSL-TD",
    "Lewica":          "Lewica",
    "Polska2050-TD":   "PL2050-TD",
    "Polska2050":      "PL2050",
    "Konfederacja":    "Konfederacja",
    "Konfederacja_KP": "Konf. KP",
    "Razem":           "Razem",
    "PSL":             "PSL",
    "Centrum":         "Centrum",
    "niez.":           "Indep.",
    "Demokracja":      "Demokracja",
}

def club_en(name: str) -> str:
    return CLUB_LABELS_EN.get(str(name), str(name))

# ── Named accent palette ───────────────────────────────────────────────────
PALETTE = {
    "primary":    "#1f77b4",
    "secondary":  "#ff7f0e",
    "accent":     "#d62728",
    "positive":   "#2ca02c",
    "negative":   "#d62728",
    "neutral":    "#7f7f7f",
    "light_grey": "#E5E5E5",
    "dark":       "#1a1a1a",
    "bg":         "#FFFFFF",
}

# ── rcParams for poster quality ────────────────────────────────────────────
POSTER_RC = {
    # backgrounds
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    # spines
    "axes.edgecolor":    "#CCCCCC",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    # text
    "font.family":        "sans-serif",
    "font.size":          12,
    "axes.titlesize":     14,
    "axes.titleweight":   "bold",
    "axes.labelsize":     12,
    "axes.labelcolor":    "#1a1a1a",
    "text.color":         "#1a1a1a",
    "xtick.color":        "#1a1a1a",
    "ytick.color":        "#1a1a1a",
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    # grid
    "grid.color":         "#E5E5E5",
    "grid.linewidth":     0.8,
    # legend
    "legend.facecolor":   "white",
    "legend.edgecolor":   "#CCCCCC",
    "legend.fontsize":    10,
    "legend.title_fontsize": 11,
    "legend.framealpha":  0.95,
    # figure
    "figure.dpi":         120,
    "savefig.dpi":        250,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
}


def apply_style() -> None:
    """Apply poster-quality rcParams. Call once at module level."""
    mpl.rcParams.update(POSTER_RC)
