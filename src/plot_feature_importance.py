"""
plot_feature_importance.py

Renders a side-by-side chart of feature importance for the bullet and
blitz models, for the README. Uses a Random Forest for the importances
regardless of which model train_baseline.py ended up deploying for each
category (HistGradientBoostingRegressor doesn't expose
feature_importances_ at all, so this keeps both panels on the same,
comparable footing rather than mixing metrics across model types).

Usage:
    python src/plot_feature_importance.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import train_baseline as tb

TOP_N = 8
OUTPUT_PATH = "docs/feature_importance.png"

DATASETS = [
    ("bullet", "data/processed/players_bullet.csv", "#2a78d6"),
    ("blitz", "data/processed/players_blitz.csv", "#eb6834"),
]

# Chart chrome, from the project's validated default palette.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

# Cosmetic: the raw column names read better with underscores replaced.
LABELS = {
    "avg_num_plies": "avg. game length",
    "avg_time_per_move": "avg. time per move",
    "avg_acpl": "avg. ACPL (own)",
    "avg_opponent_acpl": "avg. ACPL (opponent)",
    "avg_blunders": "avg. blunders",
    "avg_mistakes": "avg. mistakes",
    "avg_inaccuracies": "avg. inaccuracies",
    "avg_opponent_blunders": "avg. blunders (opponent)",
    "avg_opponent_mistakes": "avg. mistakes (opponent)",
    "avg_opponent_inaccuracies": "avg. inaccuracies (opponent)",
    "acpl_diff": "ACPL diff (own - opponent)",
    "win_rate": "win rate",
    "n_games": "games sampled",
}


def top_importances(data_path: str, speed_category: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df = df[df["main_speed_category"] == speed_category].reset_index(drop=True)
    X, y, _ = tb.build_features(df, speed_category)

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances[~importances.index.str.startswith("main_eco_grouped_")]
    return importances.nlargest(TOP_N).sort_values()  # ascending, so barh reads top-to-bottom


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(11, 4.5), facecolor=SURFACE)

    for ax, (category, data_path, color) in zip(axes, DATASETS):
        importances = top_importances(data_path, category)
        labels = [LABELS.get(name, name) for name in importances.index]

        ax.set_facecolor(SURFACE)
        bars = ax.barh(labels, importances.values, color=color, height=0.6)

        for bar, value in zip(bars, importances.values):
            ax.text(
                bar.get_width() + importances.max() * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}", va="center", ha="left", fontsize=9, color=INK_PRIMARY,
            )

        ax.set_title(category, fontsize=13, color=INK_PRIMARY, fontweight="bold", loc="left")
        ax.set_xlim(0, importances.max() * 1.2)
        ax.tick_params(axis="y", colors=INK_PRIMARY, labelsize=9.5, length=0)
        ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8.5)
        ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "Random Forest feature importance, by time control",
        fontsize=14, color=INK_PRIMARY, fontweight="bold", x=0.02, ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTPUT_PATH, dpi=150, facecolor=SURFACE)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
