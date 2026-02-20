# Plots:
# 1) Histograms (raw): tone, uncertainty_rate, CR_0_120
# 2) Boxplots by ticker (raw): tone, CR_0_120
# 3) Scatter + regression line (z): tone_z vs CR_0_120_z, uncertainty_rate_z vs RV_0_120_z, qa_ratio_z vs VOL_ratio_z
# 4) Correlation heatmap (z): selected *_z variables

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

DATA_PATH = "event_level_features.csv"
OUT_DIR = "eda_figs"
os.makedirs(OUT_DIR, exist_ok=True)

mpl.rcParams.update({
    "figure.figsize": (5, 4),
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.3,
    "savefig.dpi": 300,
})


df = pd.read_csv(DATA_PATH)

# -----------------------
# Helpers
# -----------------------
def _pick_col(df_, preferred):
    """Pick the first existing column from 'preferred'; else raise."""
    for c in preferred:
        if c in df_.columns:
            return c
    raise KeyError(f"None of these columns exist: {preferred}")

def _clean_series(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan).dropna()

def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print("Saved:", path)

# -----------------------
# plotting functions
# -----------------------
def hist_plot(series, title, xlabel, filename, bins=20):
    s = _clean_series(series)
    plt.figure()
    plt.hist(s, bins=bins, edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    savefig(filename)

def boxplot_by_ticker(df_, y_col, title, ylabel, filename, ticker_col="ticker"):
    d = df_[[ticker_col, y_col]].copy()
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna()

    tickers = sorted(d[ticker_col].unique().tolist())
    data = [d.loc[d[ticker_col] == t, y_col].values for t in tickers]

    plt.figure(figsize=(7, 4))
    plt.boxplot(
        data,
        labels=tickers,
        patch_artist=True,
        boxprops=dict(facecolor="lightgray", color="black"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker="o", markersize=4, markerfacecolor="gray", alpha=0.5),
    )
    plt.title(title)
    plt.xlabel("Ticker")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    savefig(filename)

def scatter_with_fit(df_, x_col, y_col, title, xlabel, ylabel, filename):
    d = df_[[x_col, y_col]].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna()

    x = d[x_col].values
    y = d[y_col].values

    plt.figure()
    plt.scatter(x, y, alpha=0.6, edgecolor="black", linewidth=0.3)

    # Fit line y = a + b x (only if enough variance)
    if len(x) >= 3 and np.std(x) > 1e-12:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = a + b * xs
        plt.plot(xs, ys, linestyle="--", linewidth=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    savefig(filename)

def corr_heatmap(df_, cols, title, filename):
    d = df_[cols].copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna()

    corr = d.corr(numeric_only=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(corr.values, cmap="Greys", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title(title)
    savefig(filename)


# Raw columns (for distributions/boxplots)
tone_raw = _pick_col(df, ["tone"])
unc_raw  = _pick_col(df, ["uncertainty_rate"])
cr_raw   = _pick_col(df, ["CR_0_120", "CR_0_30", "CR_0_5"])  # prefer 0_120

# Z-scored columns (for relationships/heatmap)
tone_z = _pick_col(df, ["tone_z"])
unc_z  = _pick_col(df, ["uncertainty_rate_z"])
qa_z   = _pick_col(df, ["qa_ratio_z"])
cr_z   = _pick_col(df, ["CR_0_120_z", "CR_0_30_z", "CR_0_5_z"])
rv_z   = _pick_col(df, ["RV_0_120_z", "RV_0_30_z", "RV_pre_z"])
volr_z = _pick_col(df, ["VOL_ratio_z"])

# -----------------------
# 1) Histograms (RAW)
# -----------------------
hist_plot(df[tone_raw], "Tone Distribution (Raw)", tone_raw, "hist_tone_raw.png", bins=18)
hist_plot(df[unc_raw], "Uncertainty Rate Distribution (Raw)", unc_raw, "hist_uncertainty_raw.png", bins=18)
hist_plot(df[cr_raw], f"Cumulative Log Return Distribution (Raw) [{cr_raw}]", cr_raw, "hist_return_raw.png", bins=18)

# -----------------------
# 2) Boxplots by ticker (RAW)
# -----------------------
boxplot_by_ticker(df, tone_raw, "Tone by Ticker (Raw)", tone_raw, "box_tone_by_ticker.png")
boxplot_by_ticker(df, cr_raw, f"Return by Ticker (Raw) [{cr_raw}]", cr_raw, "box_return_by_ticker.png")

# -----------------------
# 3) Scatter + regression line (Z-SCORED)
# -----------------------
scatter_with_fit(
    df, tone_z, cr_z,
    f"Tone vs Return (Z-scored): {tone_z} vs {cr_z}",
    tone_z, cr_z,
    "scatter_tone_return_z.png"
)

scatter_with_fit(
    df, unc_z, rv_z,
    f"Uncertainty vs Volatility (Z-scored): {unc_z} vs {rv_z}",
    unc_z, rv_z,
    "scatter_uncertainty_volatility_z.png"
)

scatter_with_fit(
    df, qa_z, volr_z,
    f"Q&A Share vs Volume Ratio (Z-scored): {qa_z} vs {volr_z}",
    qa_z, volr_z,
    "scatter_qa_volume_z.png"
)

# -----------------------
# 4) Correlation heatmap (Z-SCORED)
# -----------------------
heat_cols = [tone_z, unc_z, qa_z, cr_z, rv_z, volr_z]
# ensure uniqueness
heat_cols = list(dict.fromkeys(heat_cols))

corr_heatmap(df, heat_cols, "Correlation Heatmap (Z-scored variables)", "heatmap_corr_z.png")

print("\nDone. All figures saved to:", os.path.abspath(OUT_DIR))
print("Files:", sorted(os.listdir(OUT_DIR)))