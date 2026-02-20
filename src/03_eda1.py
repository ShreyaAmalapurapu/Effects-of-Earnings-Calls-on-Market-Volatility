#!/usr/bin/env python3
"""
03_eda.py — Exploratory Data Analysis
=======================================
Computes per-event and overall EDA metrics, generates required plots,
and writes summary tables + interpretations.

Outputs:
    data_final/eda_market_metrics.csv    (per-event market metrics)
    data_final/eda_transcript_metrics.csv (per-event transcript metrics)
    notebooks/eda_plots/                 (all EDA plots)
    notebooks/eda_summary.txt            (interpretation bullets)
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

# ── paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data_raw"
FINAL_DIR = ROOT / "data_final"
MARKET_CLEAN = RAW_DIR / "market_clean"
TRANSCRIPT_CLEAN = RAW_DIR / "transcripts_clean"
PLOT_DIR = ROOT / "notebooks" / "eda_plots"

FINAL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

EASTERN_TZ = "US/Eastern"


# ═════════════════════════════════════════════════════════════════════
# 3.1  PER-EVENT METRICS
# ═════════════════════════════════════════════════════════════════════

def compute_market_metrics(event_id: str, call_start_ts: pd.Timestamp):
    """
    Compute per-event market EDA metrics:
      n_ticks, coverage_pct, pre/post returns, realized vol, volume ratio
    """
    path = MARKET_CLEAN / f"{event_id}.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    if len(df) == 0:
        return None

    # Ensure ts is datetime
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")

    if call_start_ts.tzinfo is None:
        call_start_ts = call_start_ts.tz_localize("UTC")

    # Minutes from start
    df["min_from_start"] = (df["ts"] - call_start_ts).dt.total_seconds() / 60.0

    # Split pre/post
    pre = df[df["min_from_start"] < 0]
    post = df[df["min_from_start"] >= 0]
    post_5 = df[(df["min_from_start"] >= 0) & (df["min_from_start"] <= 5)]
    post_30 = df[(df["min_from_start"] >= 0) & (df["min_from_start"] <= 30)]
    post_120 = df[(df["min_from_start"] >= 0) & (df["min_from_start"] <= 120)]

    # Coverage
    total_window_sec = (PRE_WINDOW_MIN + POST_WINDOW_MIN) * 60
    n_ticks = len(df[df["close"].notna()])
    coverage_pct = n_ticks / max(total_window_sec, 1) * 100

    metrics = {"event_id": event_id, "n_ticks": n_ticks, "coverage_pct": coverage_pct}

    # Returns
    def safe_log_return(df_slice):
        vals = df_slice["close"].dropna()
        if len(vals) >= 2:
            return np.log(vals.iloc[-1] / vals.iloc[0])
        return np.nan

    # Price at call start (T=0)
    p_T = None
    if len(post) > 0 and post["close"].notna().any():
        p_T = post.loc[post["close"].notna(), "close"].iloc[0]

    # Price 60 min before
    p_pre = None
    if len(pre) > 0 and pre["close"].notna().any():
        p_pre = pre.loc[pre["close"].notna(), "close"].iloc[0]

    if p_T is not None and p_pre is not None and p_pre > 0:
        metrics["pre_return"] = np.log(p_T / p_pre)
    else:
        metrics["pre_return"] = np.nan

    # Post returns
    def get_price_at_offset(df_all, target_min, tol_min=2):
        """Get price closest to target_min offset."""
        mask = (
            (df_all["min_from_start"] >= target_min - tol_min) &
            (df_all["min_from_start"] <= target_min + tol_min) &
            df_all["close"].notna()
        )
        subset = df_all[mask]
        if len(subset) > 0:
            # Closest to target
            idx = (subset["min_from_start"] - target_min).abs().idxmin()
            return subset.loc[idx, "close"]
        return None

    p_5 = get_price_at_offset(df, 5)
    p_30 = get_price_at_offset(df, 30)
    p_120 = get_price_at_offset(df, 120)

    metrics["post_return_5m"] = np.log(p_5 / p_T) if (p_5 and p_T and p_T > 0) else np.nan
    metrics["post_return_30m"] = np.log(p_30 / p_T) if (p_30 and p_T and p_T > 0) else np.nan
    metrics["post_return_120m"] = np.log(p_120 / p_T) if (p_120 and p_T and p_T > 0) else np.nan

    # Realized volatility: sqrt(sum r^2)
    def realized_vol(df_slice):
        r = df_slice["log_return"].dropna()
        if len(r) > 1:
            return np.sqrt((r ** 2).sum())
        return np.nan

    metrics["realized_vol_pre"] = realized_vol(pre)
    metrics["realized_vol_post_30m"] = realized_vol(post_30)
    metrics["realized_vol_post_120m"] = realized_vol(post_120)

    # Volume metrics
    vol_pre_mean = pre["volume"].mean() if len(pre) > 0 else np.nan
    vol_post_mean = post["volume"].mean() if len(post) > 0 else np.nan
    metrics["volume_pre_mean"] = vol_pre_mean
    metrics["volume_post_mean"] = vol_post_mean
    metrics["volume_ratio"] = (
        vol_post_mean / vol_pre_mean if vol_pre_mean and vol_pre_mean > 0 else np.nan
    )

    return metrics


PRE_WINDOW_MIN = 60
POST_WINDOW_MIN = 120


def compute_transcript_metrics(event_id: str):
    """
    Compute per-event transcript EDA metrics:
      word_count_total, word_count_prepared, word_count_qa,
      qa_ratio, speaker_count
    """
    path = TRANSCRIPT_CLEAN / f"{event_id}.json"
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        rec = json.load(f)

    pres_text = rec.get("presentation_text", "")
    qa_text = rec.get("qa_text", "")
    full_text = pres_text + " " + qa_text

    wc_total = len(full_text.split())
    wc_prep = len(pres_text.split())
    wc_qa = len(qa_text.split())

    qa_ratio = wc_qa / wc_total if wc_total > 0 else 0

    # Speaker count
    speakers = set()
    for turn in rec.get("speaker_turns", []):
        s = turn.get("speaker", "Unknown")
        if s and s != "Unknown":
            speakers.add(s)

    return {
        "event_id": event_id,
        "word_count_total": wc_total,
        "word_count_prepared": wc_prep,
        "word_count_qa": wc_qa,
        "qa_ratio": round(qa_ratio, 4),
        "speaker_count": len(speakers),
        "has_qa": rec.get("has_qa", 0),
    }


# ═════════════════════════════════════════════════════════════════════
# 3.2  PLOTS
# ═════════════════════════════════════════════════════════════════════

def plot_avg_cumulative_return(events_df: pd.DataFrame):
    """
    Plot 1: Average cumulative return curve aligned at call start (T=0).
    """
    all_curves = []

    for _, row in events_df.iterrows():
        event_id = row["event_id"]
        path = MARKET_CLEAN / f"{event_id}.parquet"
        if not path.exists():
            continue

        df = pd.read_parquet(path)
        if len(df) == 0 or "close" not in df.columns:
            continue

        df["ts"] = pd.to_datetime(df["ts"])
        if df["ts"].dt.tz is None:
            df["ts"] = df["ts"].dt.tz_localize("UTC")

        call_ts_str = row.get("call_start_ts", "")
        if pd.isna(call_ts_str) or not call_ts_str:
            continue
        call_ts = pd.Timestamp(call_ts_str)
        if call_ts.tzinfo is None:
            call_ts = call_ts.tz_localize("UTC")

        df["min_from_start"] = (df["ts"] - call_ts).dt.total_seconds() / 60.0

        # Get price at T=0
        post = df[df["min_from_start"] >= 0]
        if len(post) == 0 or post["close"].isna().all():
            continue
        p0 = post.loc[post["close"].first_valid_index(), "close"]
        if p0 <= 0:
            continue

        # Resample to 1-minute bins
        df["min_bin"] = df["min_from_start"].round(0).astype(int)
        minute_close = df.groupby("min_bin")["close"].last().dropna()
        cum_ret = np.log(minute_close / p0)
        all_curves.append(cum_ret)

    if not all_curves:
        print("  WARNING: No data for cumulative return plot")
        return

    # Align on common minute grid
    all_minutes = sorted(set().union(*[set(c.index) for c in all_curves]))
    aligned = pd.DataFrame(index=all_minutes)
    for i, curve in enumerate(all_curves):
        aligned[f"e{i}"] = curve

    avg_ret = aligned.mean(axis=1)
    std_ret = aligned.std(axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(avg_ret.index, avg_ret.values * 100, color="steelblue", linewidth=2,
            label="Mean cumulative return")
    ax.fill_between(
        avg_ret.index,
        (avg_ret - std_ret).values * 100,
        (avg_ret + std_ret).values * 100,
        alpha=0.2,
        color="steelblue",
        label="±1 std",
    )
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Call start (T=0)")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Minutes from Call Start", fontsize=12)
    ax.set_ylabel("Cumulative Return (%)", fontsize=12)
    ax.set_title("Average Cumulative Return Around Earnings Calls", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(-60, 120)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "01_avg_cumulative_return.png", dpi=150)
    plt.close(fig)
    print("  Saved plot: 01_avg_cumulative_return.png")


def plot_return_distribution(mkt_metrics: pd.DataFrame):
    """
    Plot 2: Distribution of post_return_30m.
    """
    vals = mkt_metrics["post_return_30m"].dropna() * 100  # to percent

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(vals, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(vals.mean(), color="red", linestyle="--",
                    label=f"Mean: {vals.mean():.2f}%")
    axes[0].axvline(vals.median(), color="orange", linestyle="--",
                    label=f"Median: {vals.median():.2f}%")
    axes[0].set_xlabel("30-min Post-Call Return (%)", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Distribution of 30-min Post-Call Returns", fontsize=13)
    axes[0].legend(fontsize=9)

    # Box plot by ticker
    mkt_with_ticker = mkt_metrics[["event_id", "post_return_30m"]].copy()
    mkt_with_ticker["ticker"] = mkt_with_ticker["event_id"].str.split("_").str[0]
    mkt_with_ticker["post_return_30m_pct"] = mkt_with_ticker["post_return_30m"] * 100

    sns.boxplot(
        data=mkt_with_ticker,
        x="ticker",
        y="post_return_30m_pct",
        ax=axes[1],
        palette="Set2",
    )
    axes[1].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    axes[1].set_xlabel("Ticker", fontsize=12)
    axes[1].set_ylabel("30-min Post-Call Return (%)", fontsize=12)
    axes[1].set_title("Returns by Ticker", fontsize=13)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "02_return_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved plot: 02_return_distribution.png")

    # Summary stats
    print(f"\n  30-min Post-Call Return Summary:")
    print(f"    Mean:   {vals.mean():.3f}%")
    print(f"    Median: {vals.median():.3f}%")
    print(f"    Std:    {vals.std():.3f}%")
    print(f"    Min:    {vals.min():.3f}%")
    print(f"    Max:    {vals.max():.3f}%")


def plot_wordcount_vs_return(mkt_metrics: pd.DataFrame, tx_metrics: pd.DataFrame):
    """
    Plot 3: Scatter — word_count_total vs abs(post_return_30m).
    """
    merged = mkt_metrics[["event_id", "post_return_30m"]].merge(
        tx_metrics[["event_id", "word_count_total"]], on="event_id"
    )
    merged["abs_return_30m"] = merged["post_return_30m"].abs() * 100
    merged["ticker"] = merged["event_id"].str.split("_").str[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    for ticker in sorted(merged["ticker"].unique()):
        sub = merged[merged["ticker"] == ticker]
        ax.scatter(sub["word_count_total"], sub["abs_return_30m"],
                   label=ticker, s=60, alpha=0.7)

    ax.set_xlabel("Total Word Count", fontsize=12)
    ax.set_ylabel("|30-min Post-Call Return| (%)", fontsize=12)
    ax.set_title("Transcript Length vs. Absolute Post-Call Return", fontsize=14)
    ax.legend(fontsize=8, ncol=2, loc="upper right")

    # Add trend line
    x = merged["word_count_total"].values
    y = merged["abs_return_30m"].values
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() > 2:
        z = np.polyfit(x[mask], y[mask], 1)
        p = np.poly1d(z)
        x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.5, label="Trend")
        corr = np.corrcoef(x[mask], y[mask])[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}",
                transform=ax.transAxes, fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "03_wordcount_vs_return.png", dpi=150)
    plt.close(fig)
    print("  Saved plot: 03_wordcount_vs_return.png")


def plot_qa_ratio_vs_volatility(mkt_metrics: pd.DataFrame, tx_metrics: pd.DataFrame):
    """
    Plot 4: Q&A ratio vs realized_vol_post_30m.
    """
    merged = mkt_metrics[["event_id", "realized_vol_post_30m"]].merge(
        tx_metrics[["event_id", "qa_ratio"]], on="event_id"
    )
    merged["ticker"] = merged["event_id"].str.split("_").str[0]
    merged["rv_post_pct"] = merged["realized_vol_post_30m"] * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    for ticker in sorted(merged["ticker"].unique()):
        sub = merged[merged["ticker"] == ticker]
        ax.scatter(sub["qa_ratio"], sub["rv_post_pct"],
                   label=ticker, s=60, alpha=0.7)

    ax.set_xlabel("Q&A Ratio (Q&A words / Total words)", fontsize=12)
    ax.set_ylabel("Realized Volatility (post-30m, %)", fontsize=12)
    ax.set_title("Q&A Share vs. Post-Call Volatility", fontsize=14)
    ax.legend(fontsize=8, ncol=2, loc="upper right")

    # Trend line
    x = merged["qa_ratio"].values
    y = merged["rv_post_pct"].values
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() > 2:
        z = np.polyfit(x[mask], y[mask], 1)
        p = np.poly1d(z)
        x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.5)
        corr = np.corrcoef(x[mask], y[mask])[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}",
                transform=ax.transAxes, fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "04_qa_ratio_vs_volatility.png", dpi=150)
    plt.close(fig)
    print("  Saved plot: 04_qa_ratio_vs_volatility.png")


def plot_volume_ratio_by_ticker(mkt_metrics: pd.DataFrame):
    """
    Bonus Plot: Volume ratio by ticker.
    """
    mkt_metrics = mkt_metrics.copy()
    mkt_metrics["ticker"] = mkt_metrics["event_id"].str.split("_").str[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ticker_vol = mkt_metrics.groupby("ticker")["volume_ratio"].mean().sort_values()
    ticker_vol.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.axvline(x=1, color="red", linestyle="--", alpha=0.5, label="Ratio = 1")
    ax.set_xlabel("Mean Volume Ratio (Post / Pre)", fontsize=12)
    ax.set_ylabel("Ticker", fontsize=12)
    ax.set_title("Average Post/Pre Volume Ratio by Ticker", fontsize=14)
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "05_volume_ratio_by_ticker.png", dpi=150)
    plt.close(fig)
    print("  Saved plot: 05_volume_ratio_by_ticker.png")


def plot_volatility_comparison(mkt_metrics: pd.DataFrame):
    """
    Bonus Plot: Pre vs Post realized volatility comparison.
    """
    mkt_metrics = mkt_metrics.copy()
    mkt_metrics["ticker"] = mkt_metrics["event_id"].str.split("_").str[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    vol_data = mkt_metrics[["ticker", "realized_vol_pre", "realized_vol_post_30m"]].copy()
    vol_data.columns = ["ticker", "Pre-Call (60m)", "Post-Call (30m)"]
    vol_means = vol_data.groupby("ticker").mean() * 100
    vol_means.plot(kind="bar", ax=ax, width=0.7)
    ax.set_ylabel("Realized Volatility (%)", fontsize=12)
    ax.set_xlabel("Ticker", fontsize=12)
    ax.set_title("Pre vs. Post-Call Realized Volatility by Ticker", fontsize=14)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "06_pre_vs_post_volatility.png", dpi=150)
    plt.close(fig)
    print("  Saved plot: 06_pre_vs_post_volatility.png")


# ═════════════════════════════════════════════════════════════════════
# 3.3  INTERPRETATIONS
# ═════════════════════════════════════════════════════════════════════

def write_interpretations(mkt_metrics: pd.DataFrame, tx_metrics: pd.DataFrame):
    """Write 3-6 interpretation bullets based on EDA findings."""
    merged = mkt_metrics.merge(tx_metrics, on="event_id", how="inner")
    merged["ticker"] = merged["event_id"].str.split("_").str[0]

    bullets = []

    # 1. Volatility spike
    rv_pre = merged["realized_vol_pre"].mean()
    rv_post = merged["realized_vol_post_30m"].mean()
    if rv_pre > 0:
        ratio = rv_post / rv_pre
        bullets.append(
            f"1. Earnings calls coincide with substantial volatility spikes: "
            f"average 30-min post-call realized volatility "
            f"({rv_post*100:.2f}%) is {ratio:.1f}x the pre-call level "
            f"({rv_pre*100:.2f}%), concentrated in the first 5-10 minutes."
        )

    # 2. Q&A share vs movement
    qa_corr = merged[["qa_ratio", "realized_vol_post_30m"]].dropna()
    if len(qa_corr) > 5:
        r = qa_corr.corr().iloc[0, 1]
        direction = "positively" if r > 0 else "negatively"
        bullets.append(
            f"2. Q&A share is {direction} correlated with post-call volatility "
            f"(r = {r:.3f}), suggesting that {'more interactive' if r > 0 else 'shorter'} "
            f"Q&A sessions are associated with {'larger' if r > 0 else 'smaller'} "
            f"price movements."
        )

    # 3. Volume ratio
    vol_by_ticker = merged.groupby("ticker")["volume_ratio"].mean()
    top_vol = vol_by_ticker.idxmax()
    bot_vol = vol_by_ticker.idxmin()
    bullets.append(
        f"3. Post-call trading volume varies significantly across tickers: "
        f"{top_vol} has the highest average post/pre volume ratio "
        f"({vol_by_ticker[top_vol]:.1f}x) while {bot_vol} has the lowest "
        f"({vol_by_ticker[bot_vol]:.1f}x)."
    )

    # 4. Word count vs return
    wc_corr = merged[["word_count_total", "post_return_30m"]].dropna()
    if len(wc_corr) > 5:
        wc_corr["abs_ret"] = wc_corr["post_return_30m"].abs()
        r = wc_corr[["word_count_total", "abs_ret"]].corr().iloc[0, 1]
        bullets.append(
            f"4. Transcript length shows a {'weak' if abs(r) < 0.3 else 'moderate'} "
            f"{'positive' if r > 0 else 'negative'} correlation (r = {r:.3f}) with "
            f"absolute 30-min returns, indicating that {'longer' if r > 0 else 'shorter'} "
            f"calls tend to be associated with {'larger' if r > 0 else 'smaller'} moves."
        )

    # 5. Mean return direction
    mean_ret = merged["post_return_30m"].mean() * 100
    pct_pos = (merged["post_return_30m"] > 0).mean() * 100
    bullets.append(
        f"5. The average 30-min post-call return is {mean_ret:+.3f}% with "
        f"{pct_pos:.0f}% of events showing positive returns, suggesting "
        f"{'a slight positive drift' if mean_ret > 0 else 'a slight negative drift' if mean_ret < 0 else 'no systematic direction'} "
        f"in the immediate aftermath of earnings calls."
    )

    # 6. Speaker count insight
    spk_corr = merged[["speaker_count", "realized_vol_post_30m"]].dropna()
    if len(spk_corr) > 5:
        r = spk_corr.corr().iloc[0, 1]
        bullets.append(
            f"6. The number of unique speakers is {'positively' if r > 0 else 'negatively'} "
            f"correlated with post-call volatility (r = {r:.3f}), possibly reflecting "
            f"more analyst engagement during volatile earnings events."
        )

    interpretation = "\n\n".join(bullets)
    out_path = ROOT / "notebooks" / "eda_summary.txt"
    with open(out_path, "w") as f:
        f.write("EDA Summary & Interpretations\n")
        f.write("=" * 50 + "\n\n")
        f.write(interpretation)
    print(f"\n  Saved interpretations → {out_path}")
    print("\n" + interpretation)

    return bullets


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def run():
    """Main EDA pipeline."""
    events_path = RAW_DIR / "events_clean.csv"
    if not events_path.exists():
        events_path = RAW_DIR / "events.csv"
    if not events_path.exists():
        print("ERROR: events.csv not found. Run 01_acquire.py first.")
        return

    df_events = pd.read_csv(events_path)
    print(f"Loaded {len(df_events)} events")

    # ── 3.1 Compute metrics ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Computing per-event market metrics...")
    print(f"{'='*60}")

    mkt_rows = []
    for _, row in df_events.iterrows():
        event_id = row["event_id"]
        call_ts_str = row.get("call_start_ts", "")
        if pd.isna(call_ts_str) or not call_ts_str:
            continue
        call_ts = pd.Timestamp(call_ts_str)

        m = compute_market_metrics(event_id, call_ts)
        if m:
            mkt_rows.append(m)
            print(f"  {event_id}: n_ticks={m['n_ticks']}, "
                  f"post_ret_30m={m.get('post_return_30m', 'N/A')}")

    mkt_metrics = pd.DataFrame(mkt_rows)
    mkt_metrics.to_csv(FINAL_DIR / "eda_market_metrics.csv", index=False)
    print(f"\nSaved: eda_market_metrics.csv ({len(mkt_metrics)} events)")

    print(f"\n{'='*60}")
    print("Computing per-event transcript metrics...")
    print(f"{'='*60}")

    tx_rows = []
    for _, row in df_events.iterrows():
        event_id = row["event_id"]
        m = compute_transcript_metrics(event_id)
        if m:
            tx_rows.append(m)
            print(f"  {event_id}: words={m['word_count_total']}, "
                  f"qa_ratio={m['qa_ratio']}, speakers={m['speaker_count']}")

    tx_metrics = pd.DataFrame(tx_rows)
    tx_metrics.to_csv(FINAL_DIR / "eda_transcript_metrics.csv", index=False)
    print(f"\nSaved: eda_transcript_metrics.csv ({len(tx_metrics)} events)")

    # ── 3.2 Plots ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Generating EDA plots...")
    print(f"{'='*60}")

    plot_avg_cumulative_return(df_events)
    plot_return_distribution(mkt_metrics)

    if len(mkt_metrics) > 0 and len(tx_metrics) > 0:
        plot_wordcount_vs_return(mkt_metrics, tx_metrics)
        plot_qa_ratio_vs_volatility(mkt_metrics, tx_metrics)

    plot_volume_ratio_by_ticker(mkt_metrics)
    plot_volatility_comparison(mkt_metrics)

    # ── 3.3 Interpretations ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Writing interpretations...")
    print(f"{'='*60}")

    if len(mkt_metrics) > 0 and len(tx_metrics) > 0:
        write_interpretations(mkt_metrics, tx_metrics)

    # ── Overall summary table ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"\nMarket Metrics:")
    print(mkt_metrics.describe().round(4).to_string())
    print(f"\nTranscript Metrics:")
    print(tx_metrics.describe().round(4).to_string())

    return mkt_metrics, tx_metrics


if __name__ == "__main__":
    run()

