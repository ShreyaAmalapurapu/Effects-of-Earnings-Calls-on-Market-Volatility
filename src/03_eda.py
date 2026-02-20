#!/usr/bin/env python3
"""
An extra file for computing metrics forExploratory Data Analysis
03_eda.py — Exploratory Data Analysis Computations
Computes per-event and overall EDA metrics

Outputs:
    data_final/eda_market_metrics.csv    (per-event market metrics)
    data_final/eda_transcript_metrics.csv (per-event transcript metrics)
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

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data_raw"
FINAL_DIR = ROOT / "data_final"
MARKET_CLEAN = RAW_DIR / "market_clean"
TRANSCRIPT_CLEAN = RAW_DIR / "transcripts_clean"
PLOT_DIR = ROOT / "notebooks" / "eda_plots"

FINAL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

EASTERN_TZ = "US/Eastern"


# 3.1  PER-EVENT METRICS
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

# MAIN
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

