#!/usr/bin/env python3
"""
02_clean.py — Data Cleaning & Handling Inconsistencies
Reads raw data from data_raw/, applies cleaning rules, and writes
cleaned data to data_raw/ (cleaned copies) with quality flags.

Steps:
  2.1  Timestamp & timezone normalization
  2.2  Market data cleaning (dedup, outlier flagging, etc.)
  2.3  Transcript cleaning (whitespace, encoding, speaker tags)
  2.4  Missing values plan (forward-fill, flagging)

Outputs:
    data_raw/market_clean/{event_id}.parquet  (cleaned 1-sec bars)
    data_raw/transcripts_clean/{event_id}.json (cleaned transcripts)
    data_raw/events_clean.csv  (updated spine with quality flags)
"""

import os
import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data_raw"
MARKET_RAW = RAW_DIR / "market_raw"
TRANSCRIPT_RAW = RAW_DIR / "transcripts_raw"
MARKET_CLEAN = RAW_DIR / "market_clean"
TRANSCRIPT_CLEAN = RAW_DIR / "transcripts_clean"

MARKET_CLEAN.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_CLEAN.mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────
EASTERN_TZ = "US/Eastern"
PRE_WINDOW_MIN = 60    # minutes before call start
POST_WINDOW_MIN = 120  # minutes after call start
FFILL_THRESHOLD_SEC = 5  # forward-fill gaps ≤ this many seconds
OUTLIER_MAD_K = 10      # flag returns with |r| > k * MAD


# 2.1  TIMESTAMP & TIMEZONE NORMALIZATION
def normalize_timestamps_market(df: pd.DataFrame, call_start_ts: pd.Timestamp):
    """
    Convert market timestamps to US/Eastern, enforce 3-hour window,
    compute quality flags.
    """
    # Parse and localize
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts_event"], utc=True)
    df["ts_eastern"] = df["ts"].dt.tz_convert(EASTERN_TZ)

    # Call start in Eastern
    if call_start_ts.tzinfo is None:
        call_start_ts = call_start_ts.tz_localize("UTC")
    call_start_et = call_start_ts.tz_convert(EASTERN_TZ)

    # Window bounds
    win_start = call_start_et - pd.Timedelta(minutes=PRE_WINDOW_MIN)
    win_end = call_start_et + pd.Timedelta(minutes=POST_WINDOW_MIN)

    n_before = len(df)
    df = df[(df["ts_eastern"] >= win_start) & (df["ts_eastern"] <= win_end)].copy()
    n_after = len(df)

    # Relative time in minutes from call start
    df["minutes_from_start"] = (
        (df["ts_eastern"] - call_start_et).dt.total_seconds() / 60.0
    )

    # Quality flags
    flags = {}
    flags["rows_total"] = n_before
    flags["rows_in_window"] = n_after
    flags["rows_dropped_outside_window"] = n_before - n_after

    # Check for duplicated timestamps
    dup_mask = df["ts"].duplicated(keep=False)
    flags["n_duplicate_ts"] = int(dup_mask.sum())

    # Check for out-of-order timestamps
    ts_sorted = df["ts"].is_monotonic_increasing
    flags["ts_out_of_order"] = not ts_sorted

    # Check coverage: first 5 min after call start
    post_start = df[df["minutes_from_start"] >= 0]
    if len(post_start) > 0:
        first_post_ts = post_start["ts_eastern"].iloc[0]
        gap_at_start_sec = (first_post_ts - call_start_et).total_seconds()
        flags["gap_at_call_start_sec"] = float(gap_at_start_sec)
        flags["missing_first_5min"] = gap_at_start_sec > 300
    else:
        flags["gap_at_call_start_sec"] = float("inf")
        flags["missing_first_5min"] = True

    return df, flags


# 2.2  MARKET DATA CLEANING
def clean_market_data(df: pd.DataFrame):
    """
    Apply market data cleaning rules:
      - Remove negative prices/volumes
      - Sort by timestamp
      - Deduplicate (same ts → aggregate)
      - Compute returns and flag outliers
    """
    df = df.copy()
    n0 = len(df)

    # Remove negative prices/volumes
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if col in df.columns:
            df = df[df[col] > 0]
    if "volume" in df.columns:
        df = df[df["volume"] >= 0]

    n_neg_removed = n0 - len(df)

    # Sort by timestamp
    df = df.sort_values("ts").reset_index(drop=True)

    # Deduplicate: aggregate same-second bars
    if df["ts"].duplicated().any():
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        # Keep other columns from first occurrence
        extra_cols = {c: "first" for c in df.columns
                      if c not in list(agg_dict.keys()) + ["ts"]}
        agg_dict.update(extra_cols)
        df = df.groupby("ts", as_index=False).agg(agg_dict)
        df = df.sort_values("ts").reset_index(drop=True)

    n_after_dedup = len(df)

    # Compute log returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["log_return"] = df["log_return"].replace([np.inf, -np.inf], np.nan)

    # Outlier flagging using MAD (Median Absolute Deviation)
    returns = df["log_return"].dropna()
    if len(returns) > 10:
        median_r = returns.median()
        mad = np.median(np.abs(returns - median_r))
        if mad > 0:
            df["outlier_flag"] = (
                np.abs(df["log_return"] - median_r) > OUTLIER_MAD_K * mad
            ).astype(int)
        else:
            df["outlier_flag"] = 0
    else:
        df["outlier_flag"] = 0

    # Winsorize returns for extreme outliers (keep original, add winsorized column)
    if len(returns) > 10:
        p01 = returns.quantile(0.01)
        p99 = returns.quantile(0.99)
        df["log_return_winsorized"] = df["log_return"].clip(lower=p01, upper=p99)
    else:
        df["log_return_winsorized"] = df["log_return"]

    flags = {
        "n_neg_removed": n_neg_removed,
        "n_after_dedup": n_after_dedup,
        "n_outliers_flagged": int(df["outlier_flag"].sum()),
    }

    return df, flags


def forward_fill_gaps(df: pd.DataFrame):
    """
    Create a complete 1-second time series and forward-fill short gaps.
    Track total gap seconds.
    """
    if len(df) == 0:
        return df, 0

    # Create complete 1-second index
    full_idx = pd.date_range(
        start=df["ts"].min(),
        end=df["ts"].max(),
        freq="1s",
        tz="UTC",
    )

    df_full = pd.DataFrame({"ts": full_idx})
    df_full = df_full.merge(df, on="ts", how="left")

    # Count gap seconds
    gap_mask = df_full["close"].isna()
    gap_seconds_total = int(gap_mask.sum())

    # Forward-fill price for short gaps (≤ threshold)
    # Identify gap runs
    gap_runs = gap_mask.astype(int).groupby((~gap_mask).cumsum()).cumsum()
    short_gap_mask = gap_mask & (gap_runs <= FFILL_THRESHOLD_SEC)

    # Forward-fill only short gaps for price columns
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if col in df_full.columns:
            # Only fill where gap is short
            filled = df_full[col].ffill()
            df_full.loc[short_gap_mask, col] = filled.loc[short_gap_mask]

    # Volume for gaps = 0
    if "volume" in df_full.columns:
        df_full["volume"] = df_full["volume"].fillna(0)

    # Forward-fill other metadata columns
    for col in ["ts_eastern", "minutes_from_start", "symbol"]:
        if col in df_full.columns:
            df_full[col] = df_full[col].ffill()

    # Recompute ts_eastern for the full index
    df_full["ts_eastern"] = df_full["ts"].dt.tz_convert(EASTERN_TZ)

    return df_full, gap_seconds_total


# 2.3  TRANSCRIPT CLEANING
def clean_transcript(record: dict) -> dict:
    """
    Clean transcript record:
      - Normalize whitespace
      - Remove encoding artifacts
      - Standardize speaker tags
      - Remove boilerplate
    """
    record = record.copy()

    for key in ["raw_text", "presentation_text", "qa_text"]:
        if key in record and record[key]:
            text = record[key]
            # Normalize whitespace
            text = re.sub(r"\r\n", "\n", text)
            text = re.sub(r"\r", "\n", text)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)

            # Remove encoding artifacts
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
            text = text.replace("\ufffd", "")
            text = text.replace("[pic]", "")

            # Remove table formatting artifacts (pipe-delimited lines)
            text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
            text = re.sub(r"\n{3,}", "\n\n", text)

            text = text.strip()
            record[key] = text

    # Standardize speaker labels in turns
    speaker_map = {
        "operator": "Operator",
    }
    if "speaker_turns" in record:
        for turn in record["speaker_turns"]:
            s = turn.get("speaker", "Unknown")
            turn["speaker"] = speaker_map.get(s.lower(), s)

            # Clean turn text
            turn_text = turn.get("text", "")
            turn_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", turn_text)
            turn_text = re.sub(r"\[pic\]", "", turn_text)
            turn_text = re.sub(r"[ \t]+", " ", turn_text)
            turn["text"] = turn_text.strip()

            # Categorize role
            role = turn.get("role", "")
            if "Operator" in s or s == "Operator":
                turn["role_category"] = "Operator"
            elif any(kw in role for kw in ["CEO", "CFO", "COO", "CTO", "VP",
                                            "President", "Officer", "Director",
                                            "Relations"]):
                turn["role_category"] = "Executive"
            elif any(kw in role for kw in ["Research", "Analyst", "Securities",
                                            "Capital", "Morgan", "Goldman",
                                            "JPMorgan", "Bank", "Citigroup",
                                            "Barclays", "UBS", "Evercore",
                                            "Bernstein", "Jefferies", "Mizuho",
                                            "BofA", "Wells", "Piper", "Truist",
                                            "KeyBanc", "Stifel", "Needham",
                                            "Raymond", "Wolfe", "Loop",
                                            "Wedbush", "Canaccord"]):
                turn["role_category"] = "Analyst"
            else:
                turn["role_category"] = "Unknown"

    # If missing speaker info
    if "speaker_turns" in record:
        for turn in record["speaker_turns"]:
            if not turn.get("speaker"):
                turn["speaker"] = "Unknown"

    # If missing Q&A
    if not record.get("qa_text"):
        record["qa_text"] = ""
        record["has_qa"] = 0

    return record


# 2.4  MAIN PIPELINE
def run():
    """Main cleaning pipeline."""
    events_path = RAW_DIR / "events.csv"
    if not events_path.exists():
        print("ERROR: events.csv not found. Run 01_acquire.py first.")
        return

    df_events = pd.read_csv(events_path)
    quality_records = []

    for _, row in df_events.iterrows():
        event_id = row["event_id"]
        ticker = row["ticker"]
        print(f"\n{'─'*50}")
        print(f"Cleaning {event_id}")

        # ── Parse call start timestamp ───────────────────────────────
        call_ts_str = row.get("call_start_ts", "")
        if pd.isna(call_ts_str) or not call_ts_str:
            print(f"  WARNING: No call_start_ts for {event_id}, skipping")
            continue
        try:
            call_start_ts = pd.Timestamp(call_ts_str)
            if call_start_ts.tzinfo is None:
                call_start_ts = call_start_ts.tz_localize("UTC")
        except Exception as e:
            print(f"  WARNING: Bad timestamp '{call_ts_str}': {e}")
            continue

        # ── Clean market data ────────────────────────────────────────
        mkt_path = MARKET_RAW / f"{event_id}.csv"
        mkt_flags = {}
        if mkt_path.exists():
            df_mkt = pd.read_csv(mkt_path)
            print(f"  Market: {len(df_mkt)} raw rows")

            # 2.1 Normalize timestamps + window
            df_mkt, ts_flags = normalize_timestamps_market(df_mkt, call_start_ts)
            print(f"  After window filter: {len(df_mkt)} rows")
            mkt_flags.update(ts_flags)

            # 2.2 Clean
            df_mkt, clean_flags = clean_market_data(df_mkt)
            print(f"  After cleaning: {len(df_mkt)} rows, "
                  f"{clean_flags['n_outliers_flagged']} outliers flagged")
            mkt_flags.update(clean_flags)

            # 2.4 Forward-fill gaps
            df_mkt, gap_sec = forward_fill_gaps(df_mkt)
            mkt_flags["gap_seconds_total"] = gap_sec
            print(f"  Gap seconds total: {gap_sec}")

            # Save cleaned market data
            out_path = MARKET_CLEAN / f"{event_id}.parquet"
            # Drop timezone info for parquet compatibility (store as UTC)
            df_save = df_mkt.copy()
            if "ts" in df_save.columns:
                df_save["ts"] = df_save["ts"].dt.tz_localize(None)
            if "ts_eastern" in df_save.columns:
                df_save["ts_eastern"] = df_save["ts_eastern"].dt.tz_localize(None)
            df_save.to_parquet(out_path, index=False)
            print(f"  Saved → {out_path.name}")
        else:
            print(f"  WARNING: No market data for {event_id}")

        # ── Clean transcript ─────────────────────────────────────────
        tx_path = TRANSCRIPT_RAW / f"{event_id}.json"
        tx_flags = {}
        if tx_path.exists():
            with open(tx_path, "r", encoding="utf-8") as f:
                tx_record = json.load(f)

            tx_clean = clean_transcript(tx_record)

            # Save cleaned transcript
            out_tx = TRANSCRIPT_CLEAN / f"{event_id}.json"
            with open(out_tx, "w", encoding="utf-8") as f:
                json.dump(tx_clean, f, indent=2, ensure_ascii=False)
            print(f"  Saved → {out_tx.name}")

            tx_flags["has_qa"] = tx_clean.get("has_qa", 0)
            tx_flags["n_speaker_turns"] = len(tx_clean.get("speaker_turns", []))
        else:
            print(f"  WARNING: No transcript for {event_id}")

        # ── Collect quality record ───────────────────────────────────
        quality_records.append({
            "event_id": event_id,
            "ticker": ticker,
            **{f"mkt_{k}": v for k, v in mkt_flags.items()},
            **{f"tx_{k}": v for k, v in tx_flags.items()},
        })

    # ── Save updated events with quality flags ───────────────────────
    df_quality = pd.DataFrame(quality_records)
    df_events_clean = df_events.merge(df_quality, on=["event_id", "ticker"], how="left")
    out_events = RAW_DIR / "events_clean.csv"
    df_events_clean.to_csv(out_events, index=False)
    print(f"\n{'='*60}")
    print(f"Clean events saved → {out_events}")
    print(f"Total events processed: {len(df_events_clean)}")

    # Print summary of quality issues
    print(f"\n{'='*60}")
    print("QUALITY SUMMARY")
    print(f"{'='*60}")
    if "mkt_n_duplicate_ts" in df_events_clean.columns:
        n_dup = (df_events_clean["mkt_n_duplicate_ts"] > 0).sum()
        print(f"  Events with duplicate timestamps:  {n_dup}")
    if "mkt_ts_out_of_order" in df_events_clean.columns:
        n_ooo = df_events_clean["mkt_ts_out_of_order"].sum()
        print(f"  Events with out-of-order ts:       {n_ooo}")
    if "mkt_missing_first_5min" in df_events_clean.columns:
        n_miss = df_events_clean["mkt_missing_first_5min"].sum()
        print(f"  Events missing first 5min data:    {n_miss}")
    if "mkt_n_outliers_flagged" in df_events_clean.columns:
        tot_out = df_events_clean["mkt_n_outliers_flagged"].sum()
        print(f"  Total outlier ticks flagged:        {tot_out}")
    if "tx_has_qa" in df_events_clean.columns:
        n_no_qa = (df_events_clean["tx_has_qa"] == 0).sum()
        print(f"  Events missing Q&A section:        {n_no_qa}")

    return df_events_clean


if __name__ == "__main__":
    run()

