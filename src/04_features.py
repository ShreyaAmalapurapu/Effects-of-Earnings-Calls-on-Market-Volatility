#!/usr/bin/env python3
"""
04_features.py — Preprocessing & Feature Engineering
Creates standardized time bars, computes market + transcript features,
normalizes/standardizes, and saves final datasets.

Outputs:
    data_final/minute_bars.parquet         (1-min OHLCV bars, all events)
    data_final/event_level_features.csv    (1 row per call, all features)
"""

import os
import re
import json
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data_raw"
FINAL_DIR = ROOT / "data_final"
MARKET_CLEAN = RAW_DIR / "market_clean"
TRANSCRIPT_CLEAN = RAW_DIR / "transcripts_clean"

FINAL_DIR.mkdir(parents=True, exist_ok=True)

EASTERN_TZ = "US/Eastern"

# ── Loughran–McDonald Financial Sentiment Word Lists ────────────────
# Commonly used subset; full lists have ~2,700 negative, ~350 positive words.
# We use a compact but representative subset for this project.
POSITIVE_WORDS = {
    "achieve", "accomplished", "achievement", "advance", "advantage",
    "benefit", "benefited", "best", "better", "boost", "breakthrough",
    "confident", "create", "creative", "deliver", "delivered",
    "efficiency", "enable", "enhance", "excellent", "exceptional",
    "excited", "exciting", "expand", "favorable", "gain", "good",
    "great", "greatest", "grow", "growing", "growth", "improve",
    "improved", "improvement", "increase", "increased", "incredible",
    "innovation", "innovative", "leading", "momentum", "opportunity",
    "optimistic", "outperform", "outstanding", "pleased", "positive",
    "premium", "profit", "profitable", "progress", "promising",
    "record", "recover", "recovery", "remarkable", "resilient",
    "revenue", "reward", "robust", "solid", "strength", "strengthen",
    "strong", "stronger", "succeed", "success", "successful",
    "superior", "surpass", "sustainable", "transform", "transformative",
    "tremendous", "upgrade", "upturn", "win", "winning",
}

NEGATIVE_WORDS = {
    "abandon", "adverse", "challenge", "challenged", "challenging",
    "close", "closing", "concern", "concerned", "contraction",
    "costly", "damage", "decline", "declined", "decrease", "deficit",
    "delay", "delayed", "deteriorate", "difficult", "difficulty",
    "diminish", "disappoint", "disappointed", "disappointing",
    "discontinue", "disruption", "downturn", "drop", "dropped",
    "failure", "fall", "falling", "fear", "hurdle", "impair",
    "impairment", "inability", "inadequate", "inflation", "issue",
    "layoff", "liability", "litigation", "lose", "losing", "loss",
    "lower", "miss", "missed", "negative", "obstacle", "penalty",
    "problem", "recession", "reduce", "reduced", "restructuring",
    "risk", "risky", "setback", "shortage", "shrink", "slowing",
    "slowdown", "struggle", "struggling", "suffer", "tariff",
    "threat", "uncertain", "uncertainty", "underperform", "unfavorable",
    "volatile", "volatility", "warn", "warning", "weak", "weaken",
    "weakness", "worse", "worsen", "worst", "writedown", "writeoff",
}

UNCERTAINTY_WORDS = {
    "almost", "appear", "approximate", "approximately", "assume",
    "assumption", "believe", "cautious", "conceivable", "conditional",
    "contingent", "depend", "depends", "doubt", "estimate",
    "estimated", "expect", "expected", "forecast", "hope",
    "hypothetical", "if", "implicit", "indefinite", "intend",
    "likelihood", "may", "maybe", "might", "nearly", "pending",
    "perhaps", "possibility", "possible", "possibly", "potential",
    "potentially", "predict", "prediction", "preliminary",
    "probable", "probably", "project", "projected", "roughly",
    "seem", "seems", "should", "somewhat", "suggest",
    "tentative", "uncertain", "uncertainty", "unclear", "unknown",
    "unlikely", "unpredictable", "unresolved", "unsure",
    "variable", "volatility",
}


# ═════════════════════════════════════════════════════════════════════
# 4.1  STANDARDIZED TIME BARS
# ═════════════════════════════════════════════════════════════════════

def create_minute_bars(event_id: str, call_start_ts: pd.Timestamp):
    """
    Resample 1-second cleaned data into 1-minute OHLCV bars.
    Returns DataFrame with minute bars and log returns.
    """
    path = MARKET_CLEAN / f"{event_id}.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    if len(df) == 0 or "close" not in df.columns:
        return None

    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")

    if call_start_ts.tzinfo is None:
        call_start_ts = call_start_ts.tz_localize("UTC")

    # Floor to minute
    df["ts_minute"] = df["ts"].dt.floor("min")

    # Aggregate to 1-minute bars
    bars = df.groupby("ts_minute").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()

    bars = bars.rename(columns={"ts_minute": "ts"})
    bars = bars.dropna(subset=["close"])

    # Log returns
    bars["log_return"] = np.log(bars["close"] / bars["close"].shift(1))
    bars["log_return"] = bars["log_return"].replace([np.inf, -np.inf], np.nan)

    # Minutes from call start
    bars["minutes_from_start"] = (
        (bars["ts"] - call_start_ts).dt.total_seconds() / 60.0
    ).round(0).astype(int)

    bars["event_id"] = event_id

    return bars


# 4.2  MARKET FEATURES (PER EVENT)

def compute_market_features(bars: pd.DataFrame, call_start_ts: pd.Timestamp):
    """
    Compute engineered market features from minute bars.
    """
    if bars is None or len(bars) == 0:
        return {}

    # Ensure correct types
    bars = bars.copy()
    bars["ts"] = pd.to_datetime(bars["ts"])
    if bars["ts"].dt.tz is None:
        bars["ts"] = bars["ts"].dt.tz_localize("UTC")

    if call_start_ts.tzinfo is None:
        call_start_ts = call_start_ts.tz_localize("UTC")

    bars["min_offset"] = (
        (bars["ts"] - call_start_ts).dt.total_seconds() / 60.0
    )

    # Windows
    pre = bars[(bars["min_offset"] >= -60) & (bars["min_offset"] < 0)]
    post_5 = bars[(bars["min_offset"] >= 0) & (bars["min_offset"] <= 5)]
    post_30 = bars[(bars["min_offset"] >= 0) & (bars["min_offset"] <= 30)]
    post_120 = bars[(bars["min_offset"] >= 0) & (bars["min_offset"] <= 120)]
    post_5_120 = bars[(bars["min_offset"] >= 5) & (bars["min_offset"] <= 120)]

    feats = {}

    # ── Return features ──────────────────────────────────────────────
    def cum_return(df_slice):
        vals = df_slice["close"].dropna()
        if len(vals) >= 2:
            return np.log(vals.iloc[-1] / vals.iloc[0])
        return np.nan

    feats["CR_0_5"] = cum_return(post_5)
    feats["CR_0_30"] = cum_return(post_30)
    feats["CR_0_120"] = cum_return(post_120)

    # ── Volatility features ──────────────────────────────────────────
    def realized_vol(df_slice):
        r = df_slice["log_return"].dropna()
        if len(r) > 1:
            return np.sqrt((r ** 2).sum())
        return np.nan

    feats["RV_pre"] = realized_vol(pre)
    feats["RV_0_30"] = realized_vol(post_30)
    feats["RV_0_120"] = realized_vol(post_120)

    # ── Volume / Liquidity ───────────────────────────────────────────
    vol_pre = pre["volume"].mean() if len(pre) > 0 else np.nan
    vol_post = post_120["volume"].mean() if len(post_120) > 0 else np.nan
    feats["VOL_pre_mean"] = vol_pre
    feats["VOL_post_mean"] = vol_post
    feats["VOL_ratio"] = vol_post / vol_pre if vol_pre and vol_pre > 0 else np.nan

    # ── Stability / Shape ────────────────────────────────────────────
    if len(post_120) > 1:
        first_close = post_120["close"].dropna().iloc[0] if post_120["close"].notna().any() else np.nan
        if not np.isnan(first_close) and first_close > 0:
            post_120_cum = np.log(post_120["close"].dropna() / first_close)
            feats["max_runup_0_120"] = post_120_cum.max()
            feats["max_drawdown_0_120"] = post_120_cum.min()

            # Time to max absolute move
            abs_cum = post_120_cum.abs()
            if len(abs_cum) > 0:
                max_idx = abs_cum.idxmax()
                max_row = post_120.loc[max_idx]
                feats["time_to_max_abs_move"] = max_row["min_offset"]
            else:
                feats["time_to_max_abs_move"] = np.nan
        else:
            feats["max_runup_0_120"] = np.nan
            feats["max_drawdown_0_120"] = np.nan
            feats["time_to_max_abs_move"] = np.nan
    else:
        feats["max_runup_0_120"] = np.nan
        feats["max_drawdown_0_120"] = np.nan
        feats["time_to_max_abs_move"] = np.nan

    return feats


# ═════════════════════════════════════════════════════════════════════
# 4.3  TRANSCRIPT FEATURES (PER EVENT)
# ═════════════════════════════════════════════════════════════════════

def compute_transcript_features(event_id: str):
    """
    Compute interpretable transcript features:
      - Length/structure
      - Tone (financial dictionary)
      - Question count
    """
    path = TRANSCRIPT_CLEAN / f"{event_id}.json"
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        rec = json.load(f)

    pres_text = rec.get("presentation_text", "")
    qa_text = rec.get("qa_text", "")
    full_text = (pres_text + " " + qa_text).lower()
    words = re.findall(r"\b[a-z]+\b", full_text)
    word_count = len(words)

    feats = {}

    # ── Length / Structure ────────────────────────────────────────────
    feats["word_count_total"] = word_count
    pres_words = len(re.findall(r"\b[a-z]+\b", pres_text.lower()))
    qa_words = len(re.findall(r"\b[a-z]+\b", qa_text.lower()))
    feats["word_count_prepared"] = pres_words
    feats["word_count_qa"] = qa_words
    feats["qa_ratio"] = qa_words / word_count if word_count > 0 else 0

    # Number of questions (question marks in Q&A)
    feats["num_questions"] = qa_text.count("?")

    # Number of analyst turns
    analyst_turns = sum(
        1 for t in rec.get("speaker_turns", [])
        if t.get("role_category") == "Analyst"
    )
    feats["num_analyst_turns"] = analyst_turns

    # Speaker count
    speakers = set(
        t.get("speaker", "Unknown")
        for t in rec.get("speaker_turns", [])
        if t.get("speaker") and t.get("speaker") != "Unknown"
    )
    feats["speaker_count"] = len(speakers)

    # ── Tone (Financial Dictionary) ──────────────────────────────────
    word_counter = Counter(words)
    pos_count = sum(word_counter.get(w, 0) for w in POSITIVE_WORDS)
    neg_count = sum(word_counter.get(w, 0) for w in NEGATIVE_WORDS)
    unc_count = sum(word_counter.get(w, 0) for w in UNCERTAINTY_WORDS)

    feats["pos_count"] = pos_count
    feats["neg_count"] = neg_count
    feats["uncertainty_count"] = unc_count
    feats["tone"] = (pos_count - neg_count) / word_count if word_count > 0 else 0
    feats["uncertainty_rate"] = unc_count / word_count if word_count > 0 else 0

    # Has Q&A
    feats["has_qa"] = rec.get("has_qa", 0)

    return feats


def compute_tfidf_topics(events_df: pd.DataFrame, n_components: int = 20):
    """
    Compute TF-IDF vectors from transcripts, reduce to topic components via SVD.
    Returns DataFrame with event_id + topic_1 ... topic_N columns.
    """
    event_ids = []
    texts = []

    for _, row in events_df.iterrows():
        event_id = row["event_id"]
        path = TRANSCRIPT_CLEAN / f"{event_id}.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        text = rec.get("presentation_text", "") + " " + rec.get("qa_text", "")
        if len(text.strip()) > 100:
            event_ids.append(event_id)
            texts.append(text)

    if len(texts) < 3:
        print("  WARNING: Not enough transcripts for TF-IDF topics")
        return pd.DataFrame()

    # TF-IDF
    n_components = min(n_components, len(texts) - 1)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    # SVD dimensionality reduction
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    topic_matrix = svd.fit_transform(tfidf_matrix)

    print(f"  TF-IDF: {tfidf_matrix.shape[1]} features → {n_components} topics")
    print(f"  Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")

    # Create DataFrame
    topic_cols = [f"topic_{i+1}" for i in range(n_components)]
    df_topics = pd.DataFrame(topic_matrix, columns=topic_cols)
    df_topics["event_id"] = event_ids

    return df_topics


# ═════════════════════════════════════════════════════════════════════
# 4.4  NORMALIZATION / STANDARDIZATION / ENCODING
# ═════════════════════════════════════════════════════════════════════

def normalize_and_encode(df: pd.DataFrame, events_df: pd.DataFrame):
    """
    Standardize numeric features (z-score) and encode categoricals.
    """
    df = df.copy()

    # Add ticker from event_id
    df["ticker"] = df["event_id"].str.split("_").str[0]

    # ── Numeric columns to standardize ───────────────────────────────
    # Identify numeric columns (exclude event_id, ticker, and binary flags)
    exclude_cols = {"event_id", "ticker", "has_qa"}
    numeric_cols = [
        c for c in df.columns
        if df[c].dtype in [np.float64, np.int64, float, int]
        and c not in exclude_cols
    ]

    # Z-score standardization
    scaler = StandardScaler()
    df_z = df.copy()
    for col in numeric_cols:
        vals = df[col].values.reshape(-1, 1)
        mask = ~np.isnan(vals.ravel())
        if mask.sum() > 1:
            scaled = np.full_like(vals.ravel(), np.nan, dtype=float)
            scaled[mask] = scaler.fit_transform(vals[mask].reshape(-1, 1)).ravel()
            df_z[f"{col}_z"] = scaled

    # ── Categorical encoding ─────────────────────────────────────────
    # One-hot encode ticker
    ticker_dummies = pd.get_dummies(df["ticker"], prefix="ticker", dtype=int)
    df_z = pd.concat([df_z, ticker_dummies], axis=1)

    return df_z, numeric_cols


# 4.5  MAIN PIPELINE

def run():
    """Main feature engineering pipeline."""
    events_path = RAW_DIR / "events_clean.csv"
    if not events_path.exists():
        events_path = RAW_DIR / "events.csv"
    if not events_path.exists():
        print("ERROR: events.csv not found. Run 01_acquire.py first.")
        return

    df_events = pd.read_csv(events_path)
    print(f"Loaded {len(df_events)} events")

    # 4.1  CREATE MINUTE BARS
    print(f"\n{'='*60}")
    print("4.1  Creating 1-minute bars...")
    print(f"{'='*60}")

    all_bars = []
    for _, row in df_events.iterrows():
        event_id = row["event_id"]
        call_ts_str = row.get("call_start_ts", "")
        if pd.isna(call_ts_str) or not call_ts_str:
            continue
        call_ts = pd.Timestamp(call_ts_str)

        bars = create_minute_bars(event_id, call_ts)
        if bars is not None:
            all_bars.append(bars)
            print(f"  {event_id}: {len(bars)} minute bars")

    if all_bars:
        df_all_bars = pd.concat(all_bars, ignore_index=True)
        # Save minute bars
        df_save = df_all_bars.copy()
        if "ts" in df_save.columns:
            df_save["ts"] = pd.to_datetime(df_save["ts"]).dt.tz_localize(None)
        bars_path = FINAL_DIR / "minute_bars.parquet"
        df_save.to_parquet(bars_path, index=False)
        print(f"\nSaved: {bars_path.name} ({len(df_all_bars)} total bars)")
    else:
        print("WARNING: No minute bars created")
        df_all_bars = pd.DataFrame()

    # 4.2  MARKET FEATURES
    print(f"\n{'='*60}")
    print("4.2  Computing market features...")
    print(f"{'='*60}")

    market_feat_rows = []
    for _, row in df_events.iterrows():
        event_id = row["event_id"]
        call_ts_str = row.get("call_start_ts", "")
        if pd.isna(call_ts_str) or not call_ts_str:
            continue
        call_ts = pd.Timestamp(call_ts_str)

        # Get minute bars for this event
        if len(df_all_bars) > 0:
            event_bars = df_all_bars[df_all_bars["event_id"] == event_id].copy()
        else:
            event_bars = create_minute_bars(event_id, call_ts)

        if event_bars is not None and len(event_bars) > 0:
            feats = compute_market_features(event_bars, call_ts)
            feats["event_id"] = event_id
            market_feat_rows.append(feats)
            print(f"  {event_id}: CR_0_30={feats.get('CR_0_30', 'N/A'):.4f}" 
                  if isinstance(feats.get('CR_0_30'), float) else f"  {event_id}: computed")

    df_mkt_feats = pd.DataFrame(market_feat_rows)
    print(f"\nMarket features: {len(df_mkt_feats)} events, "
          f"{len(df_mkt_feats.columns)} columns")

    # 4.3  TRANSCRIPT FEATURES
    print(f"\n{'='*60}")
    print("4.3  Computing transcript features...")
    print(f"{'='*60}")

    tx_feat_rows = []
    for _, row in df_events.iterrows():
        event_id = row["event_id"]
        feats = compute_transcript_features(event_id)
        if feats:
            feats["event_id"] = event_id
            tx_feat_rows.append(feats)
            print(f"  {event_id}: tone={feats.get('tone', 0):.4f}, "
                  f"uncertainty={feats.get('uncertainty_rate', 0):.4f}")

    df_tx_feats = pd.DataFrame(tx_feat_rows)
    print(f"\nTranscript features: {len(df_tx_feats)} events, "
          f"{len(df_tx_feats.columns)} columns")

    # TF-IDF Topics
    print(f"\n{'='*60}")
    print("4.3b Computing TF-IDF topic features...")
    print(f"{'='*60}")

    df_topics = compute_tfidf_topics(df_events, n_components=20)
    if len(df_topics) > 0:
        df_tx_feats = df_tx_feats.merge(df_topics, on="event_id", how="left")
        print(f"  Added {len(df_topics.columns) - 1} topic features")

    # ═════════════════════════════════════════════════════════════════
    # MERGE ALL FEATURES
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Merging all features...")
    print(f"{'='*60}")

    # Start with events info
    keep_cols = ["event_id", "ticker", "company", "quarter", "fiscal_year",
                 "call_date", "call_start_ts"]
    df_base = df_events[[c for c in keep_cols if c in df_events.columns]].copy()

    # Merge market features
    if len(df_mkt_feats) > 0:
        df_base = df_base.merge(df_mkt_feats, on="event_id", how="left")

    # Merge transcript features
    if len(df_tx_feats) > 0:
        df_base = df_base.merge(df_tx_feats, on="event_id", how="left")

    print(f"  Combined: {len(df_base)} events × {len(df_base.columns)} columns")

    # 4.4  NORMALIZATION / STANDARDIZATION
    print(f"\n{'='*60}")
    print("4.4  Normalizing & encoding features...")
    print(f"{'='*60}")

    df_final, numeric_cols = normalize_and_encode(df_base, df_events)
    print(f"  Standardized {len(numeric_cols)} numeric features (z-score)")
    print(f"  One-hot encoded ticker → "
          f"{sum(1 for c in df_final.columns if c.startswith('ticker_'))} dummies")
    print(f"  Final dataset: {len(df_final)} events × {len(df_final.columns)} columns")

    # 4.5  SAVE FINAL DATASETS
    print(f"\n{'='*60}")
    print("4.5  Saving final datasets...")
    print(f"{'='*60}")

    # Event-level features
    feat_path = FINAL_DIR / "event_level_features.csv"
    df_final.to_csv(feat_path, index=False)
    print(f"  Saved: {feat_path.name}")

    # Feature justification table
    justification = []
    justification.append("Feature Justifications")
    justification.append("=" * 60)
    justification.append("")
    justification.append("MARKET FEATURES:")
    justification.append("  CR_0_5/30/120: Cumulative log returns over post-call windows — "
                        "captures directional price impact of earnings.")
    justification.append("  RV_pre/0_30/0_120: Realized volatility (sqrt sum r²) — "
                        "measures uncertainty resolution around the call.")
    justification.append("  VOL_ratio: Post/pre volume ratio — "
                        "captures change in trading activity triggered by the call.")
    justification.append("  max_runup/drawdown: Extreme cumulative moves — "
                        "captures asymmetry in price reaction.")
    justification.append("  time_to_max_abs_move: Minutes until peak absolute move — "
                        "captures how quickly information is priced in.")
    justification.append("")
    justification.append("TRANSCRIPT FEATURES:")
    justification.append("  word_count_total: Total transcript length — "
                        "proxy for information density and call complexity.")
    justification.append("  qa_ratio: Q&A share of total — "
                        "higher ratio → more analyst engagement.")
    justification.append("  num_questions: Count of questions — "
                        "direct measure of analyst scrutiny.")
    justification.append("  tone (LM dictionary): (pos - neg) / total — "
                        "net sentiment using Loughran-McDonald financial lexicon.")
    justification.append("  uncertainty_rate: uncertainty words / total — "
                        "captures forward-looking hedging language.")
    justification.append("  topic_1..20: TF-IDF + SVD latent topics — "
                        "captures thematic content without manual labeling.")
    justification.append("")
    justification.append("NORMALIZATION:")
    justification.append("  All numeric features z-scored: z = (x - mean) / std.")
    justification.append("  Ticker one-hot encoded for categorical representation.")

    just_path = FINAL_DIR / "feature_justifications.txt"
    with open(just_path, "w") as f:
        f.write("\n".join(justification))
    print(f"  Saved: {just_path.name}")

    # Print summary
    print(f"\n{'='*60}")
    print("FINAL DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Events:           {len(df_final)}")
    print(f"  Total columns:    {len(df_final.columns)}")
    print(f"  Market features:  {len([c for c in df_final.columns if c.startswith(('CR_', 'RV_', 'VOL_', 'max_', 'time_to'))])}")
    print(f"  Transcript feats: {len([c for c in df_final.columns if c.startswith(('word_', 'qa_', 'num_', 'pos_', 'neg_', 'tone', 'uncertainty', 'topic_', 'has_qa', 'speaker'))])}")
    print(f"  Standardized (_z): {len([c for c in df_final.columns if c.endswith('_z')])}")
    print(f"  Ticker dummies:   {len([c for c in df_final.columns if c.startswith('ticker_')])}")

    if len(df_all_bars) > 0:
        print(f"\n  Minute bars:      {len(df_all_bars)} total rows")
        print(f"  Events with bars: {df_all_bars['event_id'].nunique()}")

    return df_final


if __name__ == "__main__":
    run()

