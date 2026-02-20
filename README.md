# STAT5243 Project 1: Earnings Call Analysis

## Overview
This project analyzes the relationship between earnings call transcripts and market microstructure data for 10 major tech stocks (AAPL, AMD, AMZN, AVGO, GOOGL, META, MSFT, MU, NVDA, ORCL) across ~50 earnings events from late 2024 through early 2026.

**Data Sources:**
- **Transcripts:** S&P Capital IQ earnings call transcripts (DOC format)
- **Market Data:** Databento 1-second OHLCV bars (after-hours/extended trading)

## How to Run

### Prerequisites
- Python 3.10+
- `antiword` (for DOC→text conversion): `brew install antiword`

### Setup
```bash
cd "STAT5243 Project 1"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Execute Pipeline (in order)
```bash
# Step 1: Acquire & organize raw data
python src/01_acquire.py

# Step 2: Clean data & compute quality flags
python src/02_clean.py

# Step 3: Exploratory data analysis
python src/03_eda.py

# Step 4: Feature engineering & final datasets
python src/04_features.py
```

## Pipeline Details

### 1. Data Acquisition (`01_acquire.py`)
- Converts DOC-format transcripts to structured JSON (preserving speaker tags, sections)
- Copies market CSVs with standardized naming (`{TICKER}_{DATE}.csv`)
- Builds the **events.csv** spine with: event_id, ticker, call_start_ts, sources

### 2. Data Cleaning (`02_clean.py`)
- **Timestamps:** Converts to US/Eastern, enforces [T−60min, T+120min] window
- **Market:** Removes negative prices, deduplicates, flags outliers (MAD-based), forward-fills ≤5s gaps
- **Transcripts:** Normalizes whitespace, removes encoding artifacts, standardizes speaker tags
- **Quality flags:** Missing chunks, duplicate timestamps, coverage metrics

### 3. Exploratory Data Analysis (`03_eda.py`)
- **Per-event metrics:** n_ticks, coverage_pct, returns (5m/30m/120m), realized vol, volume ratio, word counts, Q&A ratio
- **6 publication-quality plots:** Cumulative return curves, return distributions, scatter plots, volume/volatility comparisons
- **Interpretation bullets** (see `notebooks/eda_summary.txt`)

### 4. Feature Engineering (`04_features.py`)
- **Minute bars:** 1-second → 1-minute OHLCV resampling with log returns
- **Market features:** CR (cumulative returns), RV (realized vol), VOL_ratio, max runup/drawdown, time-to-max-move
- **Transcript features:** Word counts, Q&A ratio, tone (Loughran-McDonald dictionary), uncertainty rate, # questions, # analyst turns
- **Topic features:** TF-IDF (5000 terms) → 20 SVD topic components (84.9% variance explained)
- **Normalization:** Z-score standardization + one-hot ticker encoding
- **Output:** `event_level_features.csv` (50 events × 107 columns) + `minute_bars.parquet`

