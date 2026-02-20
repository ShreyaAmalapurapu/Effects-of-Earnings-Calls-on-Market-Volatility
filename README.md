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

---

# 1. Data Collection Overview

This project integrates two primary data sources:

1. Earnings call transcripts (text data)
2. High-frequency stock market data (1-second OHLCV)

The objective is to synchronize narrative financial communication with real-time market behavior.

---

# 2. Earnings Call Transcripts Collection  
## Source: Capital IQ

### 2.1 Data Source

Earnings call transcripts were obtained from **S&P Capital IQ**, a professional financial database that provides structured corporate event data.

Source:
S&P Capital IQ Platform(https://www.capitaliq.com)

---

### 2.2 Company Selection

We selected 10 S&P 500 technology-related companies.  
For each company, we collected transcripts from the most recent four quarterly earnings calls plus an extra one as a backup.

Total transcripts collected:

10 companies × 5 calls = 50 transcripts

---

### 2.3 Collection Procedure

For each company:

1. Search ticker (e.g., AAPL US Equity) in Capital IQ.
2. Navigate to:
   Company → Research → Transcripts
3. Filter by:
   - Event Type: Earnings Call
   - Most recent 5 quarters
4. Download transcript in Word format.
5. Extract and record:
   - Company name
   - Ticker
   - Call date
   - Call start time
   - Full transcript text

---

### 2.4 Data Structure

Each transcript is stored in structured CSV format with the following fields:

| Variable | Description |
|----------|------------|
| ticker | Company ticker |
| call_date | Earnings call date |
| call_time | Start time (GMT) |
| speaker | Speaker name (if available) |
| section | Prepared Remarks / Q&A |
| text | Transcript content |

A separate metadata file records call-level information.

---

### 2.5 Time Standardization

Earnings call times are typically reported in Greenwich Mean Time (GMT).

For synchronization with high-frequency market data, all timestamps are converted to:

> Coordinated Universal Time (UTC)

However, GMT time is equivalent to UTC time, so we do not need to perform any additional time conversion.

---

# 3. High-Frequency Market Data Collection  
## Source: Databento API

### 3.1 Objective

The objective of this step is to collect high-frequency stock market data surrounding corporate earnings calls in order to analyze short-term market reactions.

For each earnings call, we extract:

- Call start time
- A 3-hour observation window beginning at the call start
- 1-second OHLCV (Open, High, Low, Close, Volume) data

All timestamps are standardized to UTC to ensure accurate alignment with transcript data.

---

### 3.2 Data Source

Market data is obtained from:

Databento Historical API

Databento provides exchange-level high-frequency data for U.S. equities, including Nasdaq and NYSE.

Source:
Databento(https://databento.com)

---

### 3.3 Dataset Selection

We use the following datasets depending on exchange listing:

| Exchange | Dataset |
|----------|----------|
| Nasdaq | XNAS.ITCH |
| NYSE | XNYS.PILLAR |

Schema used:

`ohlcv-1s`

This schema provides pre-aggregated 1-second OHLCV bars.

---

### 3.4 Time Standardization

Earnings calls are typically reported in GMT.

Since Databento requires UTC timestamps, all call times are converted to UTC before querying. There has been no change in timing.

---

### 3.5 API Cost

Before performing market collection operations, it is crucial to verify the data pricing. This can be accomplished using the following code for querying.

```python
import databento as db

API_KEY = "ENTER HERE" # enter your API key
client = db.Historical(API_KEY)

cost = client.metadata.get_cost( # check the cost (we only have $125 in total)
    dataset=exchange,
    schema="ohlcv-1s",
    symbols=[ticker], # Company name
    start= start_utc,  # UTC Time
    end= end_utc
)
print("The cost will be",cost) # after we find it's coverable, continue the next step
```

---

### 3.6 Data Storage

After all preparatory work is completed, utilize the collected data points—start date, start time, ticker symbol, and exchange abbreviation—to retrieve the data.

```python
exchange = "ENTER HERE" # XNAS.ITCH = Nasdaq; XNYS.PILLAR = NYSE
ticker = "ENTER HERE" # ticker name
utc_time_str = "ENTER HERE" # earning call start time in UTC

data = client.timeseries.get_range(
    dataset=exchange,
    schema="ohlcv-1s",
    symbols=[ticker], # Company name
    start=start_utc,  # UTC Time
    end=end_utc
)
df = data.to_df()
```

Then use code to save the obtained data locally.

```python
callname = "ENTER HERE"
filename2 = f"{ticker}_{callname}.csv" # save the file
df.to_csv(filename2)
```

The corresponding data should be presented in the following format.

| Variable   | Type        | Description |
|------------|------------|-------------|
| ts_event   | datetime (UTC) | Timestamp of the 1-second interval |
| open       | float      | Opening price within the second |
| high       | float      | Highest price within the second |

---
# 4. Exploratory Data Analysis

## Source: Event-Level Feature Dataset

### 4.1 Objective

The objective of this step is to explore the statistical properties and relationships between transcript-based features and short-horizon market reaction variables.

For each earnings call event, we analyze:

Tone (Loughran–McDonald dictionary based)

Uncertainty rate

Cumulative log returns (5m / 30m / 120m)

Realized volatility

Trading volume ratio

Q&A participation ratio

All variables are examined in both raw form and standardized (Z-score) form to ensure comparability.

4.2 Data Source

Event-level features are obtained from:

event_level_features.csv

This dataset contains merged transcript-derived features and high-frequency market reaction metrics constructed in previous steps.

The file includes approximately:

50 earnings call events × 100+ engineered features.

4.3 Distribution Analysis

We first examine the distribution of key raw variables.

Variables analyzed:

tone

uncertainty_rate

CR_0_120 (cumulative log return over 0–120 minutes)

Figures generated:

Histogram of tone

Histogram of uncertainty_rate

Histogram of CR_0_120

These visualizations allow us to evaluate:

Skewness and dispersion

Heavy-tailed behavior

Presence of extreme observations

Short-horizon returns exhibit fat tails, consistent with high-frequency earnings announcement reactions.

4.4 Cross-Sectional Comparison

To examine heterogeneity across firms, we generate boxplots grouped by ticker.

Variables compared:

Tone by ticker

CR_0_120 by ticker

This step allows us to:

Compare median communication tone across firms

Observe dispersion differences

Detect outlier events and extreme return reactions

Firm-level variation suggests that communication style and market sensitivity differ across companies.

4.5 Relationship Analysis

To assess preliminary linear associations, we analyze standardized (Z-scored) variables.

The following relationships are examined:

tone_z vs CR_0_120_z

uncertainty_rate_z vs RV_0_120_z

qa_ratio_z vs VOL_ratio_z

For each pair:

Scatter plot is generated

Linear regression fit line is overlaid

This step provides initial evidence regarding whether textual characteristics may be associated with short-term return, volatility, or trading activity.

4.6 Correlation Structure

To evaluate overall dependency patterns and potential multicollinearity, we compute a Pearson correlation matrix for:

tone_z

uncertainty_rate_z

qa_ratio_z

CR_0_120_z

RV_0_120_z

VOL_ratio_z

A grayscale heatmap is generated to visualize correlation magnitudes.

Correlation values range between −1 and 1.

Moderate correlation levels indicate that explanatory variables are not excessively collinear, supporting subsequent regression modeling.

4.7 Output Storage

All generated figures are automatically saved to:

eda_figs/

The directory contains:

Histogram plots (raw distributions)

Boxplots by ticker

Scatter plots with regression lines

Correlation heatmap

These figures provide publication-quality visual summaries of distributional properties and preliminary statistical relationships.
