#!/usr/bin/env python3
"""
01_acquire.py — Data Acquisition
Reads the original per-ticker folders, converts DOC transcripts to
structured JSON, copies market CSVs, and builds the events.csv.

Outputs:
    data_raw/transcripts_raw/{event_id}.json   (one per earnings call)
    data_raw/market_raw/{event_id}.csv         (one per earnings call)
    data_raw/events.csv                        (spine table)
"""

import os
import re
import json
import glob
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data_raw"
TRANSCRIPT_DIR = RAW_DIR / "transcripts_raw"
MARKET_DIR = RAW_DIR / "market_raw"

TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
MARKET_DIR.mkdir(parents=True, exist_ok=True)

# ── Ticker-to-company mapping (extracted from file names) ───────────
TICKER_DIRS = sorted([
    d for d in ROOT.iterdir()
    if d.is_dir() and d.name.isupper() and d.name not in (
        "AAPL AMD AMZN AVGO GOOGL META MSFT MU NVDA ORCL".split()
    ) == False
])
# Just get all uppercase dirs that look like tickers
TICKER_DIRS = sorted([
    d for d in ROOT.iterdir()
    if d.is_dir() and d.name.isupper() and len(d.name) <= 5
    and not d.name.startswith(".")
])

# ── Map call date from transcript filename ──────────────────────────
MONTH_MAP = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
}


def parse_transcript_filename(fname: str):
    """
    Extract company name, quarter, year, and call date from filenames like:
      'Apple Inc., Q1 2025 Earnings Call, Jan 30, 2025.rtf'
    Returns dict with keys: company, quarter, fiscal_year, call_date_str
    """
    base = Path(fname).stem  # strip extension
    # Pattern: Company Name, QX YYYY Earnings Call, Mon DD, YYYY
    m = re.match(
        r"^(.+?),\s+Q(\d)\s+(\d{4})\s+Earnings Call,\s+"
        r"(\w{3})\s+(\d{1,2}),\s+(\d{4})",
        base,
    )
    if not m:
        return None
    company = m.group(1).strip()
    quarter = int(m.group(2))
    fiscal_year = int(m.group(3))
    mon = MONTH_MAP.get(m.group(4))
    day = int(m.group(5))
    cal_year = int(m.group(6))
    call_date = f"{cal_year}-{mon}-{day:02d}"
    return {
        "company": company,
        "quarter": quarter,
        "fiscal_year": fiscal_year,
        "call_date": call_date,
    }


def extract_text_antiword(doc_path: str) -> str:
    """Use antiword to convert DOC (OLE2) file to plain text."""
    try:
        result = subprocess.run(
            ["antiword", str(doc_path)],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0:
            # Try UTF-8 first, fall back to latin-1
            try:
                return result.stdout.decode("utf-8")
            except UnicodeDecodeError:
                return result.stdout.decode("latin-1", errors="replace")
    except FileNotFoundError:
        print("WARNING: antiword not found. Install with: brew install antiword")
    except subprocess.TimeoutExpired:
        print(f"WARNING: antiword timed out on {doc_path}")
    return ""


def parse_transcript_sections(raw_text: str):
    """
    Split transcript into:
      - participants (executives, analysts)
      - presentation (prepared remarks)
      - qa (question & answer)
    Also extract speaker turns.
    """
    lines = raw_text.split("\n")

    # Find section boundaries
    pres_idx = None
    qa_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "Presentation" and pres_idx is None:
            # Skip the Table of Contents mention
            # Real Presentation section starts after participants
            if i > 50:  # heuristic: participants section is < 50 lines
                pres_idx = i
            elif any("EXECUTIVES" in lines[j] for j in range(max(0, i-30), i)):
                pres_idx = i
        if stripped == "Question and Answer" and qa_idx is None:
            if i > 50:
                qa_idx = i

    # If we didn't find them with strict rules, try again more loosely
    if pres_idx is None:
        for i, line in enumerate(lines):
            if line.strip() == "Presentation":
                pres_idx = i
                break

    if qa_idx is None:
        for i, line in enumerate(lines):
            if line.strip() == "Question and Answer" and i > (pres_idx or 0):
                qa_idx = i
                break

    # Extract sections
    pres_text = ""
    qa_text = ""

    if pres_idx is not None and qa_idx is not None:
        pres_text = "\n".join(lines[pres_idx + 1 : qa_idx]).strip()
        qa_text = "\n".join(lines[qa_idx + 1 :]).strip()
    elif pres_idx is not None:
        pres_text = "\n".join(lines[pres_idx + 1 :]).strip()

    # Extract speaker turns from both sections
    speaker_turns = extract_speaker_turns(raw_text)

    return {
        "presentation_text": pres_text,
        "qa_text": qa_text,
        "has_qa": 1 if qa_text else 0,
        "speaker_turns": speaker_turns,
    }


def extract_speaker_turns(text: str):
    """
    Identify speaker turns. In Capital IQ transcripts, speakers appear as:
    <blank line>
    Speaker Name
    Title / Affiliation
    <blank line>
    Spoken text...
    """
    turns = []
    lines = text.split("\n")
    i = 0
    current_speaker = "Unknown"
    current_role = ""
    current_text_lines = []

    # Known role keywords
    role_keywords = [
        "CEO", "CFO", "COO", "CTO", "VP", "President", "Director",
        "Analyst", "Research", "Division", "Officer", "Relations",
        "Operator", "Securities", "Capital", "Morgan", "Goldman",
        "JPMorgan", "Bank", "Citigroup", "Barclays", "Deutsche",
        "UBS", "Credit Suisse", "Evercore", "Bernstein", "Oppenheimer",
        "Wells Fargo", "Piper", "Cowen", "Canaccord", "Stifel",
        "Needham", "Wedbush", "Wolfe", "Loop", "Raymond James",
        "Jefferies", "KeyBanc", "Mizuho", "Truist", "BofA",
    ]

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Detect if this is a speaker label line
        # Speaker names are typically on their own line, followed by a role line
        if stripped == "Operator":
            if current_text_lines:
                turns.append({
                    "speaker": current_speaker,
                    "role": current_role,
                    "text": " ".join(current_text_lines).strip(),
                })
            current_speaker = "Operator"
            current_role = "Operator"
            current_text_lines = []
            continue

        # Check if line looks like a speaker name (short, no punctuation except periods)
        if (
            stripped
            and len(stripped) < 60
            and not stripped.endswith(".")
            and not stripped.endswith("?")
            and not stripped.endswith("!")
            and not any(c in stripped for c in "[]|{}()")
            and i + 1 < len(lines)
        ):
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            # Check if next line looks like a title/affiliation
            if any(kw in next_line for kw in role_keywords):
                if current_text_lines:
                    turns.append({
                        "speaker": current_speaker,
                        "role": current_role,
                        "text": " ".join(current_text_lines).strip(),
                    })
                current_speaker = stripped
                current_role = next_line
                current_text_lines = []
                continue

        # Accumulate text
        if stripped and not stripped.startswith("|"):
            current_text_lines.append(stripped)

    # Last speaker
    if current_text_lines:
        turns.append({
            "speaker": current_speaker,
            "role": current_role,
            "text": " ".join(current_text_lines).strip(),
        })

    return turns


def extract_call_time_from_text(raw_text: str) -> str:
    """
    Extract the call start timestamp from the transcript header.
    E.g., "Thursday, January 30, 2025 10:00 PM GMT"
    Returns ISO format string with timezone.
    """
    # Pattern: Day, Month DD, YYYY HH:MM AM/PM TZ
    m = re.search(
        r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+"
        r"(\w+ \d{1,2}, \d{4})\s+(\d{1,2}:\d{2}\s*[AP]M)\s+(\w+)",
        raw_text,
    )
    if m:
        date_str = m.group(1)
        time_str = m.group(2)
        tz_str = m.group(3)
        dt_str = f"{date_str} {time_str}"
        try:
            dt = datetime.strptime(dt_str, "%B %d, %Y %I:%M %p")
        except ValueError:
            dt = datetime.strptime(dt_str, "%B %d, %Y %I:%M%p")

        # Map timezone abbreviation to UTC offset
        tz_map = {"GMT": "+00:00", "UTC": "+00:00", "EST": "-05:00", "EDT": "-04:00"}
        offset = tz_map.get(tz_str, "+00:00")
        return dt.strftime(f"%Y-%m-%d %H:%M:%S") + offset
    return ""


def make_event_id(ticker: str, call_date: str) -> str:
    """Create a unique event ID like AAPL_2025-01-30."""
    return f"{ticker}_{call_date}"


def match_market_file(ticker_dir: Path, ticker: str, call_date: str):
    """
    Find the matching market CSV for a given call date.
    Market CSVs are named like AAPL_Q1_2025.csv.
    We match by checking which CSV contains data for that date.
    """
    csv_files = sorted(ticker_dir.glob(f"{ticker}_*.csv"))
    for csv_path in csv_files:
        # Quick check: read first 2 lines to see date
        try:
            df_head = pd.read_csv(csv_path, nrows=2)
            if "ts_event" in df_head.columns:
                first_ts = str(df_head["ts_event"].iloc[0])[:10]
                if first_ts == call_date:
                    return csv_path
        except Exception:
            continue
    return None


def run():
    """Main acquisition pipeline."""
    events = []

    for ticker_dir in TICKER_DIRS:
        ticker = ticker_dir.name
        print(f"\n{'='*60}")
        print(f"Processing {ticker}")
        print(f"{'='*60}")

        # Find all transcript files (RTF/DOC)
        transcript_files = sorted(ticker_dir.glob("*.rtf"))

        for tf in transcript_files:
            # Skip duplicate files (e.g., "(1)" suffix)
            if "(1)" in tf.name:
                print(f"  Skipping duplicate: {tf.name}")
                continue

            info = parse_transcript_filename(tf.name)
            if info is None:
                print(f"  WARNING: Could not parse filename: {tf.name}")
                continue

            call_date = info["call_date"]
            event_id = make_event_id(ticker, call_date)
            print(f"\n  Event: {event_id}")
            print(f"  File:  {tf.name}")
            print(f"  Q{info['quarter']} FY{info['fiscal_year']}, date={call_date}")

            # ── Extract transcript text ──────────────────────────────
            raw_text = extract_text_antiword(str(tf))
            if not raw_text:
                print(f"  WARNING: No text extracted from {tf.name}")
                continue

            # Parse sections
            sections = parse_transcript_sections(raw_text)
            call_ts = extract_call_time_from_text(raw_text)

            # Build JSON record
            transcript_record = {
                "event_id": event_id,
                "ticker": ticker,
                "company": info["company"],
                "quarter": info["quarter"],
                "fiscal_year": info["fiscal_year"],
                "call_date": call_date,
                "call_start_ts": call_ts,
                "raw_text": raw_text,
                "presentation_text": sections["presentation_text"],
                "qa_text": sections["qa_text"],
                "has_qa": sections["has_qa"],
                "speaker_turns": sections["speaker_turns"],
                "source_file": tf.name,
            }

            # Save transcript JSON
            out_path = TRANSCRIPT_DIR / f"{event_id}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(transcript_record, f, indent=2, ensure_ascii=False)
            print(f"  Saved transcript → {out_path.name}")

            # ── Copy market data ─────────────────────────────────────
            market_file = match_market_file(ticker_dir, ticker, call_date)
            market_source = "Databento"
            if market_file:
                # Copy to market_raw
                df_mkt = pd.read_csv(market_file)
                out_mkt = MARKET_DIR / f"{event_id}.csv"
                df_mkt.to_csv(out_mkt, index=False)
                print(f"  Saved market   → {out_mkt.name}  ({len(df_mkt)} rows)")
            else:
                print(f"  WARNING: No matching market CSV for {call_date}")
                market_source = "MISSING"

            # ── Add to events table ──────────────────────────────────
            events.append({
                "event_id": event_id,
                "ticker": ticker,
                "company": info["company"],
                "quarter": f"Q{info['quarter']}",
                "fiscal_year": info["fiscal_year"],
                "call_date": call_date,
                "call_start_ts": call_ts,
                "transcript_source": "S&P Capital IQ",
                "market_source": market_source,
                "transcript_file": tf.name,
                "market_file": market_file.name if market_file else "",
                "notes": "",
            })

    # ── Save events spine ────────────────────────────────────────────
    df_events = pd.DataFrame(events)
    df_events = df_events.sort_values(["ticker", "call_date"]).reset_index(drop=True)
    events_path = RAW_DIR / "events.csv"
    df_events.to_csv(events_path, index=False)
    print(f"\n{'='*60}")
    print(f"Events spine saved → {events_path}")
    print(f"Total events: {len(df_events)}")
    print(f"Tickers: {df_events['ticker'].nunique()}")
    print(f"{'='*60}")

    return df_events


if __name__ == "__main__":
    run()

