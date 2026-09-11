# src/extract_late_night_results.py

import json
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

# Change this to your actual extraction results JSON file
INPUT_FILE = Path("experiments/extraction_results/extraction_results.json")

# Output files
OUTPUT_JSON = Path("results/late_night_stability_results.json")
OUTPUT_CSV = Path("results/late_night_stability_results.csv")

# 10:30 PM IST = 17:00 UTC
CUTOFF_UTC = datetime(
    2026, 9, 4, 17, 0, 0, tzinfo=timezone.utc
)

# Intended stability experiment
EXPECTED_RUNS = {
    "test_001": 3,
    "test_002": 1,
    "test_003": 3,
    "test_004": 3,
    "test_005": 3,
}

EXPECTED_EXTRACTION_RUNS = {1, 2, 3}

IST = timezone(timedelta(hours=5, minutes=30))


# ============================================================
# HELPERS
# ============================================================

def parse_timestamp(timestamp):
    """Convert timestamp string into timezone-aware datetime."""
    return datetime.fromisoformat(timestamp)


def utc_to_ist(timestamp):
    """Convert UTC timestamp string to IST."""
    dt = parse_timestamp(timestamp)
    return dt.astimezone(IST)


def is_target_record(record):
    """Check whether a record belongs to the late-night stability experiment."""

    if record.get("status") != "success":
        return False

    image_id = record.get("image_id")
    baseline_run = record.get("baseline_run_number")
    extraction_run = record.get("extraction_run_number")

    # Check intended baseline run
    if image_id not in EXPECTED_RUNS:
        return False

    if baseline_run != EXPECTED_RUNS[image_id]:
        return False

    # Check extraction run 1, 2 or 3
    if extraction_run not in EXPECTED_EXTRACTION_RUNS:
        return False

    # Check timestamp
    timestamp = record.get("timestamp_utc")

    if not timestamp:
        return False

    try:
        dt = parse_timestamp(timestamp)
    except ValueError:
        return False

    return dt >= CUTOFF_UTC


# ============================================================
# MAIN
# ============================================================

def extract_late_night_results():

    print("=" * 70)
    print("LATE-NIGHT EXTRACTION RESULT FILTER")
    print("=" * 70)

    print(f"\nInput file:")
    print(f"  {INPUT_FILE}")

    print("\nCutoff:")
    print("  10:30 PM IST")
    print("  2026-09-04 17:00 UTC")

    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle either:
    # 1. a list directly
    # 2. {"results": [...]}
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and "results" in data:
        records = data["results"]
    else:
        raise ValueError(
            "Unexpected JSON format. Expected a list or {'results': [...]}"
        )

    print(f"\nTotal records in file: {len(records)}")

    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    matched = []

    for record in records:
        if is_target_record(record):
            matched.append(record)

    # Sort chronologically
    matched.sort(
        key=lambda r: parse_timestamp(r["timestamp_utc"])
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"MATCHED RECORDS: {len(matched)}")
    print("=" * 70)

    for record in matched:

        ist_time = utc_to_ist(record["timestamp_utc"])

        print(
            f"{ist_time.strftime('%I:%M:%S %p')} IST | "
            f"{record['image_id']} | "
            f"baseline {record['baseline_run_number']} | "
            f"extraction {record['extraction_run_number']}"
        )

    # --------------------------------------------------------
    # Check expected 15
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPECTED VS FOUND")
    print("=" * 70)

    expected_total = 15

    print(f"\nExpected: {expected_total}")
    print(f"Found:    {len(matched)}")

    # Create lookup
    found_keys = {
        (
            r["image_id"],
            r["baseline_run_number"],
            r["extraction_run_number"],
        )
        for r in matched
    }

    missing = []

    for image_id, baseline_run in EXPECTED_RUNS.items():

        for extraction_run in sorted(EXPECTED_EXTRACTION_RUNS):

            key = (
                image_id,
                baseline_run,
                extraction_run,
            )

            if key not in found_keys:
                missing.append(key)

    if missing:

        print("\nMISSING RECORDS:")
        for image_id, baseline, extraction in missing:
            print(
                f"  {image_id} | "
                f"baseline {baseline} | "
                f"extraction {extraction}"
            )

    else:
        print("\n✓ All 15 expected stability extractions found!")

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(matched, f, indent=4, ensure_ascii=False)

    print(f"\nJSON saved to:")
    print(f"  {OUTPUT_JSON}")

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_rows = []

    for record in matched:

        ist_time = utc_to_ist(record["timestamp_utc"])

        csv_rows.append({
            "image_id": record.get("image_id"),
            "baseline_run_number": record.get("baseline_run_number"),
            "extraction_run_number": record.get("extraction_run_number"),
            "timestamp_utc": record.get("timestamp_utc"),
            "timestamp_ist": ist_time.isoformat(),
            "model": record.get("model"),
            "temperature": record.get("temperature"),
            "status": record.get("status"),
        })

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "baseline_run_number",
                "extraction_run_number",
                "timestamp_utc",
                "timestamp_ist",
                "model",
                "temperature",
                "status",
            ],
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nCSV saved to:")
    print(f"  {OUTPUT_CSV}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    extract_late_night_results()