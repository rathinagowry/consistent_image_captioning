import json
import csv
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "experiments/extraction_results/extraction_results.json"
)

OUTPUT_JSON = Path(
    "results/stability_15_results.json"
)

OUTPUT_CSV = Path(
    "results/stability_15_results.csv"
)


# The exact 5 baseline captions used for the
# semantic extraction stability experiment.

STABILITY_TARGETS = {
    ("test_001", 3),
    ("test_002", 1),
    ("test_003", 3),
    ("test_004", 3),
    ("test_005", 3),
}

EXPECTED_EXTRACTION_RUNS = {1, 2, 3}


# ============================================================
# LOAD EXTRACTION RESULTS
# ============================================================

def load_results():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Extraction results not found:\n{INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "results" in data:
        return data["results"]

    raise ValueError(
        "Unexpected JSON format."
    )


# ============================================================
# FILTER TARGET RECORDS
# ============================================================

def is_target_record(record):

    if record.get("status") != "success":
        return False

    image_id = record.get("image_id")

    baseline_run = record.get(
        "baseline_run_number"
    )

    extraction_run = record.get(
        "extraction_run_number"
    )

    # Check image + baseline caption
    if (
        image_id,
        baseline_run,
    ) not in STABILITY_TARGETS:
        return False

    # Check extraction run
    if extraction_run not in EXPECTED_EXTRACTION_RUNS:
        return False

    # CSR must exist
    if not record.get("csr"):
        return False

    return True


# ============================================================
# CREATE DATASET
# ============================================================

def create_stability_dataset():

    print("=" * 70)
    print("CREATE 15-RECORD STABILITY DATASET")
    print("=" * 70)

    print(
        f"\nInput:\n  {INPUT_FILE}"
    )

    records = load_results()

    print(
        f"\nTotal records in extraction file: "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    stability_records = [
        record
        for record in records
        if is_target_record(record)
    ]

    # --------------------------------------------------------
    # Remove accidental duplicates
    # --------------------------------------------------------

    unique_records = {}

    for record in stability_records:

        key = (
            record["image_id"],
            record["baseline_run_number"],
            record["extraction_run_number"],
        )

        # Keep the first successful record
        if key not in unique_records:
            unique_records[key] = record

    stability_records = list(
        unique_records.values()
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    stability_records.sort(
        key=lambda record: (
            record["image_id"],
            record["baseline_run_number"],
            record["extraction_run_number"],
        )
    )

    # --------------------------------------------------------
    # Expected records
    # --------------------------------------------------------

    expected_keys = set()

    for image_id, baseline_run in STABILITY_TARGETS:

        for extraction_run in sorted(
            EXPECTED_EXTRACTION_RUNS
        ):

            expected_keys.add(
                (
                    image_id,
                    baseline_run,
                    extraction_run,
                )
            )

    found_keys = {
        (
            record["image_id"],
            record["baseline_run_number"],
            record["extraction_run_number"],
        )
        for record in stability_records
    }

    missing = expected_keys - found_keys

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STABILITY DATASET CHECK")
    print("=" * 70)

    print(
        f"\nExpected records : {len(expected_keys)}"
    )

    print(
        f"Found records    : {len(stability_records)}"
    )

    # --------------------------------------------------------
    # Missing records
    # --------------------------------------------------------

    if missing:

        print("\nMISSING RECORDS:")

        for image_id, baseline_run, extraction_run in sorted(
            missing
        ):

            print(
                f"  {image_id} | "
                f"baseline {baseline_run} | "
                f"extraction {extraction_run}"
            )

        raise ValueError(
            f"Dataset incomplete: "
            f"{len(missing)} record(s) missing."
        )

    print(
        "\n✓ All 15 stability records found."
    )

    # --------------------------------------------------------
    # Per-image summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-IMAGE RECORDS")
    print("=" * 70)

    for image_id, baseline_run in sorted(
        STABILITY_TARGETS
    ):

        count = sum(
            record["image_id"] == image_id
            and record["baseline_run_number"]
            == baseline_run
            for record in stability_records
        )

        print(
            f"{image_id} | "
            f"baseline {baseline_run} | "
            f"{count}/3 extractions"
        )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            stability_records,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_rows = []

    for record in stability_records:

        csv_rows.append(
            {
                "image_id": record.get(
                    "image_id"
                ),
                "baseline_run_number": record.get(
                    "baseline_run_number"
                ),
                "extraction_run_number": record.get(
                    "extraction_run_number"
                ),
                "caption": record.get(
                    "caption"
                ),
                "model": record.get(
                    "model"
                ),
                "temperature": record.get(
                    "temperature"
                ),
                "timestamp_utc": record.get(
                    "timestamp_utc"
                ),
                "generation_time_seconds": record.get(
                    "generation_time_seconds"
                ),
                "attempts": record.get(
                    "attempts"
                ),
                "status": record.get(
                    "status"
                ),
            }
        )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "baseline_run_number",
                "extraction_run_number",
                "caption",
                "model",
                "temperature",
                "timestamp_utc",
                "generation_time_seconds",
                "attempts",
                "status",
            ],
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET CREATED")
    print("=" * 70)

    print(
        f"\nJSON:\n  {OUTPUT_JSON}"
    )

    print(
        f"\nCSV:\n  {OUTPUT_CSV}"
    )

    print(
        f"\nTotal stability records: "
        f"{len(stability_records)}"
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    create_stability_dataset()