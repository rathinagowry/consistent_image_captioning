import json
from pathlib import Path

from semantic_aligner import align_csr


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "canonical_stability_results.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "aligned_stability_results.json"
)


# ============================================================
# LOAD
# ============================================================

def load_results():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# SAVE
# ============================================================

def save_results(results):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SEMANTIC ALIGNMENT EXPERIMENT")
    print("=" * 70)

    print(
        f"\nInput : {INPUT_FILE}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Load canonicalization results
    # --------------------------------------------------------

    records = load_results()

    print(
        f"\nLoaded records: {len(records)}"
    )

    aligned_results = []

    successful = 0
    failed = 0

    # --------------------------------------------------------
    # Align every canonical CSR
    # --------------------------------------------------------

    for index, record in enumerate(
        records,
        start=1,
    ):

        image_id = record.get(
            "image_id",
            "unknown",
        )

        baseline_run = record.get(
            "baseline_run_number"
        )

        extraction_run = record.get(
            "extraction_run_number"
        )

        caption = record.get(
            "caption",
            "",
        )

        canonical_csr = record.get(
            "canonical_csr"
        )

        print(
            f"\n[{index}/{len(records)}] "
            f"{image_id} | "
            f"baseline={baseline_run} | "
            f"extraction={extraction_run}"
        )

        if canonical_csr is None:

            print(
                "  ERROR: canonical CSR missing"
            )

            failed += 1
            continue

        try:

            # ------------------------------------------------
            # Deterministic semantic alignment
            # ------------------------------------------------

            aligned_csr = align_csr(
                canonical_csr,
                caption=caption,
            )

            # ------------------------------------------------
            # Preserve complete experiment history
            # ------------------------------------------------

            new_record = {
                "image_id": image_id,

                "image_file": record.get(
                    "image_file"
                ),

                "baseline_run_number":
                    baseline_run,

                "extraction_run_number":
                    extraction_run,

                "caption": caption,

                "original_csr":
                    record.get(
                        "original_csr"
                    ),

                "canonical_csr":
                    canonical_csr,

                "aligned_csr":
                    aligned_csr,

                "status": "success",
            }

            aligned_results.append(
                new_record
            )

            successful += 1

            print(
                "  ✓ Semantic alignment successful"
            )

        except Exception as e:

            print(
                f"  ERROR: {e}"
            )

            failed += 1

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        aligned_results
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "SEMANTIC ALIGNMENT COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Input records  : {len(records)}"
    )

    print(
        f"Successful     : {successful}"
    )

    print(
        f"Failed         : {failed}"
    )

    print(
        f"Output records : {len(aligned_results)}"
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()