import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    EXTRACTION_MODEL,
    EXTRACTION_TEMPERATURE,
    EXTRACTION_RUNS,
    EXTRACTION_MODE,
    STABILITY_CAPTIONS,
    EXTRACTION_RESULT_DIR,
    EXTRACTION_JSON_OUTPUT,
    EXTRACTION_CSV_OUTPUT,
    JSON_OUTPUT,
)

from src.semantic_extractor import (
    extract_semantics,
)


# ============================================================
# LOAD BASELINE CAPTIONS
# ============================================================

def load_baseline_results():
    """
    Load existing baseline generation results.
    """

    path = Path(JSON_OUTPUT)

    if not path.exists():
        raise FileNotFoundError(
            f"Baseline results not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)

def is_valid_caption(caption):
    """
    Check whether a baseline output looks like a usable
    image caption.
    """

    if not caption:
        return False

    caption = str(caption).strip()

    if len(caption) < 10:
        return False

    # Reject extremely long reasoning responses.
    if len(caption) > 500:
        return False

    lower = caption.lower()

    invalid_patterns = [
        "the user wants",
        "identify key elements",
        "synthesize into",
        "drafting the sentence",
        "final selection",
        "final polish",
        "concise version",
        "i'm not sure what you're asking",
        "i'm not sure if i can do this",
    ]

    for pattern in invalid_patterns:
        if pattern in lower:
            return False

    return True
# ============================================================
# LOAD EXISTING EXTRACTION RESULTS
# ============================================================

def load_existing_results():
    """
    Load previously saved extraction results.
    """

    path = Path(
        EXTRACTION_JSON_OUTPUT
    )

    if not path.exists():
        return []

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if isinstance(data, list):
            return data

    except Exception:
        pass

    return []


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    Path(
        EXTRACTION_RESULT_DIR
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    # JSON

    with open(
        EXTRACTION_JSON_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # CSV

    dataframe = pd.DataFrame(
        results
    )

    dataframe.to_csv(
        EXTRACTION_CSV_OUTPUT,
        index=False,
        encoding="utf-8",
    )


# ============================================================
# RUN EXPERIMENT
# ============================================================
MISSING_STABILITY_TARGETS = {
    ("test_001", 3),
    ("test_002", 1),
}
LATE_NIGHT_CUTOFF_UTC = datetime(
    2026,
    9,
    4,
    17,
    0,
    0,
    tzinfo=timezone.utc,
)
def run_extraction_experiment():

    baseline_results = (
        load_baseline_results()
    )

    extraction_results = (
        load_existing_results()
    )

    # --------------------------------------------------------
    # Only use successful baseline captions
    # --------------------------------------------------------

    captions = [
    result
    for result in baseline_results
    if result.get("status") == "success"
    and result.get("output")
    and is_valid_caption(result.get("output"))
    ]
    if not captions:
            raise ValueError(
                "No successful baseline captions found."
            )
    # if EXTRACTION_MODE == "stability":

    #     captions = [
    #         result
    #         for result in captions
    #         if (
    #             result.get("image_id"),
    #             result.get("run_number"),
    #         ) in STABILITY_CAPTIONS
    #     ]
    if EXTRACTION_MODE == "stability":

        captions = [
            result
            for result in captions
            if (
                result.get("image_id"),
                result.get("run_number"),
            ) in MISSING_STABILITY_TARGETS
        ]

        if not captions:
            raise ValueError(
                "None of the missing stability captions were found."
            )

    

    print("=" * 60)
    print("SEMANTIC EXTRACTION STABILITY EXPERIMENT")
    print("=" * 60)

    print(
        f"Extraction model : "
        f"{EXTRACTION_MODEL}"
    )

    print(
        f"Temperature      : "
        f"{EXTRACTION_TEMPERATURE}"
    )

    print(
        f"Extraction runs  : "
        f"{EXTRACTION_RUNS}"
    )

    print(
        f"Captions         : "
        f"{len(captions)}"
    )

    print(
        f"Target calls     : "
        f"{len(captions) * EXTRACTION_RUNS}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Process each baseline caption
    # --------------------------------------------------------

    for index, baseline in enumerate(
        captions,
        start=1,
    ):

        image_id = baseline[
            "image_id"
        ]

        run_number = baseline[
            "run_number"
        ]

        caption = baseline[
            "output"
        ]

        print()
        print(
            f"[{index}/{len(captions)}]"
            f" {image_id}"
            f" | baseline run {run_number}"
        )

        print(
            f"  Caption: {caption}"
        )
                # ----------------------------------------------------
        # Validate baseline caption
        # ----------------------------------------------------

        if not is_valid_caption(caption):

            print(
                "  INVALID BASELINE CAPTION"
            )

            print(
                "  Skipping semantic extraction."
            )

            continue
        # ----------------------------------------------------
        # Count existing extraction results
        # ----------------------------------------------------

        # existing = [
        #     result
        #     for result in extraction_results
        #     if result.get("image_id")
        #     == image_id
        #     and result.get(
        #         "baseline_run_number"
        #     )
        #     == run_number
        #     and result.get("status")
        #     == "success"
        # ]
        existing = [
            result
            for result in extraction_results
            if result.get("image_id") == image_id
            and result.get("baseline_run_number") == run_number
            and result.get("status") == "success"
            and result.get("timestamp_utc")
            and datetime.fromisoformat(
                result["timestamp_utc"]
            ) >= LATE_NIGHT_CUTOFF_UTC
        ]

        existing_count = len(existing)

        if existing_count >= EXTRACTION_RUNS:

            print(
                f"  Already has "
                f"{existing_count}/"
                f"{EXTRACTION_RUNS} "
                f"successful extractions."
            )

            print(
                "  Skipping."
            )

            continue

        missing = (
            EXTRACTION_RUNS
            - existing_count
        )

        print(
            f"  Existing: "
            f"{existing_count}/"
            f"{EXTRACTION_RUNS}"
        )

        print(
            f"  Missing: {missing}"
        )

        # ----------------------------------------------------
        # Run extraction
        # ----------------------------------------------------

        for extraction_run in range(
            1,
            EXTRACTION_RUNS + 1,
        ):

            # Don't duplicate existing run numbers
            # already_exists = any(
            #     result.get(
            #         "image_id"
            #     ) == image_id
            #     and result.get(
            #         "baseline_run_number"
            #     ) == run_number
            #     and result.get(
            #         "extraction_run_number"
            #     ) == extraction_run
            #     and result.get(
            #         "status"
            #     ) == "success"
            #     for result in extraction_results
            # )
            already_exists = any(
                result.get("image_id") == image_id
                and result.get("baseline_run_number") == run_number
                and result.get("extraction_run_number") == extraction_run
                and result.get("status") == "success"
                and result.get("timestamp_utc")
                and datetime.fromisoformat(
                    result["timestamp_utc"]
                ) >= LATE_NIGHT_CUTOFF_UTC
                for result in extraction_results
            )

            if already_exists:
                continue

            print(
                f"  Extraction "
                f"{extraction_run}/"
                f"{EXTRACTION_RUNS}...",
                end=" ",
                flush=True,
            )

            start_time = time.time()

            try:

                result_data = (
                    extract_semantics(
                        caption
                    )
                )

                elapsed = (
                    time.time()
                    - start_time
                )

                result = {
                    "image_id": image_id,
                    "baseline_run_number": run_number,
                    "extraction_run_number": (
                        extraction_run
                    ),
                    "image_file": baseline[
                        "image_file"
                    ],
                    "caption": caption,
                    "model": EXTRACTION_MODEL,
                    "temperature": (
                        EXTRACTION_TEMPERATURE
                    ),
                    "timestamp_utc": (
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    ),
                    "generation_time_seconds": (
                        round(
                            elapsed,
                            3,
                        )
                    ),
                    "attempts": result_data[
                        "attempts"
                    ],
                    "csr": result_data[
                        "csr"
                    ],
                    "raw_output": result_data[
                        "raw_output"
                    ],
                    "status": "success",
                }

                print("SUCCESS")

            except RuntimeError as error:

                # ------------------------------------------------
                # Daily API quota exhausted
                # ------------------------------------------------
                last_error = error
                if (
                    "daily limit"
                    in str(error).lower()
                ):

                    print()
                    print(
                        "DAILY OPENROUTER LIMIT REACHED."
                    )

                    print(
                        "Stopping experiment safely."
                    )

                    print(
                        "Existing results have been saved."
                    )

                    return

                # ------------------------------------------------
                # Other RuntimeError
                # ------------------------------------------------

                elapsed = (
                    time.time()
                    - start_time
                )

                result = {
                    "image_id": image_id,
                    "baseline_run_number": run_number,
                    "extraction_run_number": (
                        extraction_run
                    ),
                    "image_file": baseline[
                        "image_file"
                    ],
                    "caption": caption,
                    "model": EXTRACTION_MODEL,
                    "temperature": (
                        EXTRACTION_TEMPERATURE
                    ),
                    "timestamp_utc": (
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    ),
                    "generation_time_seconds": (
                        round(
                            elapsed,
                            3,
                        )
                    ),
                    "attempts": 3,
                    "csr": None,
                    "raw_output": None,
                    "status": "error",
                    "error": str(error),
                }

                print("FAILED")

                print(
                    f"    Error: {error}"
                )

            except Exception as error:

                # ------------------------------------------------
                # Ordinary extraction error
                # ------------------------------------------------

                elapsed = (
                    time.time()
                    - start_time
                )

                result = {
                    "image_id": image_id,
                    "baseline_run_number": run_number,
                    "extraction_run_number": (
                        extraction_run
                    ),
                    "image_file": baseline[
                        "image_file"
                    ],
                    "caption": caption,
                    "model": EXTRACTION_MODEL,
                    "temperature": (
                        EXTRACTION_TEMPERATURE
                    ),
                    "timestamp_utc": (
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    ),
                    "generation_time_seconds": (
                        round(
                            elapsed,
                            3,
                        )
                    ),
                    "attempts": 3,
                    "csr": None,
                    "raw_output": None,
                    "status": "error",
                    "error": str(error),
                }

                print("FAILED")

                print(
                    f"    Error: {error}"
                )

            extraction_results.append(
                result
            )

            # ------------------------------------------------
            # Save after every extraction
            # ------------------------------------------------

            save_results(
                extraction_results
            )

            # ------------------------------------------------
            # Save after every extraction
            # ------------------------------------------------

            

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    successful = sum(
        result.get("status")
        == "success"
        for result in extraction_results
    )

    failed = sum(
        result.get("status")
        == "error"
        for result in extraction_results
    )

    print()
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    print(
        f"Successful extractions: "
        f"{successful}"
    )

    print(
        f"Failed extractions: "
        f"{failed}"
    )

    print(
        f"JSON: "
        f"{EXTRACTION_JSON_OUTPUT}"
    )

    print(
        f"CSV : "
        f"{EXTRACTION_CSV_OUTPUT}"
    )

    print("=" * 60)


if __name__ == "__main__":
    run_extraction_experiment()
