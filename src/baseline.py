import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    IMAGE_DIR,
    RESULT_DIR,
    JSON_OUTPUT,
    CSV_OUTPUT,
    PROMPT,
    NUM_RUNS,
    TEMPERATURE,
    MODEL_NAME,
)

from src.model import generate_caption


def get_image_files():
    """
    Return all supported images from the image directory.
    """

    image_dir = Path(IMAGE_DIR)

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    images = [
        path
        for path in sorted(image_dir.iterdir())
        if path.suffix.lower() in supported_extensions
    ]

    return images


def load_existing_results():
    """
    Load previously saved experiment results.

    If no previous results exist, return an empty list.
    """

    output_path = Path(JSON_OUTPUT)

    if not output_path.exists():
        return []

    try:

        with open(
            output_path,
            "r",
            encoding="utf-8",
        ) as file:

            results = json.load(file)

        if not isinstance(results, list):
            print(
                "Warning: Existing JSON does not contain "
                "a list. Starting with empty results."
            )
            return []

        return results

    except json.JSONDecodeError:

        print(
            "Warning: Existing JSON could not be parsed. "
            "Starting with empty results."
        )

        return []


def save_results(results):
    """
    Save experiment results as JSON and CSV.
    """

    Path(RESULT_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------
    # Save JSON
    # -------------------------

    with open(
        JSON_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # -------------------------
    # Save CSV
    # -------------------------

    dataframe = pd.DataFrame(results)

    dataframe.to_csv(
        CSV_OUTPUT,
        index=False,
        encoding="utf-8",
    )


def get_successful_results_for_image(
    results,
    image_id,
):
    """
    Return successful generations for a particular image.
    """

    return [
        result
        for result in results
        if result.get("image_id") == image_id
        and result.get("status") == "success"
        and result.get("output")
    ]


def get_next_run_number(
    results,
    image_id,
):
    """
    Return the next unused run number for an image.

    This preserves existing run numbers and avoids
    overwriting previous records.
    """

    run_numbers = [
        result.get("run_number")
        for result in results
        if result.get("image_id") == image_id
        and isinstance(
            result.get("run_number"),
            int,
        )
    ]

    if not run_numbers:
        return 1

    return max(run_numbers) + 1


def run_baseline():

    images = get_image_files()

    if not images:
        raise ValueError(
            f"No images found in {IMAGE_DIR}"
        )

    # ------------------------------------------------
    # Load previous results
    # ------------------------------------------------

    results = load_existing_results()

    # ------------------------------------------------
    # Experiment header
    # ------------------------------------------------

    print("=" * 60)
    print("IMAGE-TO-TEXT BASELINE EXPERIMENT")
    print("=" * 60)

    print(f"Model       : {MODEL_NAME}")
    print(f"Prompt      : {PROMPT}")
    print(f"Temperature : {TEMPERATURE}")
    print(f"Images      : {len(images)}")
    print(f"Runs/image  : {NUM_RUNS}")

    print(
        f"Total target generations : "
        f"{len(images) * NUM_RUNS}"
    )

    print("=" * 60)

    # ------------------------------------------------
    # Process each image
    # ------------------------------------------------

    for image_index, image_path in enumerate(
        images,
        start=1,
    ):

        image_id = image_path.stem

        print()
        print(
            f"[Image {image_index}/{len(images)}] "
            f"{image_path.name}"
        )

        # ------------------------------------------------
        # Find existing successful generations
        # ------------------------------------------------

        successful_results = (
            get_successful_results_for_image(
                results,
                image_id,
            )
        )

        successful_count = len(
            successful_results
        )

        # ------------------------------------------------
        # Image already complete
        # ------------------------------------------------

        if successful_count >= NUM_RUNS:

            print(
                f"  Already has "
                f"{successful_count}/{NUM_RUNS} "
                f"successful runs."
            )

            print("  Skipping image.")

            continue

        # ------------------------------------------------
        # Determine missing runs
        # ------------------------------------------------

        missing_runs = (
            NUM_RUNS - successful_count
        )

        print(
            f"  Existing successful runs : "
            f"{successful_count}/{NUM_RUNS}"
        )

        print(
            f"  Missing successful runs  : "
            f"{missing_runs}"
        )

        # ------------------------------------------------
        # Generate only missing runs
        # ------------------------------------------------

        for _ in range(missing_runs):

            run_number = get_next_run_number(
                results,
                image_id,
            )

            print(
                f"  Run {run_number} "
                f"(missing run)...",
                end=" ",
                flush=True,
            )

            start_time = time.time()

            try:

                generation_result = (
                    generate_caption(
                        image_path=str(
                            image_path
                        ),
                        prompt=PROMPT,
                        temperature=TEMPERATURE,
                    )
                )

                caption = generation_result[
                    "output"
                ]

                attempts = generation_result[
                    "attempts"
                ]

                elapsed_time = (
                    time.time() - start_time
                )

                result = {
                    "image_id": image_id,
                    "image_file": image_path.name,
                    "model": MODEL_NAME,
                    "prompt": PROMPT,
                    "temperature": TEMPERATURE,
                    "run_number": run_number,
                    "attempts": attempts,
                    "timestamp_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "generation_time_seconds": round(
                        elapsed_time,
                        3,
                    ),
                    "output": caption,
                    "status": "success",
                }

                print("SUCCESS")

            except Exception as error:

                elapsed_time = (
                    time.time() - start_time
                )

                result = {
                    "image_id": image_id,
                    "image_file": image_path.name,
                    "model": MODEL_NAME,
                    "prompt": PROMPT,
                    "temperature": TEMPERATURE,
                    "run_number": run_number,
                    "attempts": 3,
                    "timestamp_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "generation_time_seconds": round(
                        elapsed_time,
                        3,
                    ),
                    "output": None,
                    "status": "error",
                    "error": str(error),
                }

                print("FAILED")
                print(
                    f"    Error: {error}"
                )

            # ------------------------------------------------
            # Add result
            # ------------------------------------------------

            results.append(result)

            # ------------------------------------------------
            # Save immediately
            # ------------------------------------------------

            save_results(results)

    # ------------------------------------------------
    # Final summary
    # ------------------------------------------------

    successful_total = sum(
        result.get("status") == "success"
        for result in results
    )

    failed_total = sum(
        result.get("status") == "error"
        for result in results
    )

    print()
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    print(
        f"JSON: {JSON_OUTPUT}"
    )

    print(
        f"CSV : {CSV_OUTPUT}"
    )

    print(
        f"Successful generations: "
        f"{successful_total}"
    )

    print(
        f"Failed generations: "
        f"{failed_total}"
    )

    print("=" * 60)


if __name__ == "__main__":
    run_baseline()