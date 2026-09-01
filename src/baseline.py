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


def run_baseline():

    images = get_image_files()

    if not images:
        raise ValueError(
            f"No images found in {IMAGE_DIR}"
        )

    print("=" * 60)
    print("IMAGE-TO-TEXT BASELINE EXPERIMENT")
    print("=" * 60)

    print(f"Model       : {MODEL_NAME}")
    print(f"Prompt      : {PROMPT}")
    print(f"Temperature : {TEMPERATURE}")
    print(f"Images      : {len(images)}")
    print(f"Runs/image  : {NUM_RUNS}")
    print(
        f"Total calls : {len(images) * NUM_RUNS}"
    )

    print("=" * 60)

    results = []

    for image_index, image_path in enumerate(images, start=1):

        image_id = image_path.stem

        print()
        print(
            f"[Image {image_index}/{len(images)}] "
            f"{image_path.name}"
        )

        for run_number in range(1, NUM_RUNS + 1):

            print(
                f"  Run {run_number}/{NUM_RUNS}...",
                end=" ",
                flush=True,
            )

            start_time = time.time()

            try:

                generation_result = generate_caption(
                    image_path=str(image_path),
                    prompt=PROMPT,
                    temperature=TEMPERATURE,
                )
                caption = generation_result["output"]           
                attempts = generation_result["attempts"]

                elapsed_time = time.time() - start_time

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

                elapsed_time = time.time() - start_time

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
                print(f"    Error: {error}")

            results.append(result)

            # Save after every generation so that
            # partial results aren't lost.
            save_results(results)

    print()
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    print(f"JSON: {JSON_OUTPUT}")
    print(f"CSV : {CSV_OUTPUT}")
    print(
        f"Successful generations: "
        f"{sum(r['status'] == 'success' for r in results)}"
    )

    print(
        f"Failed generations: "
        f"{sum(r['status'] == 'error' for r in results)}"
    )


if __name__ == "__main__":
    run_baseline()