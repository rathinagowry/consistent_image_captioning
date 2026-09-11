
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ============================================================
# IMPORTS
# ============================================================

from src.config import (
    PROMPT,
    TEMPERATURE,
    MODEL_NAME,
)

from src.model import generate_caption
from src.semantic_extractor import extract_semantics
from src.canonicalizer import canonicalize_csr
from src.semantic_aligner import (
    align_csr_pair,
    create_pairs,
)
from src.consensus_builder import build_consensus
from src.csr_validator import validate_csr
from src.deterministic_renderer_v3 import render_csr


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

PIPELINE_DIR = (
    PROJECT_ROOT
    / "results"
    / "pipeline"
)

FROZEN_CSR_DIR = (
    PROJECT_ROOT
    / "results"
    / "frozen_consensus"
)


# ============================================================
# FILE UTILITIES
# ============================================================

def save_json(path, data):
    """
    Save data as formatted JSON.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# FROZEN CONSENSUS UTILITIES
# ============================================================

def get_frozen_csr_path(image_id):
    """
    Return the persistent consensus CSR path for an image.
    """

    return (
        FROZEN_CSR_DIR
        / image_id
        / "consensus.json"
    )


def load_frozen_consensus(image_id):
    """
    Load a previously established consensus CSR.

    Returns:
        dict if a valid frozen consensus exists.
        None if no frozen consensus exists.
    """

    path = get_frozen_csr_path(
        image_id
    )

    if not path.exists():
        return None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid frozen consensus file: "
            f"{path}"
        )

    if data.get("image_id") != image_id:
        raise ValueError(
            f"Frozen consensus image_id mismatch "
            f"for {image_id}: {path}"
        )

    if "csr" not in data:
        raise ValueError(
            f"Frozen consensus missing csr: "
            f"{path}"
        )

    if not isinstance(data["csr"], dict):
        raise ValueError(
            f"Frozen consensus csr must be an object: "
            f"{path}"
        )

    return data


def save_frozen_consensus(
    image_id,
    consensus_item,
):
    """
    Persist the established consensus CSR for an image.

    This file is intentionally separate from normal
    pipeline execution artifacts.

    Once created, subsequent executions reuse it unless
    --rebuild is explicitly requested.
    """

    path = get_frozen_csr_path(
        image_id
    )

    save_json(
        path,
        consensus_item,
    )

    return path


# ============================================================
# IMAGE SELECTION
# ============================================================

def get_images(
    input_dir=None,
    image_names=None,
):
    """
    Select images for the pipeline.

    Either:
        --input_dir
    or:
        --images
    """

    if image_names:

        images = []

        for name in image_names:

            path = Path(name)

            # If only a filename was provided,
            # look inside the configured image directory.
            if not path.exists():

                from src.config import IMAGE_DIR

                path = (
                    Path(IMAGE_DIR)
                    / name
                )

            if not path.exists():

                raise FileNotFoundError(
                    f"Image not found: {name}"
                )

            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:

                raise ValueError(
                    f"Unsupported image format: "
                    f"{path}"
                )

            images.append(path)

        return images

    if input_dir:

        directory = Path(input_dir)

    else:

        from src.config import IMAGE_DIR

        directory = Path(IMAGE_DIR)

    if not directory.exists():

        raise FileNotFoundError(
            f"Image directory not found: "
            f"{directory}"
        )

    images = [
        path
        for path in sorted(
            directory.iterdir()
        )
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ]

    if not images:

        raise ValueError(
            f"No supported images found in "
            f"{directory}"
        )

    return images


# ============================================================
# STEP 1 — BASELINE CAPTION GENERATION
# ============================================================

def generate_baseline_captions(
    image_path,
    runs,
):
    """
    Generate the requested number of captions for one image.

    This directly uses generate_caption() from model.py.

    No existing experiment results are modified.
    """

    captions = []

    print(
        f"\nGenerating {runs} captions "
        f"for {image_path.name}"
    )

    for run_number in range(
        1,
        runs + 1,
    ):

        print(
            f"  Run {run_number}/{runs}...",
            end=" ",
            flush=True,
        )

        try:

            result = generate_caption(
                image_path=str(
                    image_path
                ),
                prompt=PROMPT,
                temperature=TEMPERATURE,
            )

            caption = result.get(
                "output"
            )

            if not caption:

                print("FAILED")

                continue

            captions.append(
                {
                    "image_id":
                        image_path.stem,

                    "image_file":
                        image_path.name,

                    "baseline_run_number":
                        run_number,

                    "extraction_run_number":
                        run_number,

                    "caption":
                        caption,

                    "model":
                        MODEL_NAME,

                    "temperature":
                        TEMPERATURE,

                    "status":
                        "success",

                    "attempts":
                        result.get(
                            "attempts"
                        ),
                }
            )

            print("OK")

        except Exception as error:

            print(
                f"FAILED: {error}"
            )

    if len(captions) < runs:

        raise RuntimeError(
            f"Only {len(captions)}/{runs} "
            f"caption runs succeeded for "
            f"{image_path.name}."
        )

    return captions


# ============================================================
# STEP 2 — SEMANTIC EXTRACTION
# ============================================================

def extract_csrs(records):
    """
    Extract a CSR from every generated caption.

    Canonicalization is performed immediately after
    extraction.

    original_csr:
        Raw extracted CSR.

    csr:
        Canonical CSR used by alignment and consensus.
    """

    print(
        "\nSemantic extraction..."
    )

    extracted_records = []

    for record in records:

        print(
            f"  {record['image_id']} "
            f"run {record['baseline_run_number']}...",
            end=" ",
            flush=True,
        )

        try:

            extraction = extract_semantics(
                record["caption"]
            )

            original_csr = extraction["csr"]

            canonical_csr = canonicalize_csr(
                original_csr,
                caption=record.get(
                    "caption"
                ),
            )

            new_record = dict(record)

            new_record["original_csr"] = (
                original_csr
            )

            new_record["csr"] = (
                canonical_csr
            )

            new_record["extraction_attempts"] = (
                extraction.get(
                    "attempts"
                )
            )

            new_record["raw_extraction_output"] = (
                extraction.get(
                    "raw_output"
                )
            )

            extracted_records.append(
                new_record
            )

            print("OK")

        except Exception as error:

            print(
                f"FAILED: {error}"
            )

    return extracted_records


# ============================================================
# STEP 3 — CANONICALIZATION
# ============================================================

# Canonicalization is intentionally performed inside
# extract_csrs().
#
# Therefore there is NO second canonicalization step.
#
# extracted_records contain:
#
#   original_csr -> raw extracted CSR
#   csr          -> canonical CSR


# ============================================================
# STEP 4 — SEMANTIC ALIGNMENT
# ============================================================

def run_alignment(records):
    """
    Create all within-image CSR pairs and perform
    semantic alignment.
    """

    print(
        "\nSemantic alignment..."
    )

    pairs = create_pairs(
        records
    )

    print(
        f"  Created {len(pairs)} CSR pairs."
    )

    results = []

    for number, pair in enumerate(
        pairs,
        start=1,
    ):

        pair_id = pair[
            "pair_id"
        ]

        image_id = pair[
            "image_id"
        ]

        record_a = pair[
            "record_a"
        ]

        record_b = pair[
            "record_b"
        ]

        print(
            f"  [{number}/{len(pairs)}] "
            f"{pair_id}"
        )

        try:

            alignment = align_csr_pair(
                record_a["csr"],
                record_b["csr"],
            )

            if alignment is None:

                results.append(
                    {
                        "pair_id":
                            pair_id,

                        "image_id":
                            image_id,

                        "record_a": {
                            "baseline_run_number":
                                record_a.get(
                                    "baseline_run_number"
                                ),

                            "extraction_run_number":
                                record_a.get(
                                    "extraction_run_number"
                                ),
                        },

                        "record_b": {
                            "baseline_run_number":
                                record_b.get(
                                    "baseline_run_number"
                                ),

                            "extraction_run_number":
                                record_b.get(
                                    "extraction_run_number"
                                ),
                        },

                        "status":
                            "failed",

                        "error":
                            "No valid alignment response.",
                    }
                )

                print(
                    "    FAILED"
                )

                continue

            results.append(
                {
                    "pair_id":
                        pair_id,

                    "image_id":
                        image_id,

                    "record_a": {
                        "baseline_run_number":
                            record_a.get(
                                "baseline_run_number"
                            ),

                        "extraction_run_number":
                            record_a.get(
                                "extraction_run_number"
                            ),
                    },

                    "record_b": {
                        "baseline_run_number":
                            record_b.get(
                                "baseline_run_number"
                            ),

                        "extraction_run_number":
                            record_b.get(
                                "extraction_run_number"
                            ),
                    },

                    "status":
                        "success",

                    "alignment":
                        alignment,
                }
            )

            print(
                "    OK"
            )

        except Exception as error:

            results.append(
                {
                    "pair_id":
                        pair_id,

                    "image_id":
                        image_id,

                    "record_a": {
                        "baseline_run_number":
                            record_a.get(
                                "baseline_run_number"
                            ),

                        "extraction_run_number":
                            record_a.get(
                                "extraction_run_number"
                            ),
                    },

                    "record_b": {
                        "baseline_run_number":
                            record_b.get(
                                "baseline_run_number"
                            ),

                        "extraction_run_number":
                            record_b.get(
                                "extraction_run_number"
                            ),
                    },

                    "status":
                        "failed",

                    "error":
                        str(error),
                }
            )

            print(
                f"    FAILED: {error}"
            )

    failed_alignments = [
        result
        for result in results
        if result.get(
            "status"
        ) != "success"
    ]

    if failed_alignments:

        raise RuntimeError(
            f"{len(failed_alignments)}/{len(results)} "
            "semantic alignment pairs failed."
        )

    return results


# ============================================================
# STEP 5 — CONSENSUS
# ============================================================

def build_consensus_csrs(
    records,
    alignment_results,
):
    """
    Build one consensus CSR for each image.
    """

    print(
        "\nConsensus CSR construction..."
    )

    grouped = {}

    for record in records:

        grouped.setdefault(
            record["image_id"],
            [],
        ).append(record)

    consensus_results = []

    for image_id, image_records in (
        grouped.items()
    ):

        image_alignments = [
            result
            for result in alignment_results
            if result.get(
                "image_id"
            ) == image_id
        ]

        print(
            f"  {image_id}..."
        )

        consensus_csr, provenance = (
            build_consensus(
                image_id,
                image_records,
                image_alignments,
            )
        )

        consensus_results.append(
            {
                "image_id":
                    image_id,

                "csr":
                    consensus_csr,

                "provenance":
                    provenance,
            }
        )

        print(
            "    OK"
        )

    return consensus_results


# ============================================================
# STEP 6 — VALIDATION
# ============================================================

def validate_consensus_csrs(
    consensus_results,
):
    """
    Validate each consensus CSR.

    A CSR is valid when it contains no ERROR-level
    validation issues.
    """

    print(
        "\nCSR validation..."
    )

    validation_results = []

    for item in consensus_results:

        image_id = item[
            "image_id"
        ]

        csr = item[
            "csr"
        ]

        try:

            csr_for_validation = dict(
                csr
            )

            csr_for_validation.setdefault(
                "image_id",
                image_id,
            )

            issues = validate_csr(
                csr_for_validation
            )

            errors = [
                current_issue
                for current_issue in issues
                if current_issue.get(
                    "severity"
                ) == "ERROR"
            ]

            warnings = [
                current_issue
                for current_issue in issues
                if current_issue.get(
                    "severity"
                ) == "WARNING"
            ]

            infos = [
                current_issue
                for current_issue in issues
                if current_issue.get(
                    "severity"
                ) == "INFO"
            ]

            valid = len(errors) == 0

            validation_results.append(
                {
                    "image_id":
                        image_id,

                    "valid":
                        valid,

                    "status":
                        (
                            "PASS"
                            if valid
                            else "FAIL"
                        ),

                    "error_count":
                        len(errors),

                    "warning_count":
                        len(warnings),

                    "info_count":
                        len(infos),

                    "issues":
                        issues,
                }
            )

            print(
                f"  {image_id}: "
                f"{'PASS' if valid else 'FAIL'}"
            )

            for current_issue in errors:

                print(
                    f"    [ERROR] "
                    f"{current_issue['message']}"
                )

            if warnings:

                print(
                    f"    Warnings: "
                    f"{len(warnings)}"
                )

            if infos:

                print(
                    f"    Info: "
                    f"{len(infos)}"
                )

        except Exception as error:

            print(
                f"  {image_id}: "
                f"FAIL — {error}"
            )

            validation_results.append(
                {
                    "image_id":
                        image_id,

                    "valid":
                        False,

                    "status":
                        "FAIL",

                    "error_count":
                        1,

                    "warning_count":
                        0,

                    "info_count":
                        0,

                    "issues":
                        [],

                    "error":
                        str(error),
                }
            )

    return validation_results


# ============================================================
# STEP 7 — DETERMINISTIC RENDERING
# ============================================================

def render_consensus_csrs(
    consensus_results,
    validation_results,
):
    """
    Render validated consensus CSRs using the deterministic
    renderer.
    """

    print(
        "\nDeterministic rendering..."
    )

    validation_map = {
        result["image_id"]:
            result["valid"]
        for result in validation_results
    }

    final_results = []

    for item in consensus_results:

        image_id = item[
            "image_id"
        ]

        csr = item[
            "csr"
        ]

        if not validation_map.get(
            image_id,
            False,
        ):

            print(
                f"  {image_id}: SKIPPED "
                f"(invalid CSR)"
            )

            final_results.append(
                {
                    "image_id":
                        image_id,

                    "status":
                        "invalid_csr",

                    "caption":
                        None,
                }
            )

            continue

        try:

            caption = render_csr(
                csr
            )

            final_results.append(
                {
                    "image_id":
                        image_id,

                    "status":
                        "success",

                    "caption":
                        caption,

                    "csr":
                        csr,
                }
            )

            print(
                f"\n  {image_id}:"
            )

            print(
                f"    {caption}"
            )

        except Exception as error:

            print(
                f"  {image_id}: "
                f"FAILED — {error}"
            )

            final_results.append(
                {
                    "image_id":
                        image_id,

                    "status":
                        "failed",

                    "caption":
                        None,

                    "error":
                        str(error),
                }
            )

    return final_results


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    images,
    runs,
    rebuild=False,
):
    """
    Run the complete end-to-end pipeline.

    Normal execution:
        - If a frozen CSR exists, reuse it.
        - Otherwise establish a new consensus CSR.

    Rebuild execution:
        - Ignore the existing frozen CSR.
        - Run the stochastic stages again.
        - Replace the frozen CSR with the new consensus.
    """

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    print()
    print("=" * 70)
    print(
        "IMAGE CONSISTENCY — END-TO-END PIPELINE"
    )
    print("=" * 70)

    print(
        f"Model       : {MODEL_NAME}"
    )

    print(
        f"Prompt      : {PROMPT}"
    )

    print(
        f"Temperature : {TEMPERATURE}"
    )

    print(
        f"Runs/image  : {runs}"
    )

    print(
        f"Images      : {len(images)}"
    )

    print(
        f"Mode        : "
        f"{'REBUILD' if rebuild else 'NORMAL'}"
    )

    print("=" * 70)

    all_final_results = []

    for image_number, image_path in enumerate(
        images,
        start=1,
    ):

        print()
        print(
            "#" * 70
        )

        print(
            f"IMAGE {image_number}/{len(images)}: "
            f"{image_path.name}"
        )

        print(
            "#" * 70
        )

        image_id = image_path.stem

        image_dir = (
            PIPELINE_DIR
            / image_id
        )

        image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        used_frozen_consensus = False

        try:

            # =================================================
            # CHECK FROZEN CONSENSUS
            # =================================================

            frozen_consensus = None

            if not rebuild:

                frozen_consensus = (
                    load_frozen_consensus(
                        image_id
                    )
                )

            # =================================================
            # PATH A — USE FROZEN CONSENSUS
            # =================================================

            if frozen_consensus is not None:

                used_frozen_consensus = True

                print()
                print(
                    "Frozen consensus found."
                )

                print(
                    "Skipping stochastic "
                    "caption generation."
                )

                print(
                    "Skipping semantic extraction."
                )

                print(
                    "Skipping semantic alignment."
                )

                print(
                    "Using established consensus CSR."
                )

                consensus_results = [
                    frozen_consensus
                ]

                # Save a trace copy for this execution.
                save_json(
                    image_dir
                    / "consensus.json",
                    consensus_results,
                )

                # ------------------------------------------------
                # Validation
                # ------------------------------------------------

                validation_results = (
                    validate_consensus_csrs(
                        consensus_results
                    )
                )

                save_json(
                    image_dir
                    / "validation.json",
                    validation_results,
                )

                # ------------------------------------------------
                # Deterministic rendering
                # ------------------------------------------------

                final_results = (
                    render_consensus_csrs(
                        consensus_results,
                        validation_results,
                    )
                )

                save_json(
                    image_dir
                    / "final.json",
                    final_results,
                )

                all_final_results.extend(
                    final_results
                )

                print()
                print(
                    f"✓ COMPLETED: "
                    f"{image_path.name}"
                )

                print(
                    "  Mode: frozen consensus"
                )

                continue

            # =================================================
            # PATH B — ESTABLISH NEW CONSENSUS
            # =================================================

            print()

            if rebuild:

                print(
                    "Rebuild requested."
                )

                print(
                    "Existing frozen consensus "
                    "will be replaced."
                )

            else:

                print(
                    "No frozen consensus found."
                )

                print(
                    "Establishing a new consensus CSR."
                )

            # ------------------------------------------------
            # 1. Baseline caption generation
            # ------------------------------------------------

            baseline_records = (
                generate_baseline_captions(
                    image_path,
                    runs,
                )
            )

            save_json(
                image_dir
                / "baseline.json",
                baseline_records,
            )

            # ------------------------------------------------
            # 2. Semantic extraction
            # ------------------------------------------------

            extracted_records = (
                extract_csrs(
                    baseline_records
                )
            )

            save_json(
                image_dir
                / "extractions.json",
                extracted_records,
            )

            if len(extracted_records) < runs:

                raise RuntimeError(
                    f"Only "
                    f"{len(extracted_records)}/{runs} "
                    f"CSR extractions succeeded."
                )

            # ------------------------------------------------
            # 3. Canonicalization
            # ------------------------------------------------

            # Canonicalization is performed inside
            # extract_csrs().
            #
            # original_csr = raw extracted CSR
            # csr          = canonical CSR

            canonical_records = (
                extracted_records
            )

            save_json(
                image_dir
                / "canonical_csrs.json",
                canonical_records,
            )

            # ------------------------------------------------
            # 4. Semantic alignment
            # ------------------------------------------------

            alignment_results = (
                run_alignment(
                    canonical_records
                )
            )

            save_json(
                image_dir
                / "alignments.json",
                alignment_results,
            )

            # ------------------------------------------------
            # 5. Consensus
            # ------------------------------------------------

            consensus_results = (
                build_consensus_csrs(
                    canonical_records,
                    alignment_results,
                )
            )

            if len(consensus_results) != 1:

                raise RuntimeError(
                    f"Expected exactly one consensus "
                    f"CSR for {image_id}, got "
                    f"{len(consensus_results)}."
                )

            save_json(
                image_dir
                / "consensus.json",
                consensus_results,
            )

            # ------------------------------------------------
            # 5a. Freeze consensus
            # ------------------------------------------------

            frozen_path = (
                save_frozen_consensus(
                    image_id,
                    consensus_results[0],
                )
            )

            print()
            print(
                "  Frozen consensus saved:"
            )

            print(
                f"    {frozen_path}"
            )

            # ------------------------------------------------
            # 6. Validation
            # ------------------------------------------------

            validation_results = (
                validate_consensus_csrs(
                    consensus_results
                )
            )

            save_json(
                image_dir
                / "validation.json",
                validation_results,
            )

            # ------------------------------------------------
            # 7. Deterministic rendering
            # ------------------------------------------------

            final_results = (
                render_consensus_csrs(
                    consensus_results,
                    validation_results,
                )
            )

            save_json(
                image_dir
                / "final.json",
                final_results,
            )

            all_final_results.extend(
                final_results
            )

            print()
            print(
                f"✓ COMPLETED: "
                f"{image_path.name}"
            )

            print(
                "  Mode: newly established consensus"
            )

        except Exception as error:

            print()
            print(
                f"✗ FAILED: "
                f"{image_path.name}"
            )

            print(
                f"  Error: {error}"
            )

            all_final_results.append(
                {
                    "image_id":
                        image_id,

                    "image_file":
                        image_path.name,

                    "status":
                        "failed",

                    "caption":
                        None,

                    "error":
                        str(error),

                    "used_frozen_consensus":
                        used_frozen_consensus,
                }
            )

    # ========================================================
    # SAVE GLOBAL RESULT
    # ========================================================

    pipeline_summary = {
        "timestamp":
            timestamp,

        "model":
            MODEL_NAME,

        "prompt":
            PROMPT,

        "temperature":
            TEMPERATURE,

        "runs_per_image":
            runs,

        "rebuild":
            rebuild,

        "images_processed":
            len(images),

        "successful":
            sum(
                1
                for result
                in all_final_results
                if result.get(
                    "status"
                ) == "success"
            ),

        "failed":
            sum(
                1
                for result
                in all_final_results
                if result.get(
                    "status"
                ) != "success"
            ),

        "results":
            all_final_results,
    }

    save_json(
        PIPELINE_DIR
        / "pipeline_summary.json",
        pipeline_summary,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"Images processed : "
        f"{len(images)}"
    )

    print(
        f"Successful       : "
        f"{pipeline_summary['successful']}"
    )

    print(
        f"Failed           : "
        f"{pipeline_summary['failed']}"
    )

    print()
    print(
        "Output directory:"
    )

    print(
        f"  {PIPELINE_DIR}"
    )

    print(
        "Frozen CSR directory:"
    )

    print(
        f"  {FROZEN_CSR_DIR}"
    )

    print("=" * 70)

    return pipeline_summary


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Image Consistency "
            "pipeline."
        )
    )

    parser.add_argument(
        "--images",
        nargs="+",
        help=(
            "Specific image filenames. "
            "Example: test_001.png test_003.jpg"
        ),
    )

    parser.add_argument(
        "--input_dir",
        help=(
            "Directory containing images."
        ),
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help=(
            "Number of successful caption/CSR "
            "runs per image when establishing "
            "or rebuilding consensus. "
            "Default: 3"
        ),
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Ignore existing frozen consensus CSRs, "
            "re-run the stochastic stages, and "
            "replace the frozen consensus."
        ),
    )

    args = parser.parse_args()

    if args.runs < 1:

        parser.error(
            "--runs must be at least 1."
        )

    if (
        args.images
        and args.input_dir
    ):

        parser.error(
            "Use either --images or "
            "--input_dir, not both."
        )

    images = get_images(
        input_dir=args.input_dir,
        image_names=args.images,
    )

    run_pipeline(
        images,
        args.runs,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()

