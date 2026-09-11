import json
from pathlib import Path
from itertools import combinations


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
    / "canonical_stability_analysis.json"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_json(data):
    """
    Convert a JSON object into a deterministic representation
    for exact comparison.
    """

    return json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
    )


def exact_match_rate(values):
    """
    Percentage of records matching the most frequent value.
    """

    if not values:
        return 0.0

    counts = {}

    for value in values:
        key = normalize_json(value)

        counts[key] = (
            counts.get(key, 0) + 1
        )

    maximum = max(
        counts.values()
    )

    return maximum / len(values)


def pairwise_exact_similarity(values):
    """
    Average pairwise exact similarity.

    For 3 runs there are 3 possible pairs.
    """

    if len(values) < 2:
        return 1.0

    similarities = []

    for a, b in combinations(
        values,
        2,
    ):

        if normalize_json(a) == normalize_json(b):
            similarities.append(1.0)
        else:
            similarities.append(0.0)

    return (
        sum(similarities)
        / len(similarities)
    )


def extract_component(
    csr,
    component,
):
    """
    Extract a component from a CSR.
    """

    if component == "entities":
        return csr.get(
            "entities",
            [],
        )

    if component == "attributes":

        attributes = []

        for entity in csr.get(
            "entities",
            [],
        ):

            attributes.extend(
                entity.get(
                    "attributes",
                    [],
                )
            )

        return sorted(
            attributes
        )

    if component == "actions":
        return csr.get(
            "actions",
            [],
        )

    if component == "relations":
        return csr.get(
            "relations",
            [],
        )

    if component == "scene":
        return csr.get(
            "scene",
            {},
        )

    return None


# ============================================================
# GROUP RECORDS BY IMAGE
# ============================================================

def group_by_image(
    records
):

    grouped = {}

    for record in records:

        image_id = record[
            "image_id"
        ]

        if image_id not in grouped:
            grouped[image_id] = []

        grouped[
            image_id
        ].append(record)

    return grouped


# ============================================================
# ANALYZE ONE IMAGE
# ============================================================

def analyze_image(
    records
):

    result = {
        "num_runs": len(records),

        "raw": {},
        "canonical": {},

        "improvement": {},
    }

    components = [
        "entities",
        "attributes",
        "actions",
        "relations",
        "scene",
    ]

    # --------------------------------------------------------
    # Raw vs canonical
    # --------------------------------------------------------

    for component in components:

        raw_values = [
            extract_component(
                record[
                    "original_csr"
                ],
                component,
            )
            for record in records
        ]

        canonical_values = [
            extract_component(
                record[
                    "canonical_csr"
                ],
                component,
            )
            for record in records
        ]

        raw_rate = exact_match_rate(
            raw_values
        )

        canonical_rate = (
            exact_match_rate(
                canonical_values
            )
        )

        raw_pairwise = (
            pairwise_exact_similarity(
                raw_values
            )
        )

        canonical_pairwise = (
            pairwise_exact_similarity(
                canonical_values
            )
        )

        result["raw"][
            component
        ] = {
            "exact_match_rate":
                raw_rate,

            "pairwise_similarity":
                raw_pairwise,
        }

        result["canonical"][
            component
        ] = {
            "exact_match_rate":
                canonical_rate,

            "pairwise_similarity":
                canonical_pairwise,
        }

        result["improvement"][
            component
        ] = {
            "exact_match_rate_change":
                canonical_rate
                - raw_rate,

            "pairwise_similarity_change":
                canonical_pairwise
                - raw_pairwise,
        }

    # --------------------------------------------------------
    # Whole CSR
    # --------------------------------------------------------

    raw_csrs = [
        record[
            "original_csr"
        ]
        for record in records
    ]

    canonical_csrs = [
        record[
            "canonical_csr"
        ]
        for record in records
    ]

    raw_rate = exact_match_rate(
        raw_csrs
    )

    canonical_rate = (
        exact_match_rate(
            canonical_csrs
        )
    )

    raw_pairwise = (
        pairwise_exact_similarity(
            raw_csrs
        )
    )

    canonical_pairwise = (
        pairwise_exact_similarity(
            canonical_csrs
        )
    )

    result["raw"][
        "whole_csr"
    ] = {
        "exact_match_rate":
            raw_rate,

        "pairwise_similarity":
            raw_pairwise,
    }

    result["canonical"][
        "whole_csr"
    ] = {
        "exact_match_rate":
            canonical_rate,

        "pairwise_similarity":
            canonical_pairwise,
    }

    result["improvement"][
        "whole_csr"
    ] = {
        "exact_match_rate_change":
            canonical_rate
            - raw_rate,

        "pairwise_similarity_change":
            canonical_pairwise
            - raw_pairwise,
    }

    return result


# ============================================================
# OVERALL ANALYSIS
# ============================================================

def analyze_overall(
    image_results
):

    components = [
        "entities",
        "attributes",
        "actions",
        "relations",
        "scene",
        "whole_csr",
    ]

    overall = {}

    for component in components:

        raw_rates = []
        canonical_rates = []

        raw_pairwise = []
        canonical_pairwise = []

        for result in image_results.values():

            raw_rates.append(
                result["raw"][
                    component
                ][
                    "exact_match_rate"
                ]
            )

            canonical_rates.append(
                result["canonical"][
                    component
                ][
                    "exact_match_rate"
                ]
            )

            raw_pairwise.append(
                result["raw"][
                    component
                ][
                    "pairwise_similarity"
                ]
            )

            canonical_pairwise.append(
                result["canonical"][
                    component
                ][
                    "pairwise_similarity"
                ]
            )

        raw_rate = (
            sum(raw_rates)
            / len(raw_rates)
        )

        canonical_rate = (
            sum(canonical_rates)
            / len(canonical_rates)
        )

        raw_pair = (
            sum(raw_pairwise)
            / len(raw_pairwise)
        )

        canonical_pair = (
            sum(canonical_pairwise)
            / len(canonical_pairwise)
        )

        overall[
            component
        ] = {

            "raw_exact_match_rate":
                raw_rate,

            "canonical_exact_match_rate":
                canonical_rate,

            "exact_match_improvement":
                canonical_rate
                - raw_rate,

            "raw_pairwise_similarity":
                raw_pair,

            "canonical_pairwise_similarity":
                canonical_pair,

            "pairwise_similarity_improvement":
                canonical_pair
                - raw_pair,
        }

    return overall


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CANONICAL CSR STABILITY ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        records = json.load(f)

    print(
        f"\nLoaded records: {len(records)}"
    )

    # --------------------------------------------------------
    # Group by image
    # --------------------------------------------------------

    grouped = group_by_image(
        records
    )

    print(
        f"Images found: {len(grouped)}"
    )

    # --------------------------------------------------------
    # Analyze each image
    # --------------------------------------------------------

    image_results = {}

    for image_id, image_records in (
        grouped.items()
    ):

        print(
            f"\nAnalyzing {image_id} "
            f"({len(image_records)} runs)"
        )

        image_results[
            image_id
        ] = analyze_image(
            image_records
        )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall = analyze_overall(
        image_results
    )

    # --------------------------------------------------------
    # Create final result
    # --------------------------------------------------------

    analysis = {
        "experiment": {
            "input_records":
                len(records),

            "images":
                len(grouped),

            "runs_per_image":
                3,

            "comparison":
                "raw CSR vs canonical CSR",
        },

        "per_image":
            image_results,

        "overall":
            overall,
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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
            analysis,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "OVERALL RESULTS"
    )

    print(
        "=" * 70
    )

    for component in [
        "whole_csr",
        "entities",
        "attributes",
        "actions",
        "relations",
        "scene",
    ]:

        data = overall[
            component
        ]

        print(
            f"\n{component.upper()}"
        )

        print(
            "  Raw exact match       : "
            f"{data['raw_exact_match_rate'] * 100:.2f}%"
        )

        print(
            "  Canonical exact match : "
            f"{data['canonical_exact_match_rate'] * 100:.2f}%"
        )

        print(
            "  Improvement            : "
            f"{data['exact_match_improvement'] * 100:+.2f}%"
        )

        print(
            "  Raw pairwise           : "
            f"{data['raw_pairwise_similarity'] * 100:.2f}%"
        )

        print(
            "  Canonical pairwise     : "
            f"{data['canonical_pairwise_similarity'] * 100:.2f}%"
        )

        print(
            "  Pairwise improvement   : "
            f"{data['pairwise_similarity_improvement'] * 100:+.2f}%"
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()