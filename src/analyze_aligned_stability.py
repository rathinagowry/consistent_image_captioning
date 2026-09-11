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
    / "aligned_stability_results.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "aligned_stability_analysis.json"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_json(data):
    """
    Convert JSON data into a deterministic string so that
    equivalent JSON structures can be compared exactly.
    """

    return json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
    )


def exact_match_rate(values):
    """
    Exact match rate = frequency of the most common
    representation divided by total number of runs.
    """

    if not values:
        return 0.0

    counts = {}

    for value in values:

        key = normalize_json(
            value
        )

        counts[key] = (
            counts.get(key, 0) + 1
        )

    maximum = max(
        counts.values()
    )

    return maximum / len(values)


def pairwise_exact_similarity(values):
    """
    Calculate average pairwise exact similarity.

    For 3 runs there are:
        C(3,2) = 3 pairs
    """

    if len(values) < 2:
        return 1.0

    similarities = []

    for value_a, value_b in combinations(
        values,
        2,
    ):

        if (
            normalize_json(value_a)
            == normalize_json(value_b)
        ):
            similarities.append(
                1.0
            )
        else:
            similarities.append(
                0.0
            )

    return (
        sum(similarities)
        / len(similarities)
    )


# ============================================================
# COMPONENT EXTRACTION
# ============================================================

def extract_component(
    csr,
    component,
):
    """
    Extract a specific component from a CSR.
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
# GROUP BY IMAGE
# ============================================================

def group_by_image(
    records
):

    grouped = {}

    for record in records:

        image_id = record[
            "image_id"
        ]

        grouped.setdefault(
            image_id,
            [],
        ).append(record)

    return grouped


# ============================================================
# ANALYZE ONE STAGE
# ============================================================

def analyze_stage(
    csrs
):

    components = [
        "entities",
        "attributes",
        "actions",
        "relations",
        "scene",
    ]

    result = {}

    # --------------------------------------------------------
    # Whole CSR
    # --------------------------------------------------------

    result[
        "whole_csr"
    ] = {
        "exact_match_rate":
            exact_match_rate(
                csrs
            ),

        "pairwise_similarity":
            pairwise_exact_similarity(
                csrs
            ),
    }

    # --------------------------------------------------------
    # Individual components
    # --------------------------------------------------------

    for component in components:

        values = [
            extract_component(
                csr,
                component,
            )
            for csr in csrs
        ]

        result[
            component
        ] = {
            "exact_match_rate":
                exact_match_rate(
                    values
                ),

            "pairwise_similarity":
                pairwise_exact_similarity(
                    values
                ),
        }

    return result


# ============================================================
# ANALYZE ONE IMAGE
# ============================================================

def analyze_image(
    records
):

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

    aligned_csrs = [
        record[
            "aligned_csr"
        ]
        for record in records
    ]

    raw = analyze_stage(
        raw_csrs
    )

    canonical = analyze_stage(
        canonical_csrs
    )

    aligned = analyze_stage(
        aligned_csrs
    )

    return {
        "num_runs": len(records),

        "raw": raw,

        "canonical": canonical,

        "aligned": aligned,

        "improvement": {
            "raw_to_canonical": {},
            "canonical_to_aligned": {},
            "raw_to_aligned": {},
        },
    }


# ============================================================
# CALCULATE IMAGE IMPROVEMENTS
# ============================================================

def add_improvements(
    result
):

    components = [
        "whole_csr",
        "entities",
        "attributes",
        "actions",
        "relations",
        "scene",
    ]

    for component in components:

        raw = result[
            "raw"
        ][
            component
        ]

        canonical = result[
            "canonical"
        ][
            component
        ]

        aligned = result[
            "aligned"
        ][
            component
        ]

        result[
            "improvement"
        ][
            "raw_to_canonical"
        ][
            component
        ] = {

            "exact_match_change":
                canonical[
                    "exact_match_rate"
                ]
                - raw[
                    "exact_match_rate"
                ],

            "pairwise_change":
                canonical[
                    "pairwise_similarity"
                ]
                - raw[
                    "pairwise_similarity"
                ],
        }

        result[
            "improvement"
        ][
            "canonical_to_aligned"
        ][
            component
        ] = {

            "exact_match_change":
                aligned[
                    "exact_match_rate"
                ]
                - canonical[
                    "exact_match_rate"
                ],

            "pairwise_change":
                aligned[
                    "pairwise_similarity"
                ]
                - canonical[
                    "pairwise_similarity"
                ],
        }

        result[
            "improvement"
        ][
            "raw_to_aligned"
        ][
            component
        ] = {

            "exact_match_change":
                aligned[
                    "exact_match_rate"
                ]
                - raw[
                    "exact_match_rate"
                ],

            "pairwise_change":
                aligned[
                    "pairwise_similarity"
                ]
                - raw[
                    "pairwise_similarity"
                ],
        }


# ============================================================
# OVERALL ANALYSIS
# ============================================================

def analyze_overall(
    image_results
):

    components = [
        "whole_csr",
        "entities",
        "attributes",
        "actions",
        "relations",
        "scene",
    ]

    overall = {}

    for component in components:

        raw_exact = []
        canonical_exact = []
        aligned_exact = []

        raw_pairwise = []
        canonical_pairwise = []
        aligned_pairwise = []

        for result in image_results.values():

            raw_exact.append(
                result[
                    "raw"
                ][
                    component
                ][
                    "exact_match_rate"
                ]
            )

            canonical_exact.append(
                result[
                    "canonical"
                ][
                    component
                ][
                    "exact_match_rate"
                ]
            )

            aligned_exact.append(
                result[
                    "aligned"
                ][
                    component
                ][
                    "exact_match_rate"
                ]
            )

            raw_pairwise.append(
                result[
                    "raw"
                ][
                    component
                ][
                    "pairwise_similarity"
                ]
            )

            canonical_pairwise.append(
                result[
                    "canonical"
                ][
                    component
                ][
                    "pairwise_similarity"
                ]
            )

            aligned_pairwise.append(
                result[
                    "aligned"
                ][
                    component
                ][
                    "pairwise_similarity"
                ]
            )

        raw_exact_avg = (
            sum(raw_exact)
            / len(raw_exact)
        )

        canonical_exact_avg = (
            sum(canonical_exact)
            / len(canonical_exact)
        )

        aligned_exact_avg = (
            sum(aligned_exact)
            / len(aligned_exact)
        )

        raw_pairwise_avg = (
            sum(raw_pairwise)
            / len(raw_pairwise)
        )

        canonical_pairwise_avg = (
            sum(canonical_pairwise)
            / len(canonical_pairwise)
        )

        aligned_pairwise_avg = (
            sum(aligned_pairwise)
            / len(aligned_pairwise)
        )

        overall[
            component
        ] = {

            "raw_exact_match_rate":
                raw_exact_avg,

            "canonical_exact_match_rate":
                canonical_exact_avg,

            "aligned_exact_match_rate":
                aligned_exact_avg,

            "raw_to_canonical_exact_change":
                canonical_exact_avg
                - raw_exact_avg,

            "canonical_to_aligned_exact_change":
                aligned_exact_avg
                - canonical_exact_avg,

            "raw_to_aligned_exact_change":
                aligned_exact_avg
                - raw_exact_avg,

            "raw_pairwise_similarity":
                raw_pairwise_avg,

            "canonical_pairwise_similarity":
                canonical_pairwise_avg,

            "aligned_pairwise_similarity":
                aligned_pairwise_avg,

            "raw_to_canonical_pairwise_change":
                canonical_pairwise_avg
                - raw_pairwise_avg,

            "canonical_to_aligned_pairwise_change":
                aligned_pairwise_avg
                - canonical_pairwise_avg,

            "raw_to_aligned_pairwise_change":
                aligned_pairwise_avg
                - raw_pairwise_avg,
        }

    return overall


# ============================================================
# PRINT RESULTS
# ============================================================

def print_stage(
    name,
    data,
):

    print(
        f"\n{name.upper()}"
    )

    print(
        f"  Whole CSR exact match : "
        f"{data['whole_csr']['exact_match_rate'] * 100:.2f}%"
    )

    print(
        f"  Whole CSR pairwise    : "
        f"{data['whole_csr']['pairwise_similarity'] * 100:.2f}%"
    )

    for component in [
        "entities",
        "attributes",
        "actions",
        "relations",
        "scene",
    ]:

        print(
            f"  {component:<12}: "
            f"{data[component]['exact_match_rate'] * 100:.2f}% exact | "
            f"{data[component]['pairwise_similarity'] * 100:.2f}% pairwise"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("THREE-STAGE CSR STABILITY ANALYSIS")
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
    # Group
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

    for image_id in sorted(
        grouped.keys()
    ):

        image_records = grouped[
            image_id
        ]

        print(
            f"\nAnalyzing {image_id} "
            f"({len(image_records)} runs)"
        )

        result = analyze_image(
            image_records
        )

        add_improvements(
            result
        )

        image_results[
            image_id
        ] = result

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall = analyze_overall(
        image_results
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    analysis = {

        "experiment": {

            "input_records":
                len(records),

            "images":
                len(grouped),

            "runs_per_image":
                3,

            "stages": [
                "raw_csr",
                "canonical_csr",
                "aligned_csr",
            ],

            "metrics": [
                "exact_match_rate",
                "pairwise_similarity",
            ],
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
    # Print overall results
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
            f"  Raw       exact : "
            f"{data['raw_exact_match_rate'] * 100:.2f}%"
        )

        print(
            f"  Canonical exact : "
            f"{data['canonical_exact_match_rate'] * 100:.2f}%"
        )

        print(
            f"  Aligned   exact : "
            f"{data['aligned_exact_match_rate'] * 100:.2f}%"
        )

        print(
            f"  Raw → Canonical  : "
            f"{data['raw_to_canonical_exact_change'] * 100:+.2f} pp"
        )

        print(
            f"  Canonical → Aligned : "
            f"{data['canonical_to_aligned_exact_change'] * 100:+.2f} pp"
        )

        print(
            f"  Raw → Aligned    : "
            f"{data['raw_to_aligned_exact_change'] * 100:+.2f} pp"
        )

        print(
            f"  Raw pairwise     : "
            f"{data['raw_pairwise_similarity'] * 100:.2f}%"
        )

        print(
            f"  Canonical pairwise : "
            f"{data['canonical_pairwise_similarity'] * 100:.2f}%"
        )

        print(
            f"  Aligned pairwise : "
            f"{data['aligned_pairwise_similarity'] * 100:.2f}%"
        )

        print(
            f"  Pairwise Raw → Aligned : "
            f"{data['raw_to_aligned_pairwise_change'] * 100:+.2f} pp"
        )

    # --------------------------------------------------------
    # Save complete result
    # --------------------------------------------------------

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