import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "results/stability_15_results.json"
)

OUTPUT_FILE = Path(
    "results/csr_stability_analysis.json"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_results():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Stability dataset not found:\n{INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# NORMALIZATION FOR COMPARISON
# ============================================================

def normalize(value):
    """
    Convert dictionaries/lists into a deterministic
    representation for comparison.

    This does NOT change the original CSR.
    It is only used for measuring exact equality.
    """

    if isinstance(value, dict):

        return {
            key: normalize(value[key])
            for key in sorted(value.keys())
        }

    if isinstance(value, list):

        normalized = [
            normalize(item)
            for item in value
        ]

        return sorted(
            normalized,
            key=lambda x: json.dumps(
                x,
                sort_keys=True,
                ensure_ascii=False,
            ),
        )

    return value


# ============================================================
# EXACT CSR MATCH
# ============================================================

def csr_exact_match(csrs):

    if not csrs:
        return False

    normalized = [
        normalize(csr)
        for csr in csrs
    ]

    return all(
        csr == normalized[0]
        for csr in normalized
    )


# ============================================================
# COMPONENT MATCH
# ============================================================

def component_match(csrs, component):

    values = []

    for csr in csrs:

        value = csr.get(component)

        values.append(
            normalize(value)
        )

    if not values:
        return False

    return all(
        value == values[0]
        for value in values
    )


# ============================================================
# ENTITY STABILITY
# ============================================================

def entity_stability(csrs):

    return component_match(
        csrs,
        "entities",
    )


# ============================================================
# ATTRIBUTE STABILITY
# ============================================================

def attribute_stability(csrs):

    attribute_sets = []

    for csr in csrs:

        entities = csr.get(
            "entities",
            [],
        )

        entity_attributes = {}

        for entity in entities:

            entity_id = entity.get(
                "id"
            )

            attributes = entity.get(
                "attributes",
                [],
            )

            entity_attributes[
                entity_id
            ] = sorted(attributes)

        attribute_sets.append(
            entity_attributes
        )

    if not attribute_sets:
        return False

    return all(
        value == attribute_sets[0]
        for value in attribute_sets
    )


# ============================================================
# ACTION STABILITY
# ============================================================

def action_stability(csrs):

    return component_match(
        csrs,
        "actions",
    )


# ============================================================
# RELATION STABILITY
# ============================================================

def relation_stability(csrs):

    return component_match(
        csrs,
        "relations",
    )


# ============================================================
# SCENE STABILITY
# ============================================================

def scene_stability(csrs):

    return component_match(
        csrs,
        "scene",
    )


# ============================================================
# PARTIAL COMPONENT SIMILARITY
# ============================================================

def component_similarity(
    csrs,
    component,
):

    if len(csrs) < 2:
        return 1.0

    normalized = [
        normalize(
            csr.get(component)
        )
        for csr in csrs
    ]

    comparisons = 0
    matches = 0

    for i in range(
        len(normalized)
    ):

        for j in range(
            i + 1,
            len(normalized),
        ):

            comparisons += 1

            if (
                normalized[i]
                == normalized[j]
            ):
                matches += 1

    if comparisons == 0:
        return 1.0

    return matches / comparisons


# ============================================================
# ANALYZE ONE IMAGE
# ============================================================

def analyze_image(
    image_id,
    records,
):

    records = sorted(
        records,
        key=lambda r:
            r["extraction_run_number"],
    )

    csrs = [
        record["csr"]
        for record in records
    ]

    result = {

        "image_id": image_id,

        "baseline_run_number":
            records[0][
                "baseline_run_number"
            ],

        "caption":
            records[0][
                "caption"
            ],

        "num_extractions":
            len(records),

        "csr_exact_match":
            csr_exact_match(csrs),

        "entity_stability":
            entity_stability(csrs),

        "attribute_stability":
            attribute_stability(csrs),

        "action_stability":
            action_stability(csrs),

        "relation_stability":
            relation_stability(csrs),

        "scene_stability":
            scene_stability(csrs),

        "entity_pairwise_similarity":
            component_similarity(
                csrs,
                "entities",
            ),

        "action_pairwise_similarity":
            component_similarity(
                csrs,
                "actions",
            ),

        "relation_pairwise_similarity":
            component_similarity(
                csrs,
                "relations",
            ),

        "scene_pairwise_similarity":
            component_similarity(
                csrs,
                "scene",
            ),

    }

    return result


# ============================================================
# OVERALL ANALYSIS
# ============================================================

def calculate_overall(
    image_results,
):

    total = len(image_results)

    if total == 0:
        return {}

    def percentage(
        field
    ):

        count = sum(
            result[field]
            for result in image_results
        )

        return (
            count / total
        )

    return {

        "num_images":
            total,

        "num_extractions":
            total * 3,

        "csr_exact_match_rate":
            percentage(
                "csr_exact_match"
            ),

        "entity_stability_rate":
            percentage(
                "entity_stability"
            ),

        "attribute_stability_rate":
            percentage(
                "attribute_stability"
            ),

        "action_stability_rate":
            percentage(
                "action_stability"
            ),

        "relation_stability_rate":
            percentage(
                "relation_stability"
            ),

        "scene_stability_rate":
            percentage(
                "scene_stability"
            ),

        "average_entity_pairwise_similarity":
            sum(
                result[
                    "entity_pairwise_similarity"
                ]
                for result in image_results
            ) / total,

        "average_action_pairwise_similarity":
            sum(
                result[
                    "action_pairwise_similarity"
                ]
                for result in image_results
            ) / total,

        "average_relation_pairwise_similarity":
            sum(
                result[
                    "relation_pairwise_similarity"
                ]
                for result in image_results
            ) / total,

        "average_scene_pairwise_similarity":
            sum(
                result[
                    "scene_pairwise_similarity"
                ]
                for result in image_results
            ) / total,

    }


# ============================================================
# MAIN
# ============================================================

def analyze():

    print("=" * 70)
    print("CSR STABILITY ANALYSIS")
    print("=" * 70)

    records = load_results()

    print(
        f"\nTotal records loaded: "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # Group by image
    # --------------------------------------------------------

    grouped = {}

    for record in records:

        image_id = record[
            "image_id"
        ]

        grouped.setdefault(
            image_id,
            [],
        ).append(record)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    expected_images = {
        "test_001",
        "test_002",
        "test_003",
        "test_004",
        "test_005",
    }

    missing_images = (
        expected_images
        - set(grouped.keys())
    )

    if missing_images:

        raise ValueError(
            "Missing images: "
            f"{missing_images}"
        )

    for image_id in expected_images:

        if len(
            grouped[image_id]
        ) != 3:

            raise ValueError(
                f"{image_id} does not "
                f"have exactly 3 extractions."
            )

    # --------------------------------------------------------
    # Analyze each image
    # --------------------------------------------------------

    image_results = []

    for image_id in sorted(
        expected_images
    ):

        result = analyze_image(
            image_id,
            grouped[image_id],
        )

        image_results.append(
            result
        )

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------

    overall = calculate_overall(
        image_results
    )

    final_result = {

        "experiment": {
            "name":
                "Semantic Extraction Stability",
            "num_images":
                5,
            "extractions_per_caption":
                3,
            "total_extractions":
                15,
            "model":
                records[0].get(
                    "model"
                ),
            "temperature":
                records[0].get(
                    "temperature"
                ),
        },

        "per_image":
            image_results,

        "overall":
            overall,

    }

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-IMAGE RESULTS")
    print("=" * 70)

    for result in image_results:

        print(
            f"\n{result['image_id']}"
        )

        print(
            f"  CSR exact match : "
            f"{result['csr_exact_match']}"
        )

        print(
            f"  Entities        : "
            f"{result['entity_stability']}"
        )

        print(
            f"  Attributes      : "
            f"{result['attribute_stability']}"
        )

        print(
            f"  Actions         : "
            f"{result['action_stability']}"
        )

        print(
            f"  Relations       : "
            f"{result['relation_stability']}"
        )

        print(
            f"  Scene           : "
            f"{result['scene_stability']}"
        )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL STABILITY")
    print("=" * 70)

    print(
        f"\nCSR exact match rate : "
        f"{overall['csr_exact_match_rate']:.2%}"
    )

    print(
        f"Entity stability     : "
        f"{overall['entity_stability_rate']:.2%}"
    )

    print(
        f"Attribute stability  : "
        f"{overall['attribute_stability_rate']:.2%}"
    )

    print(
        f"Action stability     : "
        f"{overall['action_stability_rate']:.2%}"
    )

    print(
        f"Relation stability   : "
        f"{overall['relation_stability_rate']:.2%}"
    )

    print(
        f"Scene stability      : "
        f"{overall['scene_stability_rate']:.2%}"
    )

    print(
        f"\nAverage entity pairwise similarity: "
        f"{overall['average_entity_pairwise_similarity']:.2%}"
    )

    print(
        f"Average action pairwise similarity: "
        f"{overall['average_action_pairwise_similarity']:.2%}"
    )

    print(
        f"Average relation pairwise similarity: "
        f"{overall['average_relation_pairwise_similarity']:.2%}"
    )

    print(
        f"Average scene pairwise similarity: "
        f"{overall['average_scene_pairwise_similarity']:.2%}"
    )

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
    ) as file:

        json.dump(
            final_result,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)

    print(
        f"Analysis saved to:\n"
        f"  {OUTPUT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    analyze()