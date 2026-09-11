import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "results/stability_15_results.json"
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
# FORMAT HELPERS
# ============================================================

def print_entities(csr):

    entities = csr.get(
        "entities",
        [],
    )

    if not entities:
        print("    None")
        return

    for entity in entities:

        entity_id = entity.get(
            "id",
            "?",
        )

        entity_type = entity.get(
            "type",
            "?",
        )

        attributes = entity.get(
            "attributes",
            [],
        )

        if attributes:
            attribute_text = ", ".join(
                attributes
            )
            print(
                f"    {entity_id}: "
                f"{entity_type} "
                f"[{attribute_text}]"
            )
        else:
            print(
                f"    {entity_id}: "
                f"{entity_type}"
            )


def print_actions(csr):

    actions = csr.get(
        "actions",
        [],
    )

    if not actions:
        print("    None")
        return

    for action in actions:

        subject = action.get(
            "subject",
            "?",
        )

        verb = action.get(
            "action",
            "?",
        )

        obj = action.get(
            "object"
        )

        if obj is None:
            object_text = "null"
        else:
            object_text = str(obj)

        print(
            f"    {subject} --{verb}--> "
            f"{object_text}"
        )


def print_relations(csr):

    relations = csr.get(
        "relations",
        [],
    )

    if not relations:
        print("    None")
        return

    for relation in relations:

        subject = relation.get(
            "subject",
            "?",
        )

        rel = relation.get(
            "relation",
            "?",
        )

        obj = relation.get(
            "object",
            "?",
        )

        print(
            f"    {subject} --{rel}--> "
            f"{obj}"
        )


def print_scene(csr):

    scene = csr.get(
        "scene",
        {},
    )

    location = scene.get(
        "location"
    )

    environment = scene.get(
        "environment",
        [],
    )

    print(
        f"    Location: "
        f"{location}"
    )

    if environment:
        print(
            "    Environment: "
            + ", ".join(environment)
        )
    else:
        print(
            "    Environment: None"
        )


# ============================================================
# PRINT ONE CSR
# ============================================================

def print_csr(record):

    extraction_run = record.get(
        "extraction_run_number"
    )

    csr = record.get(
        "csr"
    )

    print(
        f"\n  --- Extraction Run "
        f"{extraction_run} ---"
    )

    print(
        "  Entities:"
    )

    print_entities(csr)

    print(
        "\n  Actions:"
    )

    print_actions(csr)

    print(
        "\n  Relations:"
    )

    print_relations(csr)

    print(
        "\n  Scene:"
    )

    print_scene(csr)


# ============================================================
# COMPARE COMPONENTS
# ============================================================

def compare_component(
    records,
    component,
):

    values = []

    for record in records:

        csr = record.get(
            "csr",
            {},
        )

        value = csr.get(
            component
        )

        values.append(
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=False,
            )
        )

    return len(set(values)) == 1


def compare_scene(records):

    values = []

    for record in records:

        scene = record.get(
            "csr",
            {}
        ).get(
            "scene",
            {},
        )

        values.append(
            json.dumps(
                scene,
                sort_keys=True,
                ensure_ascii=False,
            )
        )

    return len(set(values)) == 1


# ============================================================
# ANALYZE ONE IMAGE
# ============================================================

def inspect_image(
    image_id,
    records,
):

    records = sorted(
        records,
        key=lambda r:
            r["extraction_run_number"],
    )

    print("\n")
    print("=" * 80)
    print(
        f"{image_id.upper()} "
        f"| Baseline Run "
        f"{records[0]['baseline_run_number']}"
    )
    print("=" * 80)

    print(
        "\nCaption:"
    )

    print(
        f"  {records[0]['caption']}"
    )

    # --------------------------------------------------------
    # Print all three CSRs
    # --------------------------------------------------------

    for record in records:

        print_csr(record)

    # --------------------------------------------------------
    # Stability summary
    # --------------------------------------------------------

    entities_same = compare_component(
        records,
        "entities",
    )

    actions_same = compare_component(
        records,
        "actions",
    )

    relations_same = compare_component(
        records,
        "relations",
    )

    scene_same = compare_scene(
        records
    )

    print("\n" + "-" * 80)
    print("COMPONENT COMPARISON")
    print("-" * 80)

    print(
        f"  Entities   : "
        f"{'SAME' if entities_same else 'DIFFERENT'}"
    )

    print(
        f"  Actions    : "
        f"{'SAME' if actions_same else 'DIFFERENT'}"
    )

    print(
        f"  Relations  : "
        f"{'SAME' if relations_same else 'DIFFERENT'}"
    )

    print(
        f"  Scene      : "
        f"{'SAME' if scene_same else 'DIFFERENT'}"
    )


# ============================================================
# MAIN
# ============================================================

def inspect():

    print("=" * 80)
    print("CSR VARIATION INSPECTION")
    print("=" * 80)

    records = load_results()

    print(
        f"\nLoaded records: "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # Group records by image
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
    # Check expected dataset
    # --------------------------------------------------------

    expected_images = [
        "test_001",
        "test_002",
        "test_003",
        "test_004",
        "test_005",
    ]

    for image_id in expected_images:

        if image_id not in grouped:

            raise ValueError(
                f"Missing image: "
                f"{image_id}"
            )

        if len(
            grouped[image_id]
        ) != 3:

            raise ValueError(
                f"{image_id} should have "
                f"3 extraction runs, "
                f"found "
                f"{len(grouped[image_id])}"
            )

    # --------------------------------------------------------
    # Inspect each image
    # --------------------------------------------------------

    for image_id in expected_images:

        inspect_image(
            image_id,
            grouped[image_id],
        )

    print("\n")
    print("=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    inspect()