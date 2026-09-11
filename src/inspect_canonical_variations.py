import json
from pathlib import Path


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "canonical_stability_results.json"
)


# ============================================================
# HELPERS
# ============================================================

def print_entities(csr):

    print("  Entities:")

    for entity in csr.get(
        "entities",
        [],
    ):

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
            print(
                f"    {entity_id}: "
                f"{entity_type} "
                f"[{', '.join(attributes)}]"
            )
        else:
            print(
                f"    {entity_id}: "
                f"{entity_type}"
            )


def print_actions(csr):

    print("  Actions:")

    for action in csr.get(
        "actions",
        [],
    ):

        subject = action.get(
            "subject",
            "?",
        )

        action_name = action.get(
            "action",
            "?",
        )

        obj = action.get(
            "object"
        )

        if obj is None:
            obj = "null"

        print(
            f"    {subject} "
            f"--{action_name}--> "
            f"{obj}"
        )


def print_relations(csr):

    print("  Relations:")

    for relation in csr.get(
        "relations",
        [],
    ):

        subject = relation.get(
            "subject",
            "?",
        )

        relation_name = relation.get(
            "relation",
            "?",
        )

        obj = relation.get(
            "object",
            "?",
        )

        print(
            f"    {subject} "
            f"--{relation_name}--> "
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

    print("  Scene:")

    print(
        f"    location: {location}"
    )

    print(
        "    environment: "
        f"{environment}"
    )


# ============================================================
# ENTITY SIGNATURE
# ============================================================

def entity_signature(entity):

    return (
        entity.get(
            "type",
            "",
        ),
        tuple(
            sorted(
                entity.get(
                    "attributes",
                    [],
                )
            )
        ),
    )


# ============================================================
# COMPARE COMPONENTS
# ============================================================

def compare_component(
    csrs,
    component,
):

    values = []

    for csr in csrs:

        if component == "entities":

            value = sorted(
                entity_signature(
                    entity
                )
                for entity in csr.get(
                    "entities",
                    [],
                )
            )

        elif component == "attributes":

            value = sorted(
                attribute
                for entity in csr.get(
                    "entities",
                    [],
                )
                for attribute in entity.get(
                    "attributes",
                    [],
                )
            )

        else:

            value = csr.get(
                component,
                {},
            )

            value = json.dumps(
                value,
                sort_keys=True,
            )

        values.append(
            value
        )

    return values


# ============================================================
# PRINT DIFFERENCES
# ============================================================

def print_component_comparison(
    csrs,
):

    print(
        "\n  Component comparison:"
    )

    components = [
        "entities",
        "attributes",
        "actions",
        "relations",
        "scene",
    ]

    for component in components:

        values = compare_component(
            csrs,
            component,
        )

        all_same = (
            len(
                {
                    json.dumps(
                        value,
                        sort_keys=True,
                    )
                    for value in values
                }
            )
            == 1
        )

        status = (
            "STABLE"
            if all_same
            else "VARIABLE"
        )

        print(
            f"    {component:<12} "
            f": {status}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CANONICAL CSR VARIATION INSPECTION")
    print("=" * 70)

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
    # Inspect each image
    # --------------------------------------------------------

    for image_id in sorted(
        grouped.keys()
    ):

        image_records = sorted(
            grouped[image_id],
            key=lambda x: x.get(
                "extraction_run_number",
                0,
            ),
        )

        print(
            "\n\n"
            + "=" * 70
        )

        print(
            f"{image_id.upper()} "
            f"({len(image_records)} runs)"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Caption
        # ----------------------------------------------------

        caption = image_records[0].get(
            "caption",
            "",
        )

        print(
            "\nOriginal caption:"
        )

        print(
            f"  {caption}"
        )

        # ----------------------------------------------------
        # Individual runs
        # ----------------------------------------------------

        for record in image_records:

            run_number = record.get(
                "extraction_run_number",
                "?",
            )

            csr = record[
                "canonical_csr"
            ]

            print(
                "\n"
                + "-" * 60
            )

            print(
                f"RUN {run_number}"
            )

            print(
                "-" * 60
            )

            print_entities(
                csr
            )

            print_actions(
                csr
            )

            print_relations(
                csr
            )

            print_scene(
                csr
            )

        # ----------------------------------------------------
        # Stability summary
        # ----------------------------------------------------

        csrs = [
            record[
                "canonical_csr"
            ]
            for record in image_records
        ]

        print_component_comparison(
            csrs
        )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "INSPECTION COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()