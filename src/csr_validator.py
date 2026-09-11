import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = RESULTS_DIR / "consensus_csrs.json"

OUTPUT_JSON = (
    RESULTS_DIR /
    "consensus_validation_results.json"
)

OUTPUT_CSV = (
    RESULTS_DIR /
    "consensus_validation_results.csv"
)


# ============================================================
# VALIDATION RESULT
# ============================================================

def issue(
    severity: str,
    category: str,
    message: str,
    location: str = "",
) -> dict:

    return {
        "severity": severity,
        "category": category,
        "message": message,
        "location": location,
    }


# ============================================================
# BASIC TYPES
# ============================================================

def validate_basic_structure(
    csr: dict,
) -> List[dict]:

    issues = []

    required_fields = {
        "image_id",
        "entities",
        "actions",
        "relations",
        "scene",
    }

    missing = (
        required_fields
        - set(csr.keys())
    )

    if missing:

        issues.append(
            issue(
                "ERROR",
                "structure",
                f"Missing fields: {sorted(missing)}",
            )
        )

        return issues

    if not isinstance(
        csr["entities"],
        list,
    ):

        issues.append(
            issue(
                "ERROR",
                "structure",
                "entities must be a list",
            )
        )

    if not isinstance(
        csr["actions"],
        list,
    ):

        issues.append(
            issue(
                "ERROR",
                "structure",
                "actions must be a list",
            )
        )

    if not isinstance(
        csr["relations"],
        list,
    ):

        issues.append(
            issue(
                "ERROR",
                "structure",
                "relations must be a list",
            )
        )

    if not isinstance(
        csr["scene"],
        dict,
    ):

        issues.append(
            issue(
                "ERROR",
                "structure",
                "scene must be an object",
            )
        )

    return issues


# ============================================================
# ENTITY VALIDATION
# ============================================================

def validate_entities(
    csr: dict,
) -> List[dict]:

    issues = []

    entities = csr.get(
        "entities",
        [],
    )

    entity_ids = set()

    for index, entity in enumerate(
        entities
    ):

        location = f"entities[{index}]"

        if not isinstance(
            entity,
            dict,
        ):

            issues.append(
                issue(
                    "ERROR",
                    "entity",
                    "Entity must be an object",
                    location,
                )
            )

            continue

        entity_id = entity.get(
            "id"
        )

        entity_type = entity.get(
            "type"
        )

        attributes = entity.get(
            "attributes"
        )

        # ----------------------------------------------------
        # ID
        # ----------------------------------------------------

        if not isinstance(
            entity_id,
            str,
        ):

            issues.append(
                issue(
                    "ERROR",
                    "entity",
                    "Entity id must be a string",
                    location,
                )
            )

        elif entity_id in entity_ids:

            issues.append(
                issue(
                    "ERROR",
                    "entity",
                    f"Duplicate entity id: {entity_id}",
                    location,
                )
            )

        else:

            entity_ids.add(
                entity_id
            )

        # ----------------------------------------------------
        # TYPE
        # ----------------------------------------------------

        if not isinstance(
            entity_type,
            str,
        ) or not entity_type.strip():

            issues.append(
                issue(
                    "ERROR",
                    "entity",
                    "Entity type must be a non-empty string",
                    location,
                )
            )

        # ----------------------------------------------------
        # ATTRIBUTES
        # ----------------------------------------------------

        if not isinstance(
            attributes,
            list,
        ):

            issues.append(
                issue(
                    "ERROR",
                    "entity",
                    "Entity attributes must be a list",
                    location,
                )
            )

        else:

            seen_attributes = set()

            for attr_index, attribute in enumerate(
                attributes
            ):

                if not isinstance(
                    attribute,
                    str,
                ):

                    issues.append(
                        issue(
                            "ERROR",
                            "attribute",
                            "Attribute must be a string",
                            f"{location}.attributes[{attr_index}]",
                        )
                    )

                    continue

                if not attribute.strip():

                    issues.append(
                        issue(
                            "WARNING",
                            "attribute",
                            "Empty attribute",
                            f"{location}.attributes[{attr_index}]",
                        )
                    )

                if attribute in seen_attributes:

                    issues.append(
                        issue(
                            "WARNING",
                            "attribute",
                            f"Duplicate attribute: {attribute}",
                            f"{location}.attributes[{attr_index}]",
                        )
                    )

                seen_attributes.add(
                    attribute
                )

    return issues


# ============================================================
# ACTION VALIDATION
# ============================================================

def validate_actions(
    csr: dict,
) -> List[dict]:

    issues = []

    entities = {
        entity.get("id")
        for entity in csr.get(
            "entities",
            []
        )
        if isinstance(entity, dict)
    }

    actions = csr.get(
        "actions",
        [],
    )

    seen = set()

    for index, action in enumerate(
        actions
    ):

        location = f"actions[{index}]"

        if not isinstance(
            action,
            dict,
        ):

            issues.append(
                issue(
                    "ERROR",
                    "action",
                    "Action must be an object",
                    location,
                )
            )

            continue

        subject = action.get(
            "subject"
        )

        action_name = action.get(
            "action"
        )

        object_id = action.get(
            "object"
        )

        # ----------------------------------------------------
        # Subject
        # ----------------------------------------------------

        if subject not in entities:

            issues.append(
                issue(
                    "ERROR",
                    "action",
                    f"Unknown action subject: {subject}",
                    location,
                )
            )

        # ----------------------------------------------------
        # Action name
        # ----------------------------------------------------

        if not isinstance(
            action_name,
            str,
        ) or not action_name.strip():

            issues.append(
                issue(
                    "ERROR",
                    "action",
                    "Action name must be a non-empty string",
                    location,
                )
            )

        # ----------------------------------------------------
        # Object
        # ----------------------------------------------------

        if (
            object_id is not None
            and object_id not in entities
        ):

            issues.append(
                issue(
                    "ERROR",
                    "action",
                    f"Unknown action object: {object_id}",
                    location,
                )
            )

        # ----------------------------------------------------
        # Duplicate action
        # ----------------------------------------------------

        signature = (
            subject,
            action_name,
            object_id,
        )

        if signature in seen:

            issues.append(
                issue(
                    "WARNING",
                    "duplicate_fact",
                    f"Duplicate action: {signature}",
                    location,
                )
            )

        seen.add(
            signature
        )

    return issues


# ============================================================
# RELATION VALIDATION
# ============================================================

def validate_relations(
    csr: dict,
) -> List[dict]:

    issues = []

    entities = {
        entity.get("id")
        for entity in csr.get(
            "entities",
            []
        )
        if isinstance(entity, dict)
    }

    relations = csr.get(
        "relations",
        [],
    )

    seen = set()

    for index, relation in enumerate(
        relations
    ):

        location = f"relations[{index}]"

        if not isinstance(
            relation,
            dict,
        ):

            issues.append(
                issue(
                    "ERROR",
                    "relation",
                    "Relation must be an object",
                    location,
                )
            )

            continue

        subject = relation.get(
            "subject"
        )

        relation_name = relation.get(
            "relation"
        )

        object_id = relation.get(
            "object"
        )

        # ----------------------------------------------------
        # Subject
        # ----------------------------------------------------

        if subject not in entities:

            issues.append(
                issue(
                    "ERROR",
                    "relation",
                    f"Unknown relation subject: {subject}",
                    location,
                )
            )

        # ----------------------------------------------------
        # Relation name
        # ----------------------------------------------------

        if not isinstance(
            relation_name,
            str,
        ) or not relation_name.strip():

            issues.append(
                issue(
                    "ERROR",
                    "relation",
                    "Relation name must be a non-empty string",
                    location,
                )
            )

        # ----------------------------------------------------
        # Object
        # ----------------------------------------------------

        if object_id not in entities:

            issues.append(
                issue(
                    "ERROR",
                    "relation",
                    f"Unknown relation object: {object_id}",
                    location,
                )
            )

        # ----------------------------------------------------
        # Duplicate relation
        # ----------------------------------------------------

        signature = (
            subject,
            relation_name,
            object_id,
        )

        if signature in seen:

            issues.append(
                issue(
                    "WARNING",
                    "duplicate_fact",
                    f"Duplicate relation: {signature}",
                    location,
                )
            )

        seen.add(
            signature
        )

    return issues


# ============================================================
# SCENE VALIDATION
# ============================================================

def validate_scene(
    csr: dict,
) -> List[dict]:

    issues = []

    scene = csr.get(
        "scene",
        {},
    )

    if not isinstance(
        scene,
        dict,
    ):
        return issues

    if "location" not in scene:

        issues.append(
            issue(
                "ERROR",
                "scene",
                "Scene is missing location",
                "scene",
            )
        )

    if "environment" not in scene:

        issues.append(
            issue(
                "ERROR",
                "scene",
                "Scene is missing environment",
                "scene",
            )
        )

    environment = scene.get(
        "environment"
    )

    if environment is not None:

        if not isinstance(
            environment,
            list,
        ):

            issues.append(
                issue(
                    "ERROR",
                    "scene",
                    "Environment must be a list",
                    "scene.environment",
                )
            )

        else:

            seen = set()

            for index, value in enumerate(
                environment
            ):

                if not isinstance(
                    value,
                    str,
                ):

                    issues.append(
                        issue(
                            "ERROR",
                            "scene",
                            "Environment value must be a string",
                            f"scene.environment[{index}]",
                        )
                    )

                elif value in seen:

                    issues.append(
                        issue(
                            "WARNING",
                            "duplicate_fact",
                            f"Duplicate environment value: {value}",
                            f"scene.environment[{index}]",
                        )
                    )

                seen.add(
                    value
                )

    return issues


# ============================================================
# POTENTIAL SEMANTIC ISSUES
# ============================================================

def detect_semantic_warnings(
    csr: dict,
) -> List[dict]:
    """
    Detect potentially questionable representations.

    IMPORTANT:
    This function DOES NOT modify the CSR.

    It intentionally does not contain rules such as:
        run -> object must be null
        wear -> relation
        behind -> in_background_of

    Such rules would introduce task-specific semantic assumptions.

    These warnings are therefore only candidates for manual review.
    """

    issues = []

    actions = csr.get(
        "actions",
        [],
    )

    relations = csr.get(
        "relations",
        [],
    )

    # --------------------------------------------------------
    # Same participant pair represented in both action and
    # spatial relation.
    #
    # This is only a warning because some actions genuinely
    # take an entity as an object.
    # --------------------------------------------------------

    action_pairs = set()

    for action in actions:

        action_pairs.add(
            (
                action.get("subject"),
                action.get("object"),
            )
        )

    for index, relation in enumerate(
        relations
    ):

        pair = (
            relation.get("subject"),
            relation.get("object"),
        )

        if pair in action_pairs:

            issues.append(
                issue(
                    "INFO",
                    "semantic_review",
                    (
                        "The same subject/object pair appears "
                        "in both an action and a relation. "
                        "Review whether the representation "
                        "contains redundant semantics."
                    ),
                    f"relations[{index}]",
                )
            )

    # --------------------------------------------------------
    # Self-relations
    # --------------------------------------------------------

    for index, relation in enumerate(
        relations
    ):

        if (
            relation.get("subject")
            == relation.get("object")
        ):

            issues.append(
                issue(
                    "WARNING",
                    "semantic_review",
                    (
                        "Relation connects an entity "
                        "to itself."
                    ),
                    f"relations[{index}]",
                )
            )

    # --------------------------------------------------------
    # Self-action objects
    # --------------------------------------------------------

    for index, action in enumerate(
        actions
    ):

        object_id = action.get(
            "object"
        )

        if (
            object_id is not None
            and action.get("subject")
            == object_id
        ):

            issues.append(
                issue(
                    "WARNING",
                    "semantic_review",
                    (
                        "Action subject and object "
                        "are the same entity."
                    ),
                    f"actions[{index}]",
                )
            )

    return issues


# ============================================================
# COMPLETE CSR VALIDATION
# ============================================================

def validate_csr(
    csr: dict,
) -> List[dict]:

    issues = []

    if not isinstance(
        csr,
        dict,
    ):

        return [
            issue(
                "ERROR",
                "structure",
                "CSR must be a JSON object",
            )
        ]

    issues.extend(
        validate_basic_structure(
            csr
        )
    )

    # If basic structure is invalid, do not assume the
    # remaining fields exist in the expected form.
    if any(
        item["severity"] == "ERROR"
        and item["category"] == "structure"
        for item in issues
    ):
        return issues

    issues.extend(
        validate_entities(
            csr
        )
    )

    issues.extend(
        validate_actions(
            csr
        )
    )

    issues.extend(
        validate_relations(
            csr
        )
    )

    issues.extend(
        validate_scene(
            csr
        )
    )

    issues.extend(
        detect_semantic_warnings(
            csr
        )
    )

    return issues


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CSR VALIDATION")
    print("=" * 70)

    data = load_json(
        INPUT_FILE
    )

    if not isinstance(
        data,
        list,
    ):

        raise ValueError(
            "consensus_csrs.json must contain a JSON list."
        )

    validation_results = []

    total_errors = 0
    total_warnings = 0
    total_info = 0

    # --------------------------------------------------------
    # Validate every CSR
    # --------------------------------------------------------

    for csr in data:

        image_id = csr.get(
            "image_id",
            "unknown",
        )

        issues = validate_csr(
            csr
        )

        errors = [
            item
            for item in issues
            if item["severity"] == "ERROR"
        ]

        warnings = [
            item
            for item in issues
            if item["severity"] == "WARNING"
        ]

        info = [
            item
            for item in issues
            if item["severity"] == "INFO"
        ]

        total_errors += len(errors)
        total_warnings += len(warnings)
        total_info += len(info)

        status = (
            "PASS"
            if not errors
            else "FAIL"
        )

        result = {
            "image_id": image_id,
            "status": status,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "info_count": len(info),
            "issues": issues,
        }

        validation_results.append(
            result
        )

        print(
            f"\n{image_id}: {status}"
        )

        print(
            f"  Errors: {len(errors)}"
        )

        print(
            f"  Warnings: {len(warnings)}"
        )

        print(
            f"  Info: {len(info)}"
        )

        for item in issues:

            print(
                f"    [{item['severity']}] "
                f"{item['message']}"
            )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    save_json(
        OUTPUT_JSON,
        validation_results,
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "status",
                "error_count",
                "warning_count",
                "info_count",
            ],
        )

        writer.writeheader()

        for result in validation_results:

            writer.writerow(
                {
                    "image_id":
                        result["image_id"],

                    "status":
                        result["status"],

                    "error_count":
                        result["error_count"],

                    "warning_count":
                        result["warning_count"],

                    "info_count":
                        result["info_count"],
                }
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    passed = sum(
        result["status"] == "PASS"
        for result in validation_results
    )

    failed = sum(
        result["status"] == "FAIL"
        for result in validation_results
    )

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"CSRs checked: {len(validation_results)}"
    )

    print(
        f"Passed: {passed}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Total errors: {total_errors}"
    )

    print(
        f"Total warnings: {total_warnings}"
    )

    print(
        f"Total informational findings: {total_info}"
    )

    print(
        f"\nJSON: {OUTPUT_JSON}"
    )

    print(
        f"CSV: {OUTPUT_CSV}"
    )


# ============================================================
# FILE UTILITIES
# ============================================================

def load_json(path: Path) -> Any:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def save_json(
    path: Path,
    data: Any,
) -> None:

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()