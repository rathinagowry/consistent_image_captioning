import json
import csv
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

STABILITY_FILE = RESULTS_DIR / "stability_15_results.json"

ALIGNMENT_FILE = RESULTS_DIR / "semantic_alignment_results.json"

OUTPUT_JSON = RESULTS_DIR / "alignment_evaluation_results.json"

OUTPUT_CSV = RESULTS_DIR / "alignment_evaluation_results.csv"


# ============================================================
# CONFIGURATION
# ============================================================

# Equal weighting keeps the metric transparent.
# We will also report every component separately.
COMPONENT_WEIGHTS = {
    "entities": 0.20,
    "attributes": 0.20,
    "actions": 0.20,
    "relations": 0.20,
    "scene": 0.20,
}


# ============================================================
# GENERAL UTILITIES
# ============================================================

def load_json(path: Path) -> Any:

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:

    if denominator == 0:
        return 1.0

    return numerator / denominator


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:

    return max(
        minimum,
        min(maximum, value),
    )


# ============================================================
# CSR VALIDATION
# ============================================================

def validate_csr(csr: Any) -> bool:

    if not isinstance(csr, dict):
        return False

    required = {
        "entities",
        "actions",
        "relations",
        "scene",
    }

    if not required.issubset(csr.keys()):
        return False

    if not isinstance(csr["entities"], list):
        return False

    if not isinstance(csr["actions"], list):
        return False

    if not isinstance(csr["relations"], list):
        return False

    if not isinstance(csr["scene"], dict):
        return False

    entity_ids = set()

    for entity in csr["entities"]:

        if not isinstance(entity, dict):
            return False

        entity_id = entity.get("id")
        entity_type = entity.get("type")
        attributes = entity.get("attributes")

        if not isinstance(entity_id, str):
            return False

        if entity_id in entity_ids:
            return False

        entity_ids.add(entity_id)

        if not isinstance(entity_type, str):
            return False

        if not isinstance(attributes, list):
            return False

    for action in csr["actions"]:

        if not isinstance(action, dict):
            return False

        if action.get("subject") not in entity_ids:
            return False

        if not isinstance(action.get("action"), str):
            return False

        obj = action.get("object")

        if obj is not None and obj not in entity_ids:
            return False

    for relation in csr["relations"]:

        if not isinstance(relation, dict):
            return False

        if relation.get("subject") not in entity_ids:
            return False

        if relation.get("object") not in entity_ids:
            return False

        if not isinstance(
            relation.get("relation"),
            str,
        ):
            return False

    scene = csr["scene"]

    if "location" not in scene:
        return False

    if "environment" not in scene:
        return False

    if not isinstance(scene["environment"], list):
        return False

    return True


# ============================================================
# ENTITY MATCH SCORE
# ============================================================

def entity_match_score(
    csr_a: dict,
    csr_b: dict,
    alignment: dict,
) -> float:
    """
    Entity consistency score.

    Score = matched entities / larger entity count.

    Entity IDs are NOT compared directly.
    The semantic aligner's entity correspondence is used.
    """

    entities_a = csr_a["entities"]
    entities_b = csr_b["entities"]

    matches = alignment.get(
        "entity_matches",
        [],
    )

    denominator = max(
        len(entities_a),
        len(entities_b),
    )

    if denominator == 0:
        return 1.0

    valid_matches = set()

    ids_a = {
        entity["id"]
        for entity in entities_a
    }

    ids_b = {
        entity["id"]
        for entity in entities_b
    }

    for match in matches:

        a = match.get("a")
        b = match.get("b")

        if a in ids_a and b in ids_b:
            valid_matches.add((a, b))

    return safe_divide(
        len(valid_matches),
        denominator,
    )


# ============================================================
# ATTRIBUTE MATCH SCORE
# ============================================================

def attribute_match_score(
    csr_a: dict,
    csr_b: dict,
    alignment: dict,
) -> float:
    """
    Attribute consistency.

    The denominator is the larger number of attributes
    associated with corresponding entities.
    """

    entities_a = {
        entity["id"]: entity
        for entity in csr_a["entities"]
    }

    entities_b = {
        entity["id"]: entity
        for entity in csr_b["entities"]
    }

    entity_matches = alignment.get(
        "entity_matches",
        [],
    )

    attribute_matches = alignment.get(
        "attribute_matches",
        [],
    )

    total_possible = 0
    total_matched = 0

    for entity_pair in entity_matches:

        a_id = entity_pair.get("a")
        b_id = entity_pair.get("b")

        if a_id not in entities_a:
            continue

        if b_id not in entities_b:
            continue

        attrs_a = entities_a[a_id].get(
            "attributes",
            [],
        )

        attrs_b = entities_b[b_id].get(
            "attributes",
            [],
        )

        total_possible += max(
            len(attrs_a),
            len(attrs_b),
        )

    for entity_item in attribute_matches:

        a_id = entity_item.get("entity_a")
        b_id = entity_item.get("entity_b")

        if a_id not in entities_a:
            continue

        if b_id not in entities_b:
            continue

        for match in entity_item.get(
            "matches",
            [],
        ):

            attr_a = match.get("a")
            attr_b = match.get("b")

            if (
                attr_a in entities_a[a_id]["attributes"]
                and
                attr_b in entities_b[b_id]["attributes"]
            ):
                total_matched += 1

    return safe_divide(
        total_matched,
        total_possible,
    )


# ============================================================
# ACTION MATCH SCORE
# ============================================================

def action_match_score(
    csr_a: dict,
    csr_b: dict,
    alignment: dict,
) -> float:
    """
    Action consistency.

    Uses semantic aligner action correspondences.

    Score:
        matched actions / larger action count
    """

    actions_a = csr_a["actions"]
    actions_b = csr_b["actions"]

    matches = alignment.get(
        "action_matches",
        [],
    )

    denominator = max(
        len(actions_a),
        len(actions_b),
    )

    if denominator == 0:
        return 1.0

    valid_matches = set()

    for match in matches:

        index_a = match.get("a_index")
        index_b = match.get("b_index")

        if not isinstance(index_a, int):
            continue

        if not isinstance(index_b, int):
            continue

        if not (
            0 <= index_a < len(actions_a)
            and
            0 <= index_b < len(actions_b)
        ):
            continue

        valid_matches.add(
            (index_a, index_b)
        )

    return safe_divide(
        len(valid_matches),
        denominator,
    )


# ============================================================
# RELATION MATCH SCORE
# ============================================================

def relation_match_score(
    csr_a: dict,
    csr_b: dict,
    alignment: dict,
) -> float:
    """
    Relation consistency.

    Score:
        matched relations / larger relation count
    """

    relations_a = csr_a["relations"]
    relations_b = csr_b["relations"]

    matches = alignment.get(
        "relation_matches",
        [],
    )

    denominator = max(
        len(relations_a),
        len(relations_b),
    )

    if denominator == 0:
        return 1.0

    valid_matches = set()

    for match in matches:

        index_a = match.get("a_index")
        index_b = match.get("b_index")

        if not isinstance(index_a, int):
            continue

        if not isinstance(index_b, int):
            continue

        if not (
            0 <= index_a < len(relations_a)
            and
            0 <= index_b < len(relations_b)
        ):
            continue

        valid_matches.add(
            (index_a, index_b)
        )

    return safe_divide(
        len(valid_matches),
        denominator,
    )


# ============================================================
# SCENE SCORE
# ============================================================

def scene_match_score(
    alignment: dict,
) -> float:
    """
    Scene consistency.

    Uses the semantic aligner's explicit scene equivalence
    judgment and confidence.
    """

    scene_match = alignment.get(
        "scene_match",
        {},
    )

    equivalent = scene_match.get(
        "equivalent",
        False,
    )

    confidence = scene_match.get(
        "confidence",
        0.0,
    )

    if not isinstance(
        equivalent,
        bool,
    ):
        return 0.0

    if not isinstance(
        confidence,
        (int, float),
    ):
        return 0.0

    confidence = clamp(
        float(confidence)
    )

    if equivalent:
        return confidence

    return 0.0


# ============================================================
# OVERALL SCORE
# ============================================================

def overall_consistency(
    component_scores: Dict[str, float],
) -> float:

    score = 0.0

    for component, weight in COMPONENT_WEIGHTS.items():

        score += (
            component_scores[component]
            * weight
        )

    return clamp(score)


# ============================================================
# PAIR EVALUATION
# ============================================================

def evaluate_pair(
    pair_result: dict,
    stability_records: Dict[Tuple, dict],
) -> dict:
    """
    Evaluate one CSR pair.
    """

    image_id = pair_result["image_id"]

    record_a_info = pair_result["record_a"]
    record_b_info = pair_result["record_b"]

    key_a = (
        image_id,
        record_a_info["baseline_run_number"],
        record_a_info["extraction_run_number"],
    )

    key_b = (
        image_id,
        record_b_info["baseline_run_number"],
        record_b_info["extraction_run_number"],
    )

    record_a = stability_records.get(key_a)
    record_b = stability_records.get(key_b)

    if record_a is None:
        raise ValueError(
            f"Could not find CSR record A: {key_a}"
        )

    if record_b is None:
        raise ValueError(
            f"Could not find CSR record B: {key_b}"
        )

    csr_a = record_a["csr"]
    csr_b = record_b["csr"]

    alignment = pair_result["alignment"]

    scores = {
        "entities": entity_match_score(
            csr_a,
            csr_b,
            alignment,
        ),

        "attributes": attribute_match_score(
            csr_a,
            csr_b,
            alignment,
        ),

        "actions": action_match_score(
            csr_a,
            csr_b,
            alignment,
        ),

        "relations": relation_match_score(
            csr_a,
            csr_b,
            alignment,
        ),

        "scene": scene_match_score(
            alignment,
        ),
    }

    overall = overall_consistency(
        scores
    )

    return {
        "pair_id": pair_result["pair_id"],
        "image_id": image_id,

        "record_a": record_a_info,
        "record_b": record_b_info,

        "scores": {
            "entity_consistency":
                round(scores["entities"], 4),

            "attribute_consistency":
                round(scores["attributes"], 4),

            "action_consistency":
                round(scores["actions"], 4),

            "relation_consistency":
                round(scores["relations"], 4),

            "scene_consistency":
                round(scores["scene"], 4),

            "overall_consistency":
                round(overall, 4),
        },

        "alignment_summary":
            pair_result.get(
                "summary",
                {},
            ),
    }


# ============================================================
# GROUPING
# ============================================================

def build_stability_index(
    records: List[dict],
) -> Dict[Tuple, dict]:

    index = {}

    for record in records:

        if record.get("status") != "success":
            continue

        key = (
            record["image_id"],
            record["baseline_run_number"],
            record["extraction_run_number"],
        )

        if validate_csr(record.get("csr")):
            index[key] = record

    return index


# ============================================================
# IMAGE-LEVEL SUMMARY
# ============================================================

def calculate_image_summary(
    pair_scores: List[dict],
) -> dict:

    if not pair_scores:
        return {}

    components = [
        "entity_consistency",
        "attribute_consistency",
        "action_consistency",
        "relation_consistency",
        "scene_consistency",
        "overall_consistency",
    ]

    summary = {}

    for component in components:

        values = [
            item["scores"][component]
            for item in pair_scores
        ]

        summary[component] = round(
            sum(values) / len(values),
            4,
        )

    return summary


# ============================================================
# GLOBAL SUMMARY
# ============================================================

def calculate_global_summary(
    pair_scores: List[dict],
) -> dict:

    if not pair_scores:
        return {}

    components = [
        "entity_consistency",
        "attribute_consistency",
        "action_consistency",
        "relation_consistency",
        "scene_consistency",
        "overall_consistency",
    ]

    summary = {}

    for component in components:

        values = [
            item["scores"][component]
            for item in pair_scores
        ]

        summary[component] = round(
            sum(values) / len(values),
            4,
        )

    return summary


# ============================================================
# CSV OUTPUT
# ============================================================

def save_csv(
    pair_scores: List[dict],
) -> None:

    fieldnames = [
        "pair_id",
        "image_id",
        "entity_consistency",
        "attribute_consistency",
        "action_consistency",
        "relation_consistency",
        "scene_consistency",
        "overall_consistency",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in pair_scores:

            scores = result["scores"]

            writer.writerow({
                "pair_id":
                    result["pair_id"],

                "image_id":
                    result["image_id"],

                "entity_consistency":
                    scores["entity_consistency"],

                "attribute_consistency":
                    scores["attribute_consistency"],

                "action_consistency":
                    scores["action_consistency"],

                "relation_consistency":
                    scores["relation_consistency"],

                "scene_consistency":
                    scores["scene_consistency"],

                "overall_consistency":
                    scores["overall_consistency"],
            })


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SEMANTIC ALIGNMENT EVALUATION")
    print("=" * 70)

    print(
        f"Stability file:\n{STABILITY_FILE}"
    )

    print(
        f"Alignment file:\n{ALIGNMENT_FILE}"
    )

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    stability_records = load_json(
        STABILITY_FILE
    )

    alignment_results = load_json(
        ALIGNMENT_FILE
    )

    if not isinstance(
        stability_records,
        list,
    ):
        raise ValueError(
            "stability_15_results.json must contain a list."
        )

    if not isinstance(
        alignment_results,
        list,
    ):
        raise ValueError(
            "semantic_alignment_results.json must contain a list."
        )

    print(
        f"\nLoaded stability records: "
        f"{len(stability_records)}"
    )

    print(
        f"Loaded alignment results: "
        f"{len(alignment_results)}"
    )

    # --------------------------------------------------------
    # Index stability records
    # --------------------------------------------------------

    stability_index = build_stability_index(
        stability_records
    )

    print(
        f"Valid CSR records: "
        f"{len(stability_index)}"
    )

    # --------------------------------------------------------
    # Evaluate pairs
    # --------------------------------------------------------

    pair_scores = []

    failed = []

    for pair_result in alignment_results:

        if pair_result.get("status") != "success":
            continue

        try:

            evaluated = evaluate_pair(
                pair_result,
                stability_index,
            )

            pair_scores.append(
                evaluated
            )

        except Exception as error:

            failed.append({
                "pair_id":
                    pair_result.get(
                        "pair_id"
                    ),
                "error":
                    str(error),
            })

    # --------------------------------------------------------
    # Image grouping
    # --------------------------------------------------------

    image_groups = defaultdict(list)

    for result in pair_scores:

        image_groups[
            result["image_id"]
        ].append(result)

    image_summary = {}

    for image_id, results in sorted(
        image_groups.items()
    ):

        image_summary[image_id] = (
            calculate_image_summary(results)
        )

    # --------------------------------------------------------
    # Global summary
    # --------------------------------------------------------

    global_summary = calculate_global_summary(
        pair_scores
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    output = {
        "experiment": {
            "input_stability_file":
                str(STABILITY_FILE),

            "input_alignment_file":
                str(ALIGNMENT_FILE),

            "total_stability_records":
                len(stability_records),

            "valid_csr_records":
                len(stability_index),

            "total_alignment_pairs":
                len(alignment_results),

            "evaluated_pairs":
                len(pair_scores),

            "failed_pairs":
                len(failed),
        },

        "metric_definition": {
            "description":
                (
                    "Pairwise semantic consistency is "
                    "computed as matched semantic elements "
                    "divided by the larger corresponding "
                    "element count. Overall consistency is "
                    "the equally weighted mean of entity, "
                    "attribute, action, relation and scene "
                    "consistency."
                ),

            "weights":
                COMPONENT_WEIGHTS,
        },

        "global_summary":
            global_summary,

        "image_summary":
            image_summary,

        "pair_results":
            pair_scores,

        "failures":
            failed,
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    save_csv(
        pair_scores
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GLOBAL CONSISTENCY")
    print("=" * 70)

    for name, value in global_summary.items():

        print(
            f"{name:25s}: "
            f"{value:.4f}"
        )

    print("\n" + "=" * 70)
    print("PER-IMAGE CONSISTENCY")
    print("=" * 70)

    for image_id, summary in sorted(
        image_summary.items()
    ):

        print(
            f"\n{image_id}"
        )

        print(
            f"  Entities:    "
            f"{summary['entity_consistency']:.4f}"
        )

        print(
            f"  Attributes:  "
            f"{summary['attribute_consistency']:.4f}"
        )

        print(
            f"  Actions:     "
            f"{summary['action_consistency']:.4f}"
        )

        print(
            f"  Relations:   "
            f"{summary['relation_consistency']:.4f}"
        )

        print(
            f"  Scene:       "
            f"{summary['scene_consistency']:.4f}"
        )

        print(
            f"  Overall:     "
            f"{summary['overall_consistency']:.4f}"
        )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Evaluated pairs: {len(pair_scores)}"
    )

    print(
        f"Failed pairs:    {len(failed)}"
    )

    print(
        f"\nJSON saved to:\n{OUTPUT_JSON}"
    )

    print(
        f"CSV saved to:\n{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()