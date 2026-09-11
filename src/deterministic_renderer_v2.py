import json
import csv
import re
from pathlib import Path
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = RESULTS_DIR / "consensus_csrs.json"

OUTPUT_JSON = (
    RESULTS_DIR /
    "deterministic_captions_v2.json"
)

OUTPUT_CSV = (
    RESULTS_DIR /
    "deterministic_captions_v2.csv"
)


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    text = str(text).strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)

    return text


def normalize_space(text):
    return re.sub(
        r"\s+",
        " ",
        text.strip(),
    )


def capitalize(text):
    if not text:
        return text

    return text[0].upper() + text[1:]


def add_period(text):
    text = text.strip()

    if not text:
        return text

    if text[-1] not in ".!?":
        text += "."

    return text


# ============================================================
# NOUN / NUMBER HANDLING
# ============================================================

def looks_plural(noun):
    """
    Generic lightweight plurality detection.

    This does not change the CSR.
    It is only used for grammatical realization.
    """

    noun = clean_text(noun).lower()

    if noun in {
        "people",
        "children",
        "men",
        "women",
        "trees",
        "mountains",
        "flowers",
        "jeans",
        "glasses",
    }:
        return True

    return noun.endswith("s")


def article_for(noun_phrase):
    """
    Choose a/an/the-compatible indefinite article.

    Plural nouns do not receive an indefinite article.
    """

    phrase = clean_text(noun_phrase)

    if not phrase:
        return ""

    first = phrase.split()[0].lower()

    if looks_plural(
        phrase.split()[-1]
    ):
        return ""

    if first[0] in "aeiou":
        return "an"

    return "a"


def noun_phrase(
    entity,
):
    """
    Convert CSR entity into a noun phrase.

    Example:
        attributes = ["tan"]
        type = "dog"

    -> "a tan dog"
    """

    entity_type = clean_text(
        entity["type"]
    )

    attributes = [
        clean_text(attribute)
        for attribute in entity.get(
            "attributes",
            [],
        )
    ]

    attributes = [
        value
        for value in attributes
        if value
    ]

    if attributes:
        phrase = (
            " ".join(attributes)
            + " "
            + entity_type
        )
    else:
        phrase = entity_type

    article = article_for(
        phrase
    )

    if article:
        return (
            article
            + " "
            + phrase
        )

    return phrase


def definite_noun_phrase(
    entity,
):
    """
    Produce a definite noun phrase when an entity is
    referenced after its introduction.
    """

    entity_type = clean_text(
        entity["type"]
    )

    attributes = [
        clean_text(attribute)
        for attribute in entity.get(
            "attributes",
            [],
        )
    ]

    attributes = [
        value
        for value in attributes
        if value
    ]

    if attributes:

        phrase = (
            " ".join(attributes)
            + " "
            + entity_type
        )

    else:

        phrase = entity_type

    return "the " + phrase


# ============================================================
# VERB REALIZATION
# ============================================================

IRREGULAR_THIRD_PERSON = {
    "be": "is",
    "have": "has",
    "do": "does",
    "go": "goes",
}


def third_person(verb):
    """
    Deterministic third-person singular realization.
    """

    verb = clean_text(
        verb
    )

    if not verb:
        return verb

    lower = verb.lower()

    if lower in IRREGULAR_THIRD_PERSON:

        result = IRREGULAR_THIRD_PERSON[
            lower
        ]

        return result

    # Already inflected.
    if lower.endswith("ies"):
        return verb

    if lower.endswith("ches"):
        return verb

    if lower.endswith("shes"):
        return verb

    if lower.endswith("xes"):
        return verb

    if lower.endswith("zes"):
        return verb

    if lower.endswith("sses"):
        return verb

    if lower.endswith("s"):
        return verb

    if lower.endswith("y"):

        if len(lower) > 1:

            if lower[-2] not in "aeiou":

                return (
                    verb[:-1]
                    + "ies"
                )

    if lower.endswith(
        (
            "ch",
            "sh",
            "x",
            "z",
            "o",
        )
    ):

        return verb + "es"

    return verb + "s"


# ============================================================
# ENTITY INDEX
# ============================================================

def build_entity_index(csr):

    return {
        entity["id"]: entity
        for entity in csr["entities"]
    }


# ============================================================
# INTRODUCTION ORDER
# ============================================================

def entity_order(
    csr,
):
    """
    Deterministic ordering of entities.

    Entities with actions are introduced first,
    followed by remaining entities.

    Ties are resolved by entity ID.
    """

    action_subjects = []

    for action in csr["actions"]:

        subject = action.get(
            "subject"
        )

        if (
            subject
            and subject not in action_subjects
        ):
            action_subjects.append(
                subject
            )

    action_objects = []

    for action in csr["actions"]:

        object_id = action.get(
            "object"
        )

        if (
            object_id
            and object_id not in action_objects
        ):
            action_objects.append(
                object_id
            )

    remaining = sorted(
        entity["id"]
        for entity in csr["entities"]
        if entity["id"]
        not in action_subjects
        and entity["id"]
        not in action_objects
    )

    result = []

    for entity_id in (
        action_subjects
        + action_objects
        + remaining
    ):

        if entity_id not in result:
            result.append(
                entity_id
            )

    return result


# ============================================================
# ACTION REALIZATION
# ============================================================

def realize_action(
    action,
    entity_index,
):
    """
    Generic action realization.

    The renderer does not alter the semantic structure
    of the action.
    """

    subject_id = action[
        "subject"
    ]

    action_name = clean_text(
        action["action"]
    )

    object_id = action.get(
        "object"
    )

    subject = entity_index.get(
        subject_id
    )

    if subject is None:
        return None

    subject_text = noun_phrase(
        subject
    )

    verb = third_person(
        action_name
    )

    # --------------------------------------------------------
    # Intransitive action
    # --------------------------------------------------------

    if object_id is None:

        return (
            subject_text
            + " "
            + verb
        )

    # --------------------------------------------------------
    # Transitive action
    # --------------------------------------------------------

    object_entity = entity_index.get(
        object_id
    )

    if object_entity is None:
        return None

    object_text = noun_phrase(
        object_entity
    )

    return (
        subject_text
        + " "
        + verb
        + " "
        + object_text
    )


# ============================================================
# RELATION REALIZATION
# ============================================================

def realize_relation(
    relation,
    entity_index,
):
    """
    Generic relation realization.

    Examples:

        dog + on + leash
        ->
        "A white dog is on a red leash."

        teacher + with + children
        ->
        "A teacher is with young children."
    """

    subject_id = relation[
        "subject"
    ]

    object_id = relation[
        "object"
    ]

    relation_name = clean_text(
        relation["relation"]
    )

    subject = entity_index.get(
        subject_id
    )

    object_entity = entity_index.get(
        object_id
    )

    if (
        subject is None
        or object_entity is None
    ):
        return None

    subject_text = noun_phrase(
        subject
    )

    object_text = noun_phrase(
        object_entity
    )

    return (
        subject_text
        + " is "
        + relation_name
        + " "
        + object_text
    )


# ============================================================
# SCENE REALIZATION
# ============================================================

def realize_scene(
    scene,
):
    """
    Deterministically realize scene information.
    """

    location = scene.get(
        "location"
    )

    environment = [
        clean_text(value)
        for value in scene.get(
            "environment",
            [],
        )
    ]

    environment = [
        value
        for value in environment
        if value
    ]

    if location:

        location = clean_text(
            location
        )

        return (
            "The scene is set in "
            + article_for(location)
            + " "
            + location
        ).strip()

    if environment:

        if len(environment) == 1:

            item = environment[0]

            article = article_for(
                item
            )

            if article:
                item = (
                    article
                    + " "
                    + item
                )

            return (
                "The scene contains "
                + item
            )

        items = []

        for item in environment:

            article = article_for(
                item
            )

            if article:
                item = (
                    article
                    + " "
                    + item
                )

            items.append(
                item
            )

        joined = join_list(
            items
        )

        return (
            "The scene contains "
            + joined
        )

    return None


# ============================================================
# RELATION GROUPING
# ============================================================

def relation_clauses(
    csr,
    entity_index,
):
    """
    Produce relation clauses in deterministic order.
    """

    clauses = []

    relations = sorted(
        csr["relations"],
        key=lambda relation: (
            relation.get("subject", ""),
            relation.get("relation", ""),
            relation.get("object", ""),
        ),
    )

    for relation in relations:

        clause = realize_relation(
            relation,
            entity_index,
        )

        if clause:
            clauses.append(
                clause
            )

    return clauses


# ============================================================
# ACTION GROUPING
# ============================================================

def action_clauses(
    csr,
    entity_index,
):
    """
    Produce action clauses in deterministic order.
    """

    clauses = []

    actions = sorted(
        csr["actions"],
        key=lambda action: (
            action.get("subject", ""),
            action.get("action", ""),
            str(action.get("object")),
        ),
    )

    for action in actions:

        clause = realize_action(
            action,
            entity_index,
        )

        if clause:
            clauses.append(
                clause
            )

    return clauses


# ============================================================
# LIST JOIN
# ============================================================

def join_list(
    items,
):
    if not items:
        return ""

    if len(items) == 1:
        return items[0]

    if len(items) == 2:

        return (
            items[0]
            + " and "
            + items[1]
        )

    return (
        ", ".join(items[:-1])
        + ", and "
        + items[-1]
    )


# ============================================================
# EXACT CLAUSE DEDUPLICATION
# ============================================================

def deduplicate(
    clauses,
):
    seen = set()
    output = []

    for clause in clauses:

        key = normalize_space(
            clause
        ).lower()

        if key in seen:
            continue

        seen.add(
            key
        )

        output.append(
            clause
        )

    return output


# ============================================================
# FINAL CAPTION
# ============================================================

def render_csr(
    csr,
):
    """
    Deterministic CSR-to-text rendering.

    No LLM.
    No randomness.
    No API.
    No image-specific logic.
    """

    entity_index = build_entity_index(
        csr
    )

    actions = action_clauses(
        csr,
        entity_index,
    )

    relations = relation_clauses(
        csr,
        entity_index,
    )

    scene = realize_scene(
        csr["scene"]
    )

    clauses = (
        actions
        + relations
    )

    if scene:
        clauses.append(
            scene
        )

    clauses = deduplicate(
        clauses
    )

    if not clauses:
        return "The image contains a scene."

    sentence_parts = []

    for clause in clauses:

        clause = add_period(
            clause
        )

        if clause.endswith("."):
            clause = clause[:-1]

        sentence_parts.append(
            clause
        )

    caption = (
        ". ".join(
            sentence_parts
        )
        + "."
    )

    return capitalize(
        caption
    )


# ============================================================
# REPRODUCIBILITY
# ============================================================

def reproducibility_test(
    csr,
    repetitions=10,
):
    outputs = [
        render_csr(csr)
        for _ in range(
            repetitions
        )
    ]

    unique_outputs = set(
        outputs
    )

    return {
        "repetitions":
            repetitions,

        "unique_outputs":
            len(unique_outputs),

        "reproducible":
            len(unique_outputs) == 1,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DETERMINISTIC RENDERER V2")
    print("=" * 70)

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        csrs = json.load(
            file
        )

    results = []

    for csr in csrs:

        image_id = csr[
            "image_id"
        ]

        caption = render_csr(
            csr
        )

        reproducibility = (
            reproducibility_test(
                csr,
                repetitions=10,
            )
        )

        result = {
            "image_id":
                image_id,

            "caption":
                caption,

            "reproducibility":
                reproducibility[
                    "reproducible"
                ],

            "unique_outputs":
                reproducibility[
                    "unique_outputs"
                ],
        }

        results.append(
            result
        )

        print("\n" + "-" * 70)
        print(image_id)
        print("-" * 70)

        print(
            f"Caption:\n{caption}"
        )

        print(
            "Reproducible:",
            reproducibility[
                "reproducible"
            ],
        )

        print(
            "Unique outputs:",
            reproducibility[
                "unique_outputs"
            ],
        )

    # ========================================================
    # JSON
    # ========================================================

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # CSV
    # ========================================================

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
                "caption",
                "reproducibility",
                "unique_outputs",
            ],
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    reproducible_count = sum(
        result["reproducibility"]
        for result in results
    )

    print("\n" + "=" * 70)
    print("RENDERING SUMMARY")
    print("=" * 70)

    print(
        f"CSRs rendered: "
        f"{len(results)}"
    )

    print(
        f"Reproducible: "
        f"{reproducible_count}/{len(results)}"
    )

    print(
        f"\nJSON:\n{OUTPUT_JSON}"
    )

    print(
        f"\nCSV:\n{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()