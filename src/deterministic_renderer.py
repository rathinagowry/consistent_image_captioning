import json
import re
from pathlib import Path
from collections import defaultdict
import csv

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = RESULTS_DIR / "consensus_csrs.json"

OUTPUT_JSON = RESULTS_DIR / "deterministic_captions.json"
OUTPUT_CSV = RESULTS_DIR / "deterministic_captions.csv"


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text(text):
    """
    Deterministically clean a CSR string for natural-language
    realization.

    No semantic transformation is performed.
    """

    if text is None:
        return ""

    text = str(text).strip()

    text = text.replace("_", " ")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def article_for(noun_phrase):
    """
    Deterministically choose 'a' or 'an'.

    This is purely grammatical.
    """

    phrase = clean_text(noun_phrase)

    if not phrase:
        return "a"

    first_word = phrase.split()[0].lower()

    if first_word[0] in "aeiou":
        return "an"

    return "a"


def entity_phrase(entity):
    """
    Convert one entity into a noun phrase.

    Example:
        {
            type: dog,
            attributes: [tan]
        }

    becomes:

        "a tan dog"
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
        attribute
        for attribute in attributes
        if attribute
    ]

    if attributes:

        noun_phrase = (
            " ".join(attributes)
            + " "
            + entity_type
        )

    else:

        noun_phrase = entity_type

    return (
        article_for(noun_phrase)
        + " "
        + noun_phrase
    )


# ============================================================
# ENTITY INDEX
# ============================================================

def build_entity_index(csr):
    """
    Create deterministic lookup:

        e1 -> entity
        e2 -> entity
        ...
    """

    return {
        entity["id"]: entity
        for entity in csr["entities"]
    }


# ============================================================
# ENTITY REFERENCES
# ============================================================

def entity_reference(
    entity_id,
    entity_index,
):
    """
    Return a noun phrase for an entity.

    The renderer uses the same description every time.
    """

    entity = entity_index.get(
        entity_id
    )

    if entity is None:
        return "the entity"

    return entity_phrase(
        entity
    )


# ============================================================
# ACTION REALIZATION
# ============================================================

def realize_action(
    action,
    entity_index,
):
    """
    Convert an action CSR node into a deterministic clause.

    Examples:

        man -> type -> laptop
        becomes:
            "the man types on the laptop"

        teacher -> sit -> null
        becomes:
            "the teacher sits"

    The renderer does not infer semantic relations.
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

    subject_entity = entity_index.get(
        subject_id
    )

    if subject_entity is None:
        return None

    subject_phrase = entity_phrase(
        subject_entity
    )

    # --------------------------------------------------------
    # Grammatical verb form
    # --------------------------------------------------------

    verb = inflect_third_person(
        action_name
    )

    # --------------------------------------------------------
    # No object
    # --------------------------------------------------------

    if object_id is None:

        return (
            subject_phrase
            + " "
            + verb
        )

    object_phrase = entity_reference(
        object_id,
        entity_index,
    )

    return (
        subject_phrase
        + " "
        + verb
        + " "
        + object_phrase
    )


# ============================================================
# THIRD PERSON VERB INFLECTION
# ============================================================

def inflect_third_person(
    verb,
):
    """
    Deterministic English third-person singular inflection.

    This is grammatical realization only.
    """

    verb = clean_text(
        verb
    )

    if not verb:
        return verb

    lower = verb.lower()

    # Already looks inflected.
    if lower.endswith(
        (
            "s",
            "shes",
            "ches",
            "xes",
            "zes",
        )
    ):
        return verb

    if lower.endswith("y") and len(lower) > 1:

        if lower[-2] not in "aeiou":
            return verb[:-1] + "ies"

    if lower.endswith(
        (
            "sh",
            "ch",
            "x",
            "z",
            "o",
        )
    ):

        return verb + "es"

    return verb + "s"


# ============================================================
# RELATION REALIZATION
# ============================================================

def realize_relation(
    relation,
    entity_index,
):
    """
    Convert a relation CSR node into a deterministic clause.

    Example:

        dog -> on -> leash

    becomes:

        "the dog is on the leash"

    """

    subject_id = relation[
        "subject"
    ]

    relation_name = clean_text(
        relation["relation"]
    )

    object_id = relation[
        "object"
    ]

    subject_entity = entity_index.get(
        subject_id
    )

    object_entity = entity_index.get(
        object_id
    )

    if (
        subject_entity is None
        or object_entity is None
    ):
        return None

    subject_phrase = entity_phrase(
        subject_entity
    )

    object_phrase = entity_phrase(
        object_entity
    )

    # --------------------------------------------------------
    # Generic copular realization
    # --------------------------------------------------------

    return (
        subject_phrase
        + " is "
        + relation_name
        + " "
        + object_phrase
    )


# ============================================================
# SCENE REALIZATION
# ============================================================

def realize_scene(
    scene,
):
    """
    Convert scene information into a deterministic phrase.
    """

    location = scene.get(
        "location"
    )

    environment = [
        clean_text(item)
        for item in scene.get(
            "environment",
            [],
        )
    ]

    environment = [
        item
        for item in environment
        if item
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
        )

    if environment:

        if len(environment) == 1:

            return (
                "The scene contains "
                + article_for(environment[0])
                + " "
                + environment[0]
            )

        joined = join_items(
            environment
        )

        return (
            "The scene contains "
            + joined
        )

    return None


# ============================================================
# LIST JOINING
# ============================================================

def join_items(
    items,
):
    """
    Deterministically join a list of phrases.
    """

    items = [
        clean_text(item)
        for item in items
        if clean_text(item)
    ]

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
# CLAUSE DEDUPLICATION
# ============================================================

def deduplicate_clauses(
    clauses,
):
    """
    Remove exact duplicate clauses while preserving
    deterministic ordering.
    """

    seen = set()
    output = []

    for clause in clauses:

        normalized = clause.strip().lower()

        if normalized in seen:
            continue

        seen.add(
            normalized
        )

        output.append(
            clause
        )

    return output


# ============================================================
# CLAUSE ORDER
# ============================================================

def order_clauses(
    actions,
    relations,
    scene,
):
    """
    Deterministic clause ordering.

    1. Actions
    2. Relations
    3. Scene

    Within each category, lexical ordering is used.
    """

    actions = sorted(
        [
            clause
            for clause in actions
            if clause
        ]
    )

    relations = sorted(
        [
            clause
            for clause in relations
            if clause
        ]
    )

    clauses = (
        actions
        + relations
    )

    if scene:
        clauses.append(
            scene
        )

    return deduplicate_clauses(
        clauses
    )


# ============================================================
# SENTENCE REALIZATION
# ============================================================

def capitalize_sentence(
    text,
):
    if not text:
        return text

    return (
        text[0].upper()
        + text[1:]
    )


def ensure_period(
    text,
):
    text = text.strip()

    if not text:
        return text

    if text[-1] not in ".!?":
        text += "."

    return text


def render_csr(
    csr,
):
    """
    Main deterministic graph-to-text renderer.

    No randomness.
    No LLM.
    No API.
    No image-specific logic.
    """

    entity_index = build_entity_index(
        csr
    )

    # --------------------------------------------------------
    # Actions
    # --------------------------------------------------------

    action_clauses = []

    for action in csr["actions"]:

        clause = realize_action(
            action,
            entity_index,
        )

        if clause:
            action_clauses.append(
                clause
            )

    # --------------------------------------------------------
    # Relations
    # --------------------------------------------------------

    relation_clauses = []

    for relation in csr["relations"]:

        clause = realize_relation(
            relation,
            entity_index,
        )

        if clause:
            relation_clauses.append(
                clause
            )

    # --------------------------------------------------------
    # Scene
    # --------------------------------------------------------

    scene_clause = realize_scene(
        csr["scene"]
    )

    # --------------------------------------------------------
    # Deterministic ordering
    # --------------------------------------------------------

    clauses = order_clauses(
        action_clauses,
        relation_clauses,
        scene_clause,
    )

    if not clauses:

        return "The image contains a scene."

    # --------------------------------------------------------
    # Combine clauses
    # --------------------------------------------------------

    sentence = (
        ". ".join(
            ensure_period(
                clause
            )[:-1]
            if ensure_period(clause).endswith(".")
            else ensure_period(clause)
            for clause in clauses
        )
        + "."
    )

    return capitalize_sentence(
        sentence
    )


# ============================================================
# REPRODUCIBILITY TEST
# ============================================================

def test_reproducibility(
    csr,
    repetitions=10,
):
    """
    Render the same CSR repeatedly.

    A deterministic renderer should produce exactly
    one unique output.
    """

    outputs = [
        render_csr(csr)
        for _ in range(repetitions)
    ]

    unique_outputs = set(
        outputs
    )

    return {
        "repetitions": repetitions,
        "unique_outputs": len(
            unique_outputs
        ),
        "reproducible":
            len(unique_outputs) == 1,
        "outputs":
            outputs,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DETERMINISTIC CSR → TEXT RENDERER")
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
            test_reproducibility(
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
    # SAVE JSON
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
    # SAVE CSV
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

        for result in results:

            writer.writerow(
                result
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    total = len(
        results
    )

    reproducible = sum(
        result["reproducibility"]
        for result in results
    )

    print("\n" + "=" * 70)
    print("RENDERING SUMMARY")
    print("=" * 70)

    print(
        f"CSRs rendered: {total}"
    )

    print(
        f"Fully reproducible: "
        f"{reproducible}/{total}"
    )

    print(
        f"\nJSON: {OUTPUT_JSON}"
    )

    print(
        f"CSV: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()