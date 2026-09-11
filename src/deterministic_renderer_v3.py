import json
import csv
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = RESULTS_DIR / "consensus_csrs.json"

OUTPUT_JSON = RESULTS_DIR / "deterministic_captions_v3.json"
OUTPUT_CSV = RESULTS_DIR / "deterministic_captions_v3.csv"


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


def normalize(text):
    return re.sub(r"\s+", " ", text.strip())


def capitalize_sentence(text):
    text = text.strip()

    if not text:
        return text

    return text[0].upper() + text[1:]


def finish_sentence(text):
    text = normalize(text)

    if not text:
        return ""

    if text[-1] not in ".!?":
        text += "."

    return text


# ============================================================
# PLURALITY
# ============================================================

IRREGULAR_PLURALS = {
    "children",
    "people",
    "men",
    "women",
    "mice",
    "geese",
    "feet",
    "teeth",
    "jeans",
    "glasses",
}


def is_plural(word):
    """
    Generic deterministic plurality detection.

    Used only for grammar.
    It never changes the CSR.
    """

    word = clean_text(word).lower()

    if not word:
        return False

    if word in IRREGULAR_PLURALS:
        return True

    # Basic plural forms.
    if word.endswith("ies"):
        return True

    if word.endswith("ves"):
        return True

    if word.endswith("ches"):
        return True

    if word.endswith("shes"):
        return True

    if word.endswith("xes"):
        return True

    if word.endswith("zes"):
        return True

    if word.endswith("sses"):
        return True

    # Avoid treating common singular words ending in s
    # as plural.
    singular_s_exceptions = {
        "grass",
        "glass",
        "class",
        "business",
        "analysis",
        "news",
    }

    if word in singular_s_exceptions:
        return False

    return word.endswith("s")


def entity_is_plural(entity):
    entity_type = clean_text(
        entity["type"]
    )

    last_word = entity_type.split()[-1]

    return is_plural(last_word)


# ============================================================
# ARTICLES
# ============================================================

def starts_with_vowel_sound(phrase):
    first = phrase.strip().lower()

    if not first:
        return False

    return first[0] in "aeiou"


def indefinite_article(phrase):
    if starts_with_vowel_sound(phrase):
        return "an"

    return "a"


# ============================================================
# ATTRIBUTE REALIZATION
# ============================================================

def looks_like_adjective(attribute):
    """
    Conservative generic adjective heuristic.

    It is intentionally lexical/grammatical rather than
    image-specific.
    """

    attribute = clean_text(attribute).lower()

    if not attribute:
        return False

    # Common adjective morphology.
    adjective_endings = (
        "ful",
        "ous",
        "ive",
        "al",
        "ic",
        "y",
        "less",
        "ish",
        "ed",
        "en",
    )

    if attribute.endswith(adjective_endings):
        return True

    adjective_words = {
        "tan",
        "brown",
        "blue",
        "red",
        "white",
        "yellow",
        "young",
        "calm",
        "rustic",
        "warm",
        "soft",
        "golden",
        "grassy",
        "majestic",
        "mirrored",
        "wooden",
        "educational",
        "snow-capped",
    }

    return attribute in adjective_words


def format_attribute(attribute):
    attribute = clean_text(attribute)

    if "-" in attribute:
        return attribute

    return attribute


def build_entity_phrase(
    entity,
    definite=False,
):
    """
    Convert a CSR entity into a deterministic noun phrase.

    Example:

        dog + tan
        -> a tan dog

        children + young
        -> young children

        man + beard, glasses, white shirt
        -> a man with a beard, glasses, and a white shirt
    """

    entity_type = clean_text(
        entity["type"]
    )

    attributes = [
        format_attribute(attribute)
        for attribute in entity.get(
            "attributes",
            [],
        )
    ]

    adjective_attributes = []
    noun_attributes = []

    for attribute in attributes:

        if looks_like_adjective(attribute):
            adjective_attributes.append(
                attribute
            )
        else:
            noun_attributes.append(
                attribute
            )

    # Deterministic attribute order.
    adjective_attributes.sort(
        key=str.lower
    )

    noun_attributes.sort(
        key=str.lower
    )

    # Adjectives come before the noun.
    base_phrase = " ".join(
        adjective_attributes
        + [entity_type]
    )

    # Noun-like attributes are realized as
    # generic "with" complements.
    if noun_attributes:

        complements = []

        for attribute in noun_attributes:

            if is_plural(
                attribute.split()[-1]
            ):
                complements.append(
                    attribute
                )
            else:
                complements.append(
                    indefinite_article(
                        attribute
                    )
                    + " "
                    + attribute
                )

        base_phrase += (
            " with "
            + join_list(
                complements
            )
        )

    # Definite reference.
    if definite:

        return "the " + base_phrase

    # Plural nouns do not use a/an.
    if entity_is_plural(entity):

        return base_phrase

    return (
        indefinite_article(base_phrase)
        + " "
        + base_phrase
    )


# ============================================================
# VERB INFLECTION
# ============================================================

IRREGULAR_THIRD_PERSON = {
    "be": "is",
    "have": "has",
    "do": "does",
    "go": "goes",
}


def third_person(verb):
    verb = clean_text(verb)

    if not verb:
        return verb

    lower = verb.lower()

    if lower in IRREGULAR_THIRD_PERSON:
        return IRREGULAR_THIRD_PERSON[
            lower
        ]

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

    if lower.endswith("y") and len(lower) > 1:

        if lower[-2] not in "aeiou":

            return (
                verb[:-1]
                + "ies"
            )

    if lower.endswith(
        ("ch", "sh", "x", "z", "o")
    ):
        return verb + "es"

    return verb + "s"


def realize_verb(
    verb,
    subject_entity,
):
    """
    Select base or third-person form according
    to the grammatical number of the subject.
    """

    verb = clean_text(verb)

    if entity_is_plural(
        subject_entity
    ):
        return verb

    return third_person(
        verb
    )


# ============================================================
# ENTITY REFERENCES
# ============================================================

def entity_reference(
    entity,
    introduced,
):
    """
    First mention:
        a white dog

    Later mention:
        the white dog
    """

    return build_entity_phrase(
        entity,
        definite=introduced,
    )


# ============================================================
# LIST JOINING
# ============================================================

def join_list(items):
    items = [
        item
        for item in items
        if item
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
# ACTION REALIZATION
# ============================================================

def realize_action(
    action,
    entity_index,
    introduced,
):
    subject_id = action.get(
        "subject"
    )

    object_id = action.get(
        "object"
    )

    subject_entity = entity_index.get(
        subject_id
    )

    if subject_entity is None:
        return None

    subject_text = entity_reference(
        subject_entity,
        subject_id in introduced,
    )

    verb = realize_verb(
        action.get("action", ""),
        subject_entity,
    )

    # Mark subject as introduced.
    introduced.add(
        subject_id
    )

    if object_id is None:

        return {
            "subject_id": subject_id,
            "subject_text": subject_text,
            "verb": verb,
            "object_text": None,
        }

    object_entity = entity_index.get(
        object_id
    )

    if object_entity is None:
        return None

    object_text = entity_reference(
        object_entity,
        object_id in introduced,
    )

    introduced.add(
        object_id
    )

    return {
        "subject_id": subject_id,
        "subject_text": subject_text,
        "verb": verb,
        "object_text": object_text,
    }


# ============================================================
# ACTION SENTENCE GROUPING
# ============================================================

def render_action_group(
    actions,
    entity_index,
    introduced,
):
    if not actions:
        return None

    realized = []

    for action in actions:

        result = realize_action(
            action,
            entity_index,
            introduced,
        )

        if result:
            realized.append(
                result
            )

    if not realized:
        return None

    # All actions are grouped by subject.
    subject_id = realized[0][
        "subject_id"
    ]

    subject_text = realized[0][
        "subject_text"
    ]

    clauses = []

    for item in realized:

        clause = item["verb"]

        if item["object_text"]:

            clause += (
                " "
                + item["object_text"]
            )

        clauses.append(
            clause
        )

    if len(clauses) == 1:

        sentence = (
            subject_text
            + " "
            + clauses[0]
        )

    else:

        sentence = (
            subject_text
            + " "
            + join_list(clauses)
        )

    return capitalize_sentence(
        finish_sentence(sentence)
    )


# ============================================================
# RELATION REALIZATION
# ============================================================

def relation_surface(
    relation,
):
    relation = clean_text(
        relation
    )

    # Generic grammatical expansion.
    if relation == "in background of":
        return "in the background of"

    return relation


def realize_relation(
    relation,
    entity_index,
    introduced,
):
    subject_id = relation.get(
        "subject"
    )

    object_id = relation.get(
        "object"
    )

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

    subject_text = entity_reference(
        subject_entity,
        subject_id in introduced,
    )

    introduced.add(
        subject_id
    )

    object_text = entity_reference(
        object_entity,
        object_id in introduced,
    )

    introduced.add(
        object_id
    )

    relation_text = relation_surface(
        relation.get(
            "relation",
            "",
        )
    )

    return {
        "subject_id": subject_id,
        "subject_text": subject_text,
        "relation": relation_text,
        "object_text": object_text,
    }


# ============================================================
# RELATION SENTENCE
# ============================================================

def render_relation_group(
    relations,
    entity_index,
    introduced,
):
    if not relations:
        return None

    realized = []

    for relation in relations:

        result = realize_relation(
            relation,
            entity_index,
            introduced,
        )

        if result:
            realized.append(
                result
            )

    if not realized:
        return None

    # Group relations by subject.
    grouped = {}

    for item in realized:

        grouped.setdefault(
            item["subject_id"],
            [],
        ).append(item)

    sentences = []

    for subject_id in sorted(
        grouped
    ):

        items = grouped[
            subject_id
        ]

        subject_text = items[0][
            "subject_text"
        ]

        complements = []

        for item in items:

            complements.append(
                item["relation"]
                + " "
                + item["object_text"]
            )

        sentence = (
            subject_text
            + " is "
            + join_list(
                complements
            )
        )

        sentences.append(
            capitalize_sentence(
                finish_sentence(
                    sentence
                )
            )
        )

    return sentences


# ============================================================
# SCENE REALIZATION
# ============================================================

def scene_sentence(
    scene,
):
    if not scene:
        return None

    location = clean_text(
        scene.get("location")
    )

    environment = [
        clean_text(value)
        for value in scene.get(
            "environment",
            [],
        )
        if clean_text(value)
    ]

    if location:

        article = (
            ""
            if is_plural(
                location.split()[-1]
            )
            else indefinite_article(
                location
            )
            + " "
        )

        return (
            "The scene is set in "
            + article
            + location
            + "."
        )

    if not environment:
        return None

    # Separate adjective-like environment descriptions
    # from noun-like descriptions.
    adjectives = []
    nouns = []

    for item in environment:

        if looks_like_adjective(item):
            adjectives.append(item)
        else:
            nouns.append(item)

    parts = []

    if adjectives:

        parts.append(
            "the environment is "
            + join_list(
                adjectives
            )
        )

    if nouns:

        noun_phrases = []

        for item in nouns:

            if is_plural(
                item.split()[-1]
            ):
                noun_phrases.append(
                    item
                )
            else:
                noun_phrases.append(
                    indefinite_article(item)
                    + " "
                    + item
                )

        parts.append(
            "the environment includes "
            + join_list(
                noun_phrases
            )
        )

    if not parts:
        return None

    if len(parts) == 1:
        return (
            capitalize_sentence(
                parts[0]
            )
            + "."
        )

    return (
        capitalize_sentence(
            parts[0]
        )
        + " and "
        + parts[1]
        + "."
    )


# ============================================================
# GROUP ACTIONS BY SUBJECT
# ============================================================

def group_actions_by_subject(
    actions,
):
    grouped = {}

    for action in actions:

        subject = action.get(
            "subject"
        )

        grouped.setdefault(
            subject,
            [],
        ).append(action)

    return grouped


# ============================================================
# MAIN CSR RENDERER
# ============================================================

def render_csr(
    csr,
):
    entity_index = {
        entity["id"]: entity
        for entity in csr["entities"]
    }

    sentences = []

    introduced = set()

    # --------------------------------------------------------
    # Actions
    # --------------------------------------------------------

    action_groups = (
        group_actions_by_subject(
            csr.get(
                "actions",
                [],
            )
        )
    )

    # Most active subjects first.
    action_subject_order = sorted(
        action_groups,
        key=lambda subject: (
            -len(
                action_groups[
                    subject
                ]
            ),
            subject,
        ),
    )

    for subject_id in action_subject_order:

        sentence = render_action_group(
            action_groups[
                subject_id
            ],
            entity_index,
            introduced,
        )

        if sentence:
            sentences.append(
                sentence
            )

    # --------------------------------------------------------
    # Relations
    # --------------------------------------------------------

    relation_sentences = (
        render_relation_group(
            csr.get(
                "relations",
                [],
            ),
            entity_index,
            introduced,
        )
    )

    if relation_sentences:
        sentences.extend(
            relation_sentences
        )

    # --------------------------------------------------------
    # Scene
    # --------------------------------------------------------

    scene = scene_sentence(
        csr.get(
            "scene",
            {},
        )
    )

    if scene:
        sentences.append(
            scene
        )

    # --------------------------------------------------------
    # Remove exact duplicate sentences.
    # --------------------------------------------------------

    final_sentences = []

    seen = set()

    for sentence in sentences:

        key = normalize(
            sentence
        ).lower()

        if key in seen:
            continue

        seen.add(key)

        final_sentences.append(
            sentence
        )

    if not final_sentences:

        return (
            "The image contains a scene."
        )

    return " ".join(
        final_sentences
    )


# ============================================================
# REPRODUCIBILITY TEST
# ============================================================

def test_reproducibility(
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
        "repetitions": repetitions,
        "unique_outputs": len(
            unique_outputs
        ),
        "reproducible":
            len(unique_outputs) == 1,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "DETERMINISTIC RENDERER V3"
    )
    print("=" * 70)

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        csrs = json.load(file)

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
            "Caption:"
        )

        print(
            caption
        )

        print(
            "\nReproducible:",
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

        writer.writerows(
            results
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    reproducible_count = sum(
        result[
            "reproducibility"
        ]
        for result in results
    )

    print("\n" + "=" * 70)
    print(
        "RENDERING SUMMARY"
    )
    print("=" * 70)

    print(
        "CSRs rendered:",
        len(results),
    )

    print(
        "Reproducible:",
        f"{reproducible_count}/{len(results)}",
    )

    print(
        "\nJSON:"
    )

    print(
        OUTPUT_JSON
    )

    print(
        "\nCSV:"
    )

    print(
        OUTPUT_CSV
    )


if __name__ == "__main__":
    main()