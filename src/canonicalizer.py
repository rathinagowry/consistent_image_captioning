import re
from copy import deepcopy

from nltk.stem import WordNetLemmatizer


# =========================================================
# NLTK
# =========================================================

lemmatizer = WordNetLemmatizer()


# =========================================================
# BASIC TEXT NORMALIZATION
# =========================================================

def normalize_surface(value):
    """
    Basic deterministic surface normalization.

    This function does not perform semantic normalization.

    Examples:
        " Desk Lamp " -> "desk lamp"
        "Snow-Capped" -> "snow capped"
        "  grassy   field " -> "grassy field"
    """

    if value is None:
        return None

    value = str(value).strip().lower()

    # Normalize hyphens to spaces
    value = value.replace("-", " ")

    # Normalize repeated whitespace
    value = re.sub(r"\s+", " ", value)

    return value


# =========================================================
# ACTION LEMMATIZATION
# =========================================================

def lemmatize_action(value):
    """
    Lemmatize actions using NLTK WordNet.

    This is morphological normalization, not
    manually defined semantic mapping.

    Examples:

        runs     -> run
        running  -> run
        walked   -> walk
        walking  -> walk
        sits     -> sit
        sitting  -> sit
        typing   -> type
    """

    value = normalize_surface(value)

    if not value:
        return value

    tokens = value.split()

    normalized_tokens = []

    for token in tokens:
        lemma = lemmatizer.lemmatize(
            token,
            pos="v"
        )

        normalized_tokens.append(lemma)

    return "_".join(normalized_tokens)


# =========================================================
# ENTITY TYPE NORMALIZATION
# =========================================================

def normalize_entity_type(value):
    """
    Normalize entity types lexically.

    IMPORTANT:
    No noun lemmatization is performed.

    This avoids transformations such as:

        jeans    -> jean
        glasses  -> glass
        children -> child

    Instead, entity names are preserved while
    surface formatting is normalized.

    Examples:

        "desk lamp" -> "desk_lamp"
        "park area" -> "park_area"
        "jeans"     -> "jeans"
        "children"  -> "children"
        "mountains" -> "mountains"
    """

    value = normalize_surface(value)

    if not value:
        return value

    # Convert multi-word entity names to one canonical form.
    return value.replace(" ", "_")


# =========================================================
# ATTRIBUTE NORMALIZATION
# =========================================================

def normalize_attribute(value):
    """
    Conservative lexical normalization of attributes.

    No semantic synonym mapping.
    No noun/verb lemmatization.

    Examples:

        "white shirt" -> "white_shirt"
        "snow-capped" -> "snow_capped"
        "in background" -> "in_background"
    """

    value = normalize_surface(value)

    if not value:
        return value

    return value.replace(" ", "_")


# =========================================================
# RELATION NORMALIZATION
# =========================================================

def normalize_relation(value):
    """
    Lexically normalize relations.

    IMPORTANT:
    No semantic relation mapping is performed.

    Therefore:

        "background of" -> "background_of"
        "in background" -> "in_background"
        "on top of"     -> "on_top_of"
        "beside"        -> "beside"
        "inside"        -> "inside"

    We do NOT assume that any of these are equivalent.
    Semantic equivalence is handled later by the
    semantic alignment stage.
    """

    value = normalize_surface(value)

    if not value:
        return value

    return value.replace(" ", "_")


# =========================================================
# ENTITY NORMALIZATION
# =========================================================

def normalize_entity(entity):
    """
    Normalize one entity.
    """

    normalized = {
        "id": entity.get("id"),
        "type": normalize_entity_type(
            entity.get("type")
        ),
        "attributes": [],
    }

    attributes = entity.get(
        "attributes",
        []
    )

    if attributes:
        normalized["attributes"] = sorted(
            set(
                normalize_attribute(attribute)
                for attribute in attributes
                if attribute
            )
        )

    return normalized


# =========================================================
# ACTION NORMALIZATION
# =========================================================

def normalize_action(action):
    """
    Normalize one action.

    IMPORTANT:
    The object is preserved exactly as extracted.

    We do NOT decide whether an action should have
    an object here.
    """

    return {
        "subject": action.get("subject"),
        "action": lemmatize_action(
            action.get("action")
        ),
        "object": action.get("object"),
    }


# =========================================================
# RELATION ENTRY NORMALIZATION
# =========================================================

def normalize_relation_entry(relation):
    """
    Normalize one relation entry.
    """

    return {
        "subject": relation.get("subject"),
        "relation": normalize_relation(
            relation.get("relation")
        ),
        "object": relation.get("object"),
    }


# =========================================================
# SCENE NORMALIZATION
# =========================================================

def normalize_scene(scene):
    """
    Lexically normalize scene information.

    No semantic inference is performed.
    """

    if not scene:
        return {
            "location": None,
            "environment": [],
        }

    location = scene.get("location")

    if location:
        location = normalize_surface(
            location
        )

        location = location.replace(
            " ",
            "_"
        )

    environment = scene.get(
        "environment",
        []
    )

    normalized_environment = []

    for item in environment:

        item = normalize_surface(item)

        if item:
            item = item.replace(
                " ",
                "_"
            )

            normalized_environment.append(
                item
            )

    # Deterministic ordering
    normalized_environment = sorted(
        set(normalized_environment)
    )

    return {
        "location": location,
        "environment": normalized_environment,
    }


# =========================================================
# ENTITY SORTING
# =========================================================

def entity_sort_key(entity):
    """
    Deterministic ordering of entities.

    This is only used to make the representation
    reproducible.

    It does NOT determine semantic importance.
    """

    return (
        entity.get("type", ""),
        tuple(
            entity.get(
                "attributes",
                []
            )
        ),
    )


# =========================================================
# ENTITY ID REMAPPING
# =========================================================

def remap_entity_ids(csr):
    """
    Assign deterministic entity IDs:

        e1
        e2
        e3
        ...

    Entity references in actions and relations
    are updated accordingly.
    """

    entities = csr["entities"]

    sorted_entities = sorted(
        entities,
        key=entity_sort_key
    )

    id_mapping = {}

    for index, entity in enumerate(
        sorted_entities,
        start=1
    ):
        old_id = entity["id"]
        new_id = f"e{index}"

        id_mapping[old_id] = new_id

    # Update entity IDs
    for entity in sorted_entities:

        old_id = entity["id"]

        entity["id"] = id_mapping[
            old_id
        ]

    # Update action references
    for action in csr["actions"]:

        subject = action.get("subject")
        obj = action.get("object")

        if subject in id_mapping:
            action["subject"] = id_mapping[
                subject
            ]

        if obj in id_mapping:
            action["object"] = id_mapping[
                obj
            ]

    # Update relation references
    for relation in csr["relations"]:

        subject = relation.get("subject")
        obj = relation.get("object")

        if subject in id_mapping:
            relation["subject"] = id_mapping[
                subject
            ]

        if obj in id_mapping:
            relation["object"] = id_mapping[
                obj
            ]

    csr["entities"] = sorted(
        sorted_entities,
        key=lambda entity: entity["id"]
    )

    return csr


# =========================================================
# DUPLICATE REMOVAL
# =========================================================

def remove_duplicates(csr):
    """
    Remove exact duplicate entities,
    actions and relations.

    No semantic equivalence is assumed.
    """

    # -----------------------------------------------------
    # Entities
    # -----------------------------------------------------

    unique_entities = []
    seen_entities = set()

    for entity in csr["entities"]:

        key = (
            entity["type"],
            tuple(
                entity["attributes"]
            ),
        )

        if key not in seen_entities:

            seen_entities.add(key)
            unique_entities.append(
                entity
            )

    csr["entities"] = unique_entities

    # -----------------------------------------------------
    # Actions
    # -----------------------------------------------------

    unique_actions = []
    seen_actions = set()

    for action in csr["actions"]:

        key = (
            action["subject"],
            action["action"],
            action["object"],
        )

        if key not in seen_actions:

            seen_actions.add(key)
            unique_actions.append(
                action
            )

    csr["actions"] = unique_actions

    # -----------------------------------------------------
    # Relations
    # -----------------------------------------------------

    unique_relations = []
    seen_relations = set()

    for relation in csr["relations"]:

        key = (
            relation["subject"],
            relation["relation"],
            relation["object"],
        )

        if key not in seen_relations:

            seen_relations.add(key)
            unique_relations.append(
                relation
            )

    csr["relations"] = unique_relations

    return csr


# =========================================================
# DETERMINISTIC SORTING
# =========================================================

def sort_csr(csr):
    """
    Apply deterministic ordering.
    """

    csr["entities"] = sorted(
        csr["entities"],
        key=lambda entity: entity["id"]
    )

    csr["actions"] = sorted(
        csr["actions"],
        key=lambda action: (
            action["subject"],
            action["action"],
            str(action["object"]),
        )
    )

    csr["relations"] = sorted(
        csr["relations"],
        key=lambda relation: (
            relation["subject"],
            relation["relation"],
            str(relation["object"]),
        )
    )

    csr["scene"]["environment"] = sorted(
        set(
            csr["scene"]["environment"]
        )
    )

    return csr


# =========================================================
# MAIN CANONICALIZATION
# =========================================================

def canonicalize_csr(csr, caption=None):
    """
    Convert an extracted CSR into a deterministic
    lexically canonical CSR.

    -------------------------------------------------------
    IMPORTANT DESIGN PRINCIPLE
    -------------------------------------------------------

    This function does NOT:

        - use the image ID
        - use the caption
        - use image-specific rules
        - use manually defined synonym dictionaries
        - perform semantic relation mapping
        - perform semantic entity matching
        - infer missing information
        - use an LLM

    It only performs deterministic lexical
    normalization and morphological normalization
    of actions.
    """

    canonical = deepcopy(csr)

    # -----------------------------------------------------
    # Normalize entities
    # -----------------------------------------------------

    canonical["entities"] = [
        normalize_entity(entity)
        for entity in canonical.get(
            "entities",
            []
        )
    ]

    # -----------------------------------------------------
    # Normalize actions
    # -----------------------------------------------------

    canonical["actions"] = [
        normalize_action(action)
        for action in canonical.get(
            "actions",
            []
        )
    ]

    # -----------------------------------------------------
    # Normalize relations
    # -----------------------------------------------------

    canonical["relations"] = [
        normalize_relation_entry(relation)
        for relation in canonical.get(
            "relations",
            []
        )
    ]

    # -----------------------------------------------------
    # Normalize scene
    # -----------------------------------------------------

    canonical["scene"] = normalize_scene(
        canonical.get("scene")
    )

    # -----------------------------------------------------
    # Remove exact duplicates
    # -----------------------------------------------------

    canonical = remove_duplicates(
        canonical
    )

    # -----------------------------------------------------
    # Assign deterministic IDs
    # -----------------------------------------------------

    canonical = remap_entity_ids(
        canonical
    )

    # -----------------------------------------------------
    # Final deterministic ordering
    # -----------------------------------------------------

    canonical = sort_csr(
        canonical
    )

    return canonical


# =========================================================
# VALIDATION
# =========================================================

def validate_canonical_csr(csr):
    """
    Validate canonical CSR structure.
    """

    required_fields = {
        "entities",
        "actions",
        "relations",
        "scene",
    }

    if not required_fields.issubset(
        csr.keys()
    ):
        return False

    # -----------------------------------------------------
    # Entity IDs
    # -----------------------------------------------------

    entity_ids = {
        entity["id"]
        for entity in csr["entities"]
    }

    if len(entity_ids) != len(
        csr["entities"]
    ):
        return False

    # -----------------------------------------------------
    # Actions
    # -----------------------------------------------------

    for action in csr["actions"]:

        if action["subject"] not in entity_ids:
            return False

        if (
            action["object"] is not None
            and action["object"] not in entity_ids
        ):
            return False

    # -----------------------------------------------------
    # Relations
    # -----------------------------------------------------

    for relation in csr["relations"]:

        if relation["subject"] not in entity_ids:
            return False

        if (
            relation["object"] is not None
            and relation["object"] not in entity_ids
        ):
            return False

    # -----------------------------------------------------
    # Scene
    # -----------------------------------------------------

    if not isinstance(
        csr["scene"].get(
            "environment"
        ),
        list
    ):
        return False

    return True