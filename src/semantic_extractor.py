
import json
import re
import time

from openai import OpenAI

from src.config import (
    OPENROUTER_API_KEY,
    EXTRACTION_MODEL,
    EXTRACTION_TEMPERATURE,
    EXTRACTION_MAX_RETRIES,
)


# ============================================================
# OPENROUTER CLIENT
# ============================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    max_retries=0,
)


# ============================================================
# CSR SCHEMA
# ============================================================

def empty_csr():
    """
    Return an empty CSR structure.
    """

    return {
        "entities": [],
        "actions": [],
        "relations": [],
        "scene": {
            "location": None,
            "environment": [],
        },
    }


# ============================================================
# EXTRACTION PROMPT
# ============================================================

EXTRACTION_SYSTEM_PROMPT = """
You are a semantic information extraction system.

Your task is to convert an image description into
a structured Canonical Semantic Representation (CSR).

Extract ONLY information explicitly stated or clearly
supported by the description.

DO NOT:
- invent objects
- invent attributes
- invent actions
- invent relationships
- add external knowledge
- add explanations
- rewrite the description

Return ONLY valid JSON.

The required schema is:

{
    "entities": [
        {
            "id": "e1",
            "type": "string",
            "attributes": ["string"]
        }
    ],
    "actions": [
        {
            "subject": "entity_id",
            "action": "string",
            "object": "entity_id or null"
        }
    ],
    "relations": [
        {
            "subject": "entity_id",
            "relation": "string",
            "object": "entity_id"
        }
    ],
    "scene": {
        "location": "string or null",
        "environment": ["string"]
    }
}

RULES:

1. Every entity must have a unique ID.
2. Entity IDs must be e1, e2, e3, etc.
3. Actions must reference valid entity IDs.
4. Relations must reference valid entity IDs.
5. Use singular nouns for entity types.
6. Use base verb forms for actions.
7. Preserve explicitly stated attributes.
8. Do not infer attributes that are not stated.
9. Use null when location is not available.
10. Use an empty list when no actions or relations exist.
11. Return JSON only.
12. Do not return Markdown.
13. Do not return explanations.
14. Do not return reasoning.
15. Assign an action object when the description explicitly
    identifies a distinct entity that receives, accompanies,
    or is directly acted upon by the action.

    Examples:
    - "A woman walks a dog." → walk(object=dog)
    - "A teacher guides a child." → guide(object=child)
    - "A child plays with a toy." → play(object=toy)
    - "A man types on a laptop." → type(object=laptop)

16. Do not use a location, environment, or scene entity as the
    object of an action when the action merely occurs in,
    through, near, above, or around that location.

    Examples:
    - "A dog runs through a field."
      → run(object=null)
      → dog through field

    - "A woman walks through a park."
      → walk(object=null)
      → woman through park

17. Determine whether an action is transitive or intransitive
    from the wording of the description, not from the action
    name alone.

    In particular, "walk" may be either transitive or
    intransitive:

    - "A woman walks." → walk(object=null)
    - "A woman walks through a park." → walk(object=null)
    - "A woman walks a dog." → walk(object=dog)
    - "A woman walks a white dog through a park."
      → walk(object=dog)
      → woman through park

18. For genuinely intransitive actions such as run, sit, stand,
    swim, or rise, use "object": null unless the description
    explicitly identifies a distinct entity as the target or
    object of the action.

19. Represent spatial relationships separately in "relations".

    Prepositional phrases describing where an action occurs
    should normally become relations rather than action objects.

    Examples:
    - "runs through a field"
      → run(object=null)
      → through(field)

    - "sits in a classroom"
      → sit(object=null)
      → in(classroom)

    - "walks through a park"
      → walk(object=null)
      → through(park)

20. Do not duplicate the same semantic fact as both an action
    object and a spatial relation.

    However, an action object and a spatial relation should both
    be represented when they describe different facts.

    Example:
    "A woman walks a dog through a park."

    Correct:
    → woman walk dog
    → woman through park

    Incorrect:
    → woman walk park
    → woman through park

21. Do not treat a scene location such as a lake, field, park,
    classroom, or room as the object of an action unless the
    description explicitly states that the action is performed
    directly on that location.

22. Preserve the grammatical semantic roles expressed by the
    description. Do not convert an explicitly mentioned object
    into a null object merely because the verb is commonly
    used intransitively.

23. When a noun phrase clearly functions as the direct object
    of an action, create or reuse the corresponding entity and
    reference its entity ID in the action object field.

24. When a noun phrase is introduced only as the location,
    setting, or spatial context of an action, represent it as a
    relation rather than an action object.

25. When a spatial phrase describes the location or path of
    the main subject's action, attach the spatial relation to
    the entity performing the action.

    Example:

    "A woman walks a dog through a park."

    Correct:
    woman → through → park

    Do not assign:
    dog → through → park

    unless the description explicitly states that the dog
    itself moves through the park independently.

For example, for:

"A dog runs through a field."

Use:

"actions": [
    {
        "subject": "e1",
        "action": "run",
        "object": null
    }
],

"relations": [
    {
        "subject": "e1",
        "relation": "through",
        "object": "e2"
    }
]

Do NOT represent "field" as the object of "run".

For:

"A woman walks a white dog through a grassy park."

Use:

"actions": [
    {
        "subject": "e1",
        "action": "walk",
        "object": "e2"
    }
],

"relations": [
    {
        "subject": "e1",
        "relation": "through",
        "object": "e3"
    }
]

where:

e1 = woman
e2 = white dog
e3 = grassy park_area

The dog is the object of "walk" because the description
explicitly states "walks a dog".

The park is a spatial context because the description states
"through a park".
"""


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json_from_response(text):
    """
    Extract a JSON object from the model response.

    Handles:
    - pure JSON
    - JSON inside ```json ... ```
    - accidental surrounding text
    """

    if not text:
        raise ValueError(
            "Model returned empty response."
        )

    text = text.strip()

    # --------------------------------------------------------
    # Remove markdown code fences
    # --------------------------------------------------------

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^```\s*",
        "",
        text,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    text = text.strip()

    # --------------------------------------------------------
    # Try direct JSON parsing first
    # --------------------------------------------------------

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Try extracting first JSON object
    # --------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "No JSON object found in model response."
        )

    json_text = text[start:end + 1]

    try:
        return json.loads(json_text)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON returned by model: {error}"
        )


# ============================================================
# CSR VALIDATION
# ============================================================

def validate_csr(csr):
    """
    Validate the extracted CSR structure.

    Returns:
        True if valid.

    Raises:
        ValueError if invalid.
    """

    if not isinstance(csr, dict):
        raise ValueError(
            "CSR must be a JSON object."
        )

    required_top_level = {
        "entities",
        "actions",
        "relations",
        "scene",
    }

    missing = (
        required_top_level
        - set(csr.keys())
    )

    if missing:
        raise ValueError(
            f"CSR missing fields: {missing}"
        )

    # --------------------------------------------------------
    # Entities
    # --------------------------------------------------------

    if not isinstance(
        csr["entities"],
        list,
    ):
        raise ValueError(
            "'entities' must be a list."
        )

    entity_ids = set()

    for entity in csr["entities"]:

        if not isinstance(entity, dict):
            raise ValueError(
                "Each entity must be an object."
            )

        if "id" not in entity:
            raise ValueError(
                "Entity missing 'id'."
            )

        if "type" not in entity:
            raise ValueError(
                "Entity missing 'type'."
            )

        if "attributes" not in entity:
            raise ValueError(
                "Entity missing 'attributes'."
            )

        entity_id = entity["id"]

        if entity_id in entity_ids:
            raise ValueError(
                f"Duplicate entity ID: {entity_id}"
            )

        entity_ids.add(entity_id)

        if not isinstance(
            entity["attributes"],
            list,
        ):
            raise ValueError(
                "Entity attributes must be a list."
            )

    # --------------------------------------------------------
    # Actions
    # --------------------------------------------------------

    if not isinstance(
        csr["actions"],
        list,
    ):
        raise ValueError(
            "'actions' must be a list."
        )

    for action in csr["actions"]:

        if not isinstance(action, dict):
            raise ValueError(
                "Each action must be an object."
            )

        required = {
            "subject",
            "action",
            "object",
        }

        if not required.issubset(
            action.keys()
        ):
            raise ValueError(
                "Action missing required fields."
            )

        subject = action["subject"]
        obj = action["object"]

        if subject not in entity_ids:
            raise ValueError(
                f"Unknown action subject: {subject}"
            )

        if (
            obj is not None
            and obj not in entity_ids
        ):
            raise ValueError(
                f"Unknown action object: {obj}"
            )

    # --------------------------------------------------------
    # Relations
    # --------------------------------------------------------

    if not isinstance(
        csr["relations"],
        list,
    ):
        raise ValueError(
            "'relations' must be a list."
        )

    for relation in csr["relations"]:

        if not isinstance(
            relation,
            dict,
        ):
            raise ValueError(
                "Each relation must be an object."
            )

        required = {
            "subject",
            "relation",
            "object",
        }

        if not required.issubset(
            relation.keys()
        ):
            raise ValueError(
                "Relation missing required fields."
            )

        subject = relation["subject"]
        obj = relation["object"]

        if subject not in entity_ids:
            raise ValueError(
                f"Unknown relation subject: {subject}"
            )

        if obj not in entity_ids:
            raise ValueError(
                f"Unknown relation object: {obj}"
            )

    # --------------------------------------------------------
    # Scene
    # --------------------------------------------------------

    if not isinstance(
        csr["scene"],
        dict,
    ):
        raise ValueError(
            "'scene' must be an object."
        )

    if "location" not in csr["scene"]:
        raise ValueError(
            "Scene missing 'location'."
        )

    if "environment" not in csr["scene"]:
        raise ValueError(
            "Scene missing 'environment'."
        )

    if not isinstance(
        csr["scene"]["environment"],
        list,
    ):
        raise ValueError(
            "Scene environment must be a list."
        )

    return True


# ============================================================
# SINGLE EXTRACTION
# ============================================================

def is_daily_rate_limit_error(error):
    """
    Check whether the API error indicates that the
    OpenRouter free-model daily quota has been exhausted.
    """

    error_text = str(error).lower()

    return (
        "free-models-per-day" in error_text
        or "daily limit" in error_text
        or "x-ratelimit-remaining" in error_text
    )


def extract_semantics(
    caption,
    max_retries=EXTRACTION_MAX_RETRIES,
):
    """
    Convert one caption into a CSR.

    The extraction model receives ONLY the caption.
    """

    if not caption or not caption.strip():
        raise ValueError(
            "Caption is empty."
        )

    last_error = None

    for attempt in range(
        1,
        max_retries + 1,
    ):

        try:

            response = client.chat.completions.create(
                model=EXTRACTION_MODEL,

                temperature=EXTRACTION_TEMPERATURE,

                messages=[
                    {
                        "role": "system",
                        "content": (
                            EXTRACTION_SYSTEM_PROMPT
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Description:\n\n"
                            + caption
                        ),
                    },
                ],

                # Ask the provider not to return
                # reasoning content when supported.
                extra_body={
                    "reasoning": {
                        "exclude": True
                    }
                },
            )

            # ------------------------------------------------
            # Check response structure
            # ------------------------------------------------

            if not response.choices:
                raise ValueError(
                    "Model response contains no choices."
                )

            message = response.choices[0].message

            content = message.content

            if not content:
                raise ValueError(
                    "Model returned empty content."
                )

            # ------------------------------------------------
            # Parse JSON
            # ------------------------------------------------

            csr = extract_json_from_response(
                content
            )

            # ------------------------------------------------
            # Validate CSR
            # ------------------------------------------------

            validate_csr(csr)

            return {
                "csr": csr,
                "attempts": attempt,
                "raw_output": content,
            }

        except Exception as error:

            last_error = error

            if is_daily_rate_limit_error(error):
                print(
                    "    Daily OpenRouter limit reached."
                )
                print(
                    "    Stopping extraction experiment."
                )

                raise RuntimeError(
                    "OpenRouter free-model daily limit reached."
                ) from error

            print(
                f"    Extraction attempt "
                f"{attempt}/{max_retries} failed: {error}"
            )

            if attempt < max_retries:

                wait_time = 2 ** attempt

                print(
                    f"    Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

    raise RuntimeError(
        "Semantic extraction failed after "
        f"{max_retries} attempts: "
        f"{last_error}"
    )
