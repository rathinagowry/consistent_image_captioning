
import os
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# ENVIRONMENT
# ============================================================

# Project root:
# C:\Projects\Image_Consistency
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY was not found.\n"
        f"Expected .env at: {PROJECT_ROOT / '.env'}"
    )


# ============================================================
# CONFIGURATION
# ============================================================

ALIGNMENT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

ALIGNMENT_TEMPERATURE = 0.0

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = RESULTS_DIR / "stability_15_results.json"

OUTPUT_FILE = RESULTS_DIR / "semantic_alignment_results.json"


# ============================================================
# OPENROUTER CLIENT
# ============================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    max_retries=0,
)


# ============================================================
# CSR STRUCTURAL VALIDATION
# ============================================================

def validate_csr_structure(csr: Any) -> bool:
    """
    Validate the minimum structure expected from a canonical CSR.

    This validator is intentionally local to the semantic alignment
    stage because the inner CSR does not contain image_id.

    The full project-level CSR validator is used later during the
    final consensus validation stage.

    This function does not modify the CSR.
    """

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

    # --------------------------------------------------------
    # Entities
    # --------------------------------------------------------

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

        if not all(isinstance(x, str) for x in attributes):
            return False

    # --------------------------------------------------------
    # Actions
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Relations
    # --------------------------------------------------------

    for relation in csr["relations"]:

        if not isinstance(relation, dict):
            return False

        if relation.get("subject") not in entity_ids:
            return False

        if relation.get("object") not in entity_ids:
            return False

        if not isinstance(relation.get("relation"), str):
            return False

    # --------------------------------------------------------
    # Scene
    # --------------------------------------------------------

    scene = csr["scene"]

    if "location" not in scene:
        return False

    if "environment" not in scene:
        return False

    if scene["location"] is not None and not isinstance(
        scene["location"],
        str,
    ):
        return False

    if not isinstance(scene["environment"], list):
        return False

    return True


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json_from_response(
    text: Optional[str],
) -> Optional[dict]:
    """
    Robustly extract a JSON object from model output.

    Handles:
    - plain JSON
    - ```json ... ```
    - surrounding explanatory text
    """

    if not text:
        return None

    text = text.strip()

    # Remove markdown code fences.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Try direct parsing first.
    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # Locate the first JSON object.
    start = text.find("{")

    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):

        char = text[index]

        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:

                candidate = text[start:index + 1]

                try:
                    parsed = json.loads(candidate)

                    if isinstance(parsed, dict):
                        return parsed

                except json.JSONDecodeError:
                    return None

    return None


# ============================================================
# MODEL RESPONSE HANDLING
# ============================================================

def get_response_content(
    response: Any,
) -> Optional[str]:
    """
    Safely retrieve message content from an OpenAI-compatible response.

    Prevents crashes when:
        response.choices is None
        choices is empty
        message is missing
        content is missing
    """

    if response is None:
        return None

    choices = getattr(response, "choices", None)

    if not choices:
        print("WARNING: Model returned no choices.")
        print(response)
        return None

    first_choice = choices[0]

    message = getattr(
        first_choice,
        "message",
        None,
    )

    if message is None:
        print(
            "WARNING: Model response contains no message."
        )
        print(first_choice)
        return None

    content = getattr(
        message,
        "content",
        None,
    )

    if content is None:
        print(
            "WARNING: Model message contains no content."
        )
        print(message)
        return None

    if not isinstance(content, str):
        return str(content)

    return content.strip()


# ============================================================
# ALIGNMENT RESULT VALIDATION
# ============================================================

def validate_alignment_result(
    result: Any,
    csr_a: dict,
    csr_b: dict,
) -> bool:
    """
    Validate the semantic alignment response.

    The model is NOT allowed to create new entities/actions/
    relations. Every reference must exist in the input CSRs.
    """

    if not isinstance(result, dict):
        return False

    required = {
        "entity_matches",
        "attribute_matches",
        "action_matches",
        "relation_matches",
        "scene_match",
    }

    if not required.issubset(result.keys()):
        return False

    # --------------------------------------------------------
    # Basic list structure
    # --------------------------------------------------------

    if not isinstance(
        result["entity_matches"],
        list,
    ):
        return False

    if not isinstance(
        result["attribute_matches"],
        list,
    ):
        return False

    if not isinstance(
        result["action_matches"],
        list,
    ):
        return False

    if not isinstance(
        result["relation_matches"],
        list,
    ):
        return False

    # --------------------------------------------------------
    # Entity IDs
    # --------------------------------------------------------

    entity_ids_a = {
        entity["id"]
        for entity in csr_a["entities"]
    }

    entity_ids_b = {
        entity["id"]
        for entity in csr_b["entities"]
    }

    # --------------------------------------------------------
    # Entity matches
    # --------------------------------------------------------

    entity_pairs = set()

    for item in result["entity_matches"]:

        if not isinstance(item, dict):
            return False

        a = item.get("a")
        b = item.get("b")

        if a not in entity_ids_a:
            return False

        if b not in entity_ids_b:
            return False

        pair = (a, b)

        if pair in entity_pairs:
            return False

        entity_pairs.add(pair)

    # --------------------------------------------------------
    # Attribute matches
    # --------------------------------------------------------

    entities_a = {
        entity["id"]: entity
        for entity in csr_a["entities"]
    }

    entities_b = {
        entity["id"]: entity
        for entity in csr_b["entities"]
    }

    for item in result["attribute_matches"]:

        if not isinstance(item, dict):
            return False

        entity_a = item.get("entity_a")
        entity_b = item.get("entity_b")

        if entity_a not in entity_ids_a:
            return False

        if entity_b not in entity_ids_b:
            return False

        matches = item.get("matches", [])

        if not isinstance(matches, list):
            return False

        attributes_a = entities_a[
            entity_a
        ]["attributes"]

        attributes_b = entities_b[
            entity_b
        ]["attributes"]

        for pair in matches:

            if not isinstance(pair, dict):
                return False

            attr_a = pair.get("a")
            attr_b = pair.get("b")

            if attr_a not in attributes_a:
                return False

            if attr_b not in attributes_b:
                return False

    # --------------------------------------------------------
    # Action matches
    # --------------------------------------------------------

    actions_a = csr_a["actions"]
    actions_b = csr_b["actions"]

    for item in result["action_matches"]:

        if not isinstance(item, dict):
            return False

        index_a = item.get("a_index")
        index_b = item.get("b_index")

        if not isinstance(index_a, int):
            return False

        if not isinstance(index_b, int):
            return False

        if not 0 <= index_a < len(actions_a):
            return False

        if not 0 <= index_b < len(actions_b):
            return False

        if "confidence" not in item:
            return False

        confidence = item["confidence"]

        if not isinstance(
            confidence,
            (int, float),
        ):
            return False

        if not 0 <= confidence <= 1:
            return False

    # --------------------------------------------------------
    # Relation matches
    # --------------------------------------------------------

    relations_a = csr_a["relations"]
    relations_b = csr_b["relations"]

    for item in result["relation_matches"]:

        if not isinstance(item, dict):
            return False

        index_a = item.get("a_index")
        index_b = item.get("b_index")

        if not isinstance(index_a, int):
            return False

        if not isinstance(index_b, int):
            return False

        if not 0 <= index_a < len(relations_a):
            return False

        if not 0 <= index_b < len(relations_b):
            return False

        if "confidence" not in item:
            return False

        confidence = item["confidence"]

        if not isinstance(
            confidence,
            (int, float),
        ):
            return False

        if not 0 <= confidence <= 1:
            return False

    # --------------------------------------------------------
    # Scene match
    # --------------------------------------------------------

    scene_match = result["scene_match"]

    if not isinstance(
        scene_match,
        dict,
    ):
        return False

    if not isinstance(
        scene_match.get("equivalent"),
        bool,
    ):
        return False

    if "confidence" not in scene_match:
        return False

    confidence = scene_match["confidence"]

    if not isinstance(
        confidence,
        (int, float),
    ):
        return False

    if not 0 <= confidence <= 1:
        return False

    return True


# ============================================================
# SEMANTIC ALIGNMENT PROMPT
# ============================================================

def build_alignment_prompt(
    csr_a: dict,
    csr_b: dict,
) -> str:
    """
    Construct the alignment prompt.

    No image-specific information is inserted.
    """

    return f"""
You are a semantic alignment system.

You will receive two Canonical Semantic Representations (CSRs)
generated independently from descriptions of the same image.

Your task is to determine which elements in CSR A and CSR B
express the same underlying meaning.

IMPORTANT:
- Do not rewrite either CSR.
- Do not invent information.
- Do not add entities, actions, relations, or attributes.
- Entity IDs are arbitrary local identifiers.
- Compare semantic content rather than IDs.
- Be conservative.
- If equivalence is uncertain, do not mark it as a match.
- A semantic match does not require identical wording.
- Morphological variation may represent the same concept.
- Grammatical reformulation may represent the same event or relation.
- A more specific description and a less specific description may
  refer to the same entity when the evidence supports it.
- Do not merge distinct entities merely because they are related.
- Do not infer unstated properties.

CSR A:
{json.dumps(csr_a, indent=2, ensure_ascii=False)}

CSR B:
{json.dumps(csr_b, indent=2, ensure_ascii=False)}

Return ONLY valid JSON using exactly this structure:

{{
  "entity_matches": [
    {{
      "a": "entity_id_from_A",
      "b": "entity_id_from_B"
    }}
  ],

  "attribute_matches": [
    {{
      "entity_a": "entity_id_from_A",
      "entity_b": "entity_id_from_B",
      "matches": [
        {{
          "a": "attribute_from_A",
          "b": "attribute_from_B"
        }}
      ]
    }}
  ],

  "action_matches": [
    {{
      "a_index": 0,
      "b_index": 0,
      "confidence": 0.0
    }}
  ],

  "relation_matches": [
    {{
      "a_index": 0,
      "b_index": 0,
      "confidence": 0.0
    }}
  ],

  "scene_match": {{
    "equivalent": false,
    "confidence": 0.0
  }}
}}

Confidence must be a number between 0 and 1.

Only include an attribute match when the two attributes describe
the same semantic property.

Only include an action match when the two actions describe the
same underlying event.

Only include a relation match when the two relations describe the
same underlying relationship between corresponding entities.

Return no explanation and no Markdown.
"""


# ============================================================
# ONE MODEL CALL
# ============================================================

def align_csr_pair(
    csr_a: dict,
    csr_b: dict,
) -> Optional[dict]:
    """
    Perform semantic alignment using exactly ONE model call
    per attempt.

    Returns a validated alignment result or None.
    """

    # IMPORTANT:
    # These are inner CSRs and intentionally do not contain image_id.
    # Therefore use the local structural validator here.

    if not validate_csr_structure(csr_a):
        raise ValueError("CSR A has invalid structure.")

    if not validate_csr_structure(csr_b):
        raise ValueError("CSR B has invalid structure.")

    prompt = build_alignment_prompt(
        csr_a,
        csr_b,
    )

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = client.chat.completions.create(
                model=ALIGNMENT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You perform conservative semantic "
                            "alignment and return valid JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=ALIGNMENT_TEMPERATURE,
                extra_body={
                    "reasoning": {
                        "exclude": True
                    }
                },
            )

            content = get_response_content(
                response
            )

            if content is None:

                print(
                    f"  Attempt {attempt}: "
                    "empty model response."
                )

            else:

                result = extract_json_from_response(
                    content
                )

                if result is not None:

                    if validate_alignment_result(
                        result,
                        csr_a,
                        csr_b,
                    ):
                        return result

                    print(
                        f"  Attempt {attempt}: "
                        "invalid alignment structure."
                    )

                else:

                    print(
                        f"  Attempt {attempt}: "
                        "could not parse JSON."
                    )

        except Exception as error:

            print(
                f"  Attempt {attempt}: "
                f"{type(error).__name__}: {error}"
            )

        if attempt < MAX_RETRIES:

            time.sleep(
                RETRY_DELAY_SECONDS * attempt
            )

    return None


# ============================================================
# DATA LOADING
# ============================================================

def load_stability_results() -> List[dict]:
    """
    Load the existing 15-record stability dataset.
    """

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(data, list):

        raise ValueError(
            "Expected stability_15_results.json "
            "to contain a list."
        )

    return data


# ============================================================
# PAIR CREATION
# ============================================================

def group_by_image(
    records: List[dict],
) -> Dict[str, List[dict]]:
    """
    Group records by image ID.
    """

    groups: Dict[str, List[dict]] = {}

    for record in records:

        if record.get("status") != "success":
            continue

        image_id = record.get("image_id")

        if not image_id:
            continue

        csr = record.get("csr")

        if not validate_csr_structure(csr):
            continue

        groups.setdefault(
            image_id,
            [],
        ).append(record)

    return groups


def create_pairs(
    records: List[dict],
) -> List[dict]:
    """
    Create unique unordered CSR pairs within each image.

    For 3 runs per image:
        3 choose 2 = 3 pairs/image

    With 5 images:
        15 pairs total.
    """

    groups = group_by_image(records)

    pairs = []

    for image_id in sorted(groups):

        image_records = sorted(
            groups[image_id],
            key=lambda record: (
                record.get(
                    "baseline_run_number",
                    0,
                ),
                record.get(
                    "extraction_run_number",
                    0,
                ),
            ),
        )

        for i in range(
            len(image_records)
        ):

            for j in range(
                i + 1,
                len(image_records),
            ):

                record_a = image_records[i]
                record_b = image_records[j]

                pairs.append(
                    {
                        "pair_id": (
                            f"{image_id}_"
                            f"{record_a.get('baseline_run_number')}_"
                            f"{record_a.get('extraction_run_number')}_"
                            f"{record_b.get('extraction_run_number')}"
                        ),
                        "image_id": image_id,
                        "record_a": record_a,
                        "record_b": record_b,
                    }
                )

    return pairs


# ============================================================
# RESUME SUPPORT
# ============================================================

def load_existing_results() -> Dict[str, dict]:
    """
    Load previously completed alignment results.

    This makes the experiment resumable.
    """

    if not OUTPUT_FILE.exists():
        return {}

    try:

        with open(
            OUTPUT_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if not isinstance(data, list):
            return {}

        return {
            item["pair_id"]: item
            for item in data
            if isinstance(item, dict)
            and "pair_id" in item
            and item.get("status") == "success"
        }

    except Exception:

        print(
            "WARNING: Existing alignment results "
            "could not be loaded. Starting fresh."
        )

        return {}


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results: List[dict],
) -> None:
    """
    Save results after every pair.

    This prevents loss of completed API calls.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered = sorted(
        results,
        key=lambda item: item.get(
            "pair_id",
            "",
        ),
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            ordered,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_alignment_experiment() -> None:

    print("=" * 70)
    print("SEMANTIC ALIGNMENT EXPERIMENT")
    print("=" * 70)

    print(
        f"Model: {ALIGNMENT_MODEL}"
    )

    print(
        f"Temperature: {ALIGNMENT_TEMPERATURE}"
    )

    print(
        f"Input: {INPUT_FILE}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    records = load_stability_results()

    print(
        f"\nLoaded {len(records)} stability records."
    )

    # --------------------------------------------------------
    # Create pairs
    # --------------------------------------------------------

    pairs = create_pairs(records)

    print(
        f"Created {len(pairs)} CSR pairs."
    )

    # --------------------------------------------------------
    # Existing results
    # --------------------------------------------------------

    existing = load_existing_results()

    print(
        f"Existing successful alignments: "
        f"{len(existing)}"
    )

    results = list(existing.values())

    # --------------------------------------------------------
    # Process pairs
    # --------------------------------------------------------

    for number, pair in enumerate(
        pairs,
        start=1,
    ):

        pair_id = pair["pair_id"]

        if pair_id in existing:

            print(
                f"\n[{number}/{len(pairs)}] "
                f"{pair_id} -- already completed"
            )

            continue

        print(
            f"\n[{number}/{len(pairs)}] "
            f"{pair_id}"
        )

        record_a = pair["record_a"]
        record_b = pair["record_b"]

        csr_a = record_a["csr"]
        csr_b = record_b["csr"]

        print(
            "  Running semantic alignment..."
        )

        alignment = align_csr_pair(
            csr_a,
            csr_b,
        )

        if alignment is None:

            result = {
                "pair_id": pair_id,
                "image_id": pair["image_id"],
                "record_a": {
                    "baseline_run_number":
                        record_a.get(
                            "baseline_run_number"
                        ),
                    "extraction_run_number":
                        record_a.get(
                            "extraction_run_number"
                        ),
                },
                "record_b": {
                    "baseline_run_number":
                        record_b.get(
                            "baseline_run_number"
                        ),
                    "extraction_run_number":
                        record_b.get(
                            "extraction_run_number"
                        ),
                },
                "model": ALIGNMENT_MODEL,
                "temperature": ALIGNMENT_TEMPERATURE,
                "status": "failed",
                "error": (
                    "No valid alignment response "
                    "after retries."
                ),
            }

            # Keep failures for debugging, but they
            # will not count as successful results.
            results.append(result)

            save_results(results)

            print(
                "  FAILED -- saved for later retry."
            )

            continue

        # ----------------------------------------------------
        # Count matches for quick inspection
        # ----------------------------------------------------

        entity_count = len(
            alignment["entity_matches"]
        )

        attribute_count = sum(
            len(item.get("matches", []))
            for item in alignment[
                "attribute_matches"
            ]
        )

        action_count = len(
            alignment["action_matches"]
        )

        relation_count = len(
            alignment["relation_matches"]
        )

        scene_equivalent = alignment[
            "scene_match"
        ]["equivalent"]

        result = {
            "pair_id": pair_id,
            "image_id": pair["image_id"],
            "record_a": {
                "baseline_run_number":
                    record_a.get(
                        "baseline_run_number"
                    ),
                "extraction_run_number":
                    record_a.get(
                        "extraction_run_number"
                    ),
            },
            "record_b": {
                "baseline_run_number":
                    record_b.get(
                        "baseline_run_number"
                    ),
                "extraction_run_number":
                    record_b.get(
                        "extraction_run_number"
                    ),
            },
            "model": ALIGNMENT_MODEL,
            "temperature": ALIGNMENT_TEMPERATURE,
            "status": "success",
            "alignment": alignment,
            "summary": {
                "entity_matches": entity_count,
                "attribute_matches": attribute_count,
                "action_matches": action_count,
                "relation_matches": relation_count,
                "scene_equivalent": scene_equivalent,
            },
        }

        results.append(result)

        save_results(results)

        print(
            f"  Entities:   {entity_count}"
        )

        print(
            f"  Attributes: {attribute_count}"
        )

        print(
            f"  Actions:    {action_count}"
        )

        print(
            f"  Relations:  {relation_count}"
        )

        print(
            f"  Scene:      {scene_equivalent}"
        )

        print("  Saved.")

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    successful = [
        result
        for result in results
        if result.get("status") == "success"
    ]

    failed = [
        result
        for result in results
        if result.get("status") == "failed"
    ]

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"Total pairs:      {len(pairs)}"
    )

    print(
        f"Successful:       {len(successful)}"
    )

    print(
        f"Failed:           {len(failed)}"
    )

    print(
        f"Results saved to: {OUTPUT_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_alignment_experiment()

