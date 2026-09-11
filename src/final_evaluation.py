import os
import json
import csv
from collections import Counter, defaultdict

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

BASELINE_T07 = os.path.join(
    PROJECT_ROOT, "experiments", "baseline_results", "baseline_results_t0.7.json"
)
BASELINE_T00 = os.path.join(
    PROJECT_ROOT, "experiments", "baseline_results", "baseline_results_t0.0.json"
)
ALIGNMENT_FILE = os.path.join(
    PROJECT_ROOT, "results", "alignment_evaluation_results.json"
)
RENDERER_FILE = os.path.join(
    PROJECT_ROOT, "results", "deterministic_captions_v3.json"
)
STABILITY_FILE = os.path.join(
    PROJECT_ROOT, "results", "stability_15_results.json"
)
FROZEN_CONSENSUS_DIR = os.path.join(
    PROJECT_ROOT, "results", "frozen_consensus"
)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "final_evaluation.json")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "final_evaluation.csv")

# ============================================================
# MODEL
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Manually selected reference captions for the five evaluation images.
# These are reference descriptions, not model-generated evaluation targets.
REFERENCE_CAPTIONS = {
    "test_001": "A tan dog runs through a grassy field dotted with yellow flowers.",
    "test_002": (
        "A rustic log cabin sits on a calm lake, its reflection mirrored in the water "
        "with majestic snow-capped mountains rising in the background under a soft, golden sky."
    ),
    "test_003": (
        "A man with glasses and a beard, wearing a white shirt, is focused on typing "
        "on a laptop at a desk illuminated by a warm desk lamp."
    ),
    "test_004": (
        "A woman wearing a brown tank top and blue jeans walks a white dog on a red leash "
        "through a grassy park area with trees in the background."
    ),
    "test_005": (
        "A teacher sits on the floor with a group of young children in a classroom, "
        "guiding them as they play with wooden toys and engage in educational activities."
    ),
}

# ============================================================
# HELPERS
# ============================================================

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_outputs(records):
    grouped = defaultdict(list)
    for record in records:
        if record.get("status") != "success":
            continue
        text = record.get("caption")
        if text is None:
            text = record.get("output")
        if not isinstance(text, str) or not text.strip():
            continue
        image_id = record.get("image_id")
        if image_id is not None:
            grouped[image_id].append(text.strip())
    return dict(grouped)


def count_record_status(records):
    successful = sum(1 for r in records if r.get("status") == "success")
    return {
        "total_records": len(records),
        "successful_records": successful,
        "failed_records": len(records) - successful,
    }

# ============================================================
# REPRODUCIBILITY
# ============================================================

def calculate_exact_match(outputs):
    if not outputs:
        return 0.0
    return Counter(outputs).most_common(1)[0][1] / len(outputs)


def calculate_unique_ratio(outputs):
    if not outputs:
        return 0.0
    return len(set(outputs)) / len(outputs)


def calculate_semantic_similarity(outputs, model):
    if len(outputs) < 2:
        return 1.0
    embeddings = model.encode(outputs)
    matrix = cosine_similarity(embeddings)
    values = [
        matrix[i][j]
        for i in range(len(outputs))
        for j in range(i + 1, len(outputs))
    ]
    return float(np.mean(values)) if values else 1.0


def evaluate_baseline(path, model):
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records in: {path}")

    status_counts = count_record_status(records)
    grouped = group_outputs(records)
    image_results = {}

    for image_id, outputs in sorted(grouped.items()):
        image_results[image_id] = {
            "num_outputs": len(outputs),
            "num_unique_outputs": len(set(outputs)),
            "exact_match_rate": round(calculate_exact_match(outputs), 4),
            "unique_output_ratio": round(calculate_unique_ratio(outputs), 4),
            "mean_pairwise_semantic_similarity": round(
                calculate_semantic_similarity(outputs, model), 4
            ),
        }

    if image_results:
        overall_exact = float(np.mean([x["exact_match_rate"] for x in image_results.values()]))
        overall_unique = float(np.mean([x["unique_output_ratio"] for x in image_results.values()]))
        overall_semantic = float(
            np.mean([x["mean_pairwise_semantic_similarity"] for x in image_results.values()])
        )
    else:
        overall_exact = overall_unique = overall_semantic = 0.0

    return {
        "experiment": {
            "source_file": os.path.relpath(path, PROJECT_ROOT),
            **status_counts,
            "num_images": len(image_results),
        },
        "image_summary": image_results,
        "overall": {
            "num_outputs": sum(x["num_outputs"] for x in image_results.values()),
            "num_unique_outputs": sum(x["num_unique_outputs"] for x in image_results.values()),
            "exact_match_rate": round(overall_exact, 4),
            "unique_output_ratio": round(overall_unique, 4),
            "mean_pairwise_semantic_similarity": round(overall_semantic, 4),
        },
    }

# ============================================================
# SEMANTIC ALIGNMENT
# ============================================================

def extract_alignment_results():
    data = load_json(ALIGNMENT_FILE)
    required = ["global_summary", "image_summary", "experiment"]
    for section in required:
        if section not in data:
            raise ValueError(f"Missing '{section}' in alignment evaluation.")

    summary = data["global_summary"]
    metrics = [
        "entity_consistency",
        "attribute_consistency",
        "action_consistency",
        "relation_consistency",
        "scene_consistency",
        "overall_consistency",
    ]
    for metric in metrics:
        if metric not in summary:
            raise ValueError(f"Missing alignment metric: {metric}")

    return {
        **{metric: summary[metric] for metric in metrics},
        "image_summary": data["image_summary"],
        "experiment": data["experiment"],
    }

# ============================================================
# DETERMINISTIC RENDERER
# ============================================================

def evaluate_renderer():
    data = load_json(RENDERER_FILE)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in renderer file: {RENDERER_FILE}")

    grouped = defaultdict(list)
    for record in data:
        if record.get("status", "success") != "success":
            continue
        image_id = record.get("image_id")
        caption = record.get("rendered_caption", record.get("caption"))
        if image_id is not None and isinstance(caption, str) and caption.strip():
            grouped[image_id].append(caption.strip())

    image_summary = {}
    for image_id, captions in sorted(grouped.items()):
        unique_count = len(set(captions))
        image_summary[image_id] = {
            "num_renders": len(captions),
            "num_unique_outputs": unique_count,
            "reproducible": unique_count == 1,
        }

    reproducible_images = sum(x["reproducible"] for x in image_summary.values())
    total_images = len(image_summary)

    return {
        "num_images": total_images,
        "total_renders": sum(len(v) for v in grouped.values()),
        "reproducible_images": reproducible_images,
        "reproducibility_rate": round(
            reproducible_images / total_images if total_images else 0.0, 4
        ),
        "total_unique_outputs": sum(x["num_unique_outputs"] for x in image_summary.values()),
        "image_summary": image_summary,
    }

# ============================================================
# PERSISTENT FROZEN PIPELINE
# ============================================================

def evaluate_frozen_reproducibility(expected_image_ids=None):
    if not os.path.exists(FROZEN_CONSENSUS_DIR):
        raise FileNotFoundError(
            f"Frozen consensus directory not found: {FROZEN_CONSENSUS_DIR}"
        )

    if expected_image_ids is None:
        expected_image_ids = [
            x for x in os.listdir(FROZEN_CONSENSUS_DIR)
            if os.path.isdir(os.path.join(FROZEN_CONSENSUS_DIR, x))
        ]

    image_summary = {}
    for image_id in sorted(expected_image_ids):
        path = os.path.join(FROZEN_CONSENSUS_DIR, image_id, "consensus.json")
        exists = os.path.exists(path)
        valid = False
        if exists:
            try:
                data = load_json(path)
                valid = isinstance(data, dict) and isinstance(data.get("csr"), dict)
            except Exception:
                valid = False
        image_summary[image_id] = {
            "frozen_consensus_exists": exists,
            "valid_structure": valid,
            "reproducible": exists and valid,
        }

    total = len(image_summary)
    frozen = sum(x["frozen_consensus_exists"] for x in image_summary.values())
    reproducible = sum(x["reproducible"] for x in image_summary.values())

    return {
        "num_images": total,
        "frozen_consensus_images": frozen,
        "reproducible_images": reproducible,
        "reproducibility_rate": round(reproducible / total if total else 0.0, 4),
        "image_summary": image_summary,
        "interpretation": (
            "This measures persistent consensus coverage. Actual repeated execution "
            "reproducibility is verified by comparing final captions across runs."
        ),
    }

# ============================================================
# SEMANTIC FIDELITY
# ============================================================

def evaluate_semantic_fidelity(model, expected_image_ids):
    """
    Semantic fidelity = cosine similarity between the final rendered caption
    and a manually prepared reference caption for the same image.

    This is an embedding-based semantic fidelity metric. It does not claim
    pixel-level correctness and does not use an LLM during evaluation.
    """
    image_results = {}
    missing_references = []

    for image_id in sorted(expected_image_ids):
        reference = REFERENCE_CAPTIONS.get(image_id)
        if reference is None:
            missing_references.append(image_id)
            continue

        consensus_path = os.path.join(
            FROZEN_CONSENSUS_DIR, image_id, "consensus.json"
        )
        consensus = load_json(consensus_path)
        final_caption = consensus.get("rendered_caption") or consensus.get("caption")

        # The frozen consensus file normally contains CSR only. In that case,
        # obtain the deterministic final caption from the renderer output.
        if not isinstance(final_caption, str):
            final_caption = None
            renderer_data = load_json(RENDERER_FILE)
            for record in renderer_data:
                if record.get("image_id") == image_id:
                    candidate = record.get("rendered_caption", record.get("caption"))
                    if isinstance(candidate, str) and candidate.strip():
                        final_caption = candidate.strip()
                        break

        if not final_caption:
            image_results[image_id] = {
                "reference_caption": reference,
                "final_caption": None,
                "semantic_similarity": None,
                "status": "missing_final_caption",
            }
            continue

        embeddings = model.encode([reference, final_caption])
        score = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

        image_results[image_id] = {
            "reference_caption": reference,
            "final_caption": final_caption,
            "semantic_similarity": round(score, 4),
            "status": "success",
        }

    valid_scores = [
        x["semantic_similarity"]
        for x in image_results.values()
        if x["semantic_similarity"] is not None
    ]

    return {
        "metric": "embedding_semantic_fidelity",
        "embedding_model": EMBEDDING_MODEL,
        "num_images": len(expected_image_ids),
        "evaluated_images": len(valid_scores),
        "mean_semantic_fidelity": round(float(np.mean(valid_scores)), 4) if valid_scores else 0.0,
        "image_summary": image_results,
        "missing_references": missing_references,
        "interpretation": (
            "Higher values indicate greater semantic similarity between the final "
            "description and the reference description."
        ),
    }

# ============================================================
# INFORMATIVENESS
# ============================================================

def csr_fact_count(csr):
    """Count explicit semantic information units in a CSR."""
    if not isinstance(csr, dict):
        return 0

    entities = csr.get("entities") or []
    actions = csr.get("actions") or []
    relations = csr.get("relations") or []
    scene = csr.get("scene") or {}

    entity_facts = len(entities)
    attribute_facts = sum(len(e.get("attributes") or []) for e in entities if isinstance(e, dict))
    action_facts = len(actions)
    relation_facts = len(relations)
    scene_facts = len(scene.get("environment") or [])
    if scene.get("location"):
        scene_facts += 1

    return entity_facts + attribute_facts + action_facts + relation_facts + scene_facts


def csr_breakdown(csr):
    if not isinstance(csr, dict):
        return {
            "entities": 0,
            "attributes": 0,
            "actions": 0,
            "relations": 0,
            "scene_facts": 0,
            "total_semantic_facts": 0,
        }

    entities = csr.get("entities") or []
    actions = csr.get("actions") or []
    relations = csr.get("relations") or []
    scene = csr.get("scene") or {}

    result = {
        "entities": len(entities),
        "attributes": sum(len(e.get("attributes") or []) for e in entities if isinstance(e, dict)),
        "actions": len(actions),
        "relations": len(relations),
        "scene_facts": len(scene.get("environment") or []) + (1 if scene.get("location") else 0),
    }
    result["total_semantic_facts"] = sum(result.values())
    return result


def load_stability_csrs():
    data = load_json(STABILITY_FILE)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in stability file: {STABILITY_FILE}")

    grouped = defaultdict(list)
    for record in data:
        if record.get("status") != "success":
            continue
        image_id = record.get("image_id")
        csr = record.get("csr")
        if image_id is not None and isinstance(csr, dict):
            grouped[image_id].append(csr)
    return grouped


def evaluate_informativeness(expected_image_ids):
    """
    Informativeness is evaluated as semantic information coverage:

        consensus semantic facts / maximum semantic facts in candidate CSRs

    The candidate CSRs are the previously extracted representations from the
    stability experiment. Supporting counts are also reported so the metric
    is transparent rather than relying only on caption length.
    """
    stability_csrs = load_stability_csrs()
    image_results = {}

    for image_id in sorted(expected_image_ids):
        consensus_path = os.path.join(
            FROZEN_CONSENSUS_DIR, image_id, "consensus.json"
        )

        if not os.path.exists(consensus_path):
            image_results[image_id] = {"status": "missing_consensus"}
            continue

        data = load_json(consensus_path)
        consensus_csr = data.get("csr")
        consensus_breakdown = csr_breakdown(consensus_csr)
        consensus_facts = consensus_breakdown["total_semantic_facts"]

        candidates = stability_csrs.get(image_id, [])
        candidate_counts = [csr_fact_count(csr) for csr in candidates]
        max_candidate_facts = max(candidate_counts) if candidate_counts else 0

        coverage = (
            consensus_facts / max_candidate_facts
            if max_candidate_facts
            else 0.0
        )

        image_results[image_id] = {
            "status": "success",
            "consensus_semantic_facts": consensus_facts,
            "maximum_candidate_semantic_facts": max_candidate_facts,
            "information_coverage": round(min(coverage, 1.0), 4),
            "candidate_fact_counts": candidate_counts,
            "consensus_breakdown": consensus_breakdown,
        }

    scores = [
        x["information_coverage"]
        for x in image_results.values()
        if x.get("status") == "success"
    ]

    return {
        "metric": "semantic_information_coverage",
        "num_images": len(expected_image_ids),
        "evaluated_images": len(scores),
        "mean_information_coverage": round(float(np.mean(scores)), 4) if scores else 0.0,
        "image_summary": image_results,
        "interpretation": (
            "Higher values indicate that the consensus CSR retains more of the "
            "semantic information present in the candidate CSR set. This metric "
            "measures retained semantic detail, not factual correctness by itself."
        ),
    }

# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    print("\nLoading embedding model:")
    print(EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("\nEvaluating baseline T=0.7...")
    baseline_t07 = evaluate_baseline(BASELINE_T07, model)
    print("T=0.7 records:", baseline_t07["experiment"]["total_records"])
    print("T=0.7 successful:", baseline_t07["experiment"]["successful_records"])
    print("T=0.7 failed:", baseline_t07["experiment"]["failed_records"])

    print("\nEvaluating baseline T=0.0...")
    baseline_t00 = evaluate_baseline(BASELINE_T00, model)
    print("T=0.0 records:", baseline_t00["experiment"]["total_records"])
    print("T=0.0 successful:", baseline_t00["experiment"]["successful_records"])
    print("T=0.0 failed:", baseline_t00["experiment"]["failed_records"])

    print("\nLoading semantic alignment evaluation...")
    alignment = extract_alignment_results()

    print("Evaluating deterministic renderer...")
    renderer = evaluate_renderer()

    print("Evaluating persistent frozen consensus...")
    expected_image_ids = sorted(
        set(baseline_t07["image_summary"]) | set(baseline_t00["image_summary"])
    )
    frozen_reproducibility = evaluate_frozen_reproducibility(expected_image_ids)

    print("Evaluating semantic fidelity...")
    semantic_fidelity = evaluate_semantic_fidelity(model, expected_image_ids)

    print("Evaluating informativeness...")
    informativeness = evaluate_informativeness(expected_image_ids)

    num_images = max(
        baseline_t07["experiment"]["num_images"],
        baseline_t00["experiment"]["num_images"],
    )

    final_results = {
        "experiment": {
            "num_images": num_images,
            "baseline": {
                "temperature_0.7": baseline_t07["experiment"],
                "temperature_0.0": baseline_t00["experiment"],
            },
            "embedding_model": EMBEDDING_MODEL,
            "renderer": "deterministic_renderer_v3",
        },
        "reproducibility": {
            "baseline_t0.7": baseline_t07["overall"],
            "baseline_t0.0": baseline_t00["overall"],
            "deterministic_renderer": renderer,
            "persistent_frozen_pipeline": frozen_reproducibility,
        },
        "semantic_alignment": alignment,
        "semantic_fidelity": semantic_fidelity,
        "informativeness": informativeness,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(final_results), f, indent=2)

    rows = []

    # Baselines
    for temperature, result in [(0.7, baseline_t07), (0.0, baseline_t00)]:
        for image_id, metrics in result["image_summary"].items():
            for metric in [
                "exact_match_rate",
                "unique_output_ratio",
                "mean_pairwise_semantic_similarity",
            ]:
                rows.append({
                    "stage": "baseline",
                    "temperature": temperature,
                    "image_id": image_id,
                    "metric": metric,
                    "value": metrics[metric],
                })

    # Alignment
    for metric in [
        "entity_consistency", "attribute_consistency", "action_consistency",
        "relation_consistency", "scene_consistency", "overall_consistency",
    ]:
        rows.append({
            "stage": "semantic_alignment",
            "temperature": "",
            "image_id": "GLOBAL",
            "metric": metric,
            "value": alignment[metric],
        })

    # Reproducibility
    for stage, data in [
        ("deterministic_renderer", renderer),
        ("persistent_frozen_pipeline", frozen_reproducibility),
    ]:
        rows.append({
            "stage": stage, "temperature": "", "image_id": "GLOBAL",
            "metric": "reproducibility_rate", "value": data["reproducibility_rate"],
        })

    # Semantic fidelity
    rows.append({
        "stage": "semantic_fidelity", "temperature": "", "image_id": "GLOBAL",
        "metric": "mean_semantic_fidelity",
        "value": semantic_fidelity["mean_semantic_fidelity"],
    })
    for image_id, data in semantic_fidelity["image_summary"].items():
        if data.get("semantic_similarity") is not None:
            rows.append({
                "stage": "semantic_fidelity", "temperature": "", "image_id": image_id,
                "metric": "semantic_similarity", "value": data["semantic_similarity"],
            })

    # Informativeness
    rows.append({
        "stage": "informativeness", "temperature": "", "image_id": "GLOBAL",
        "metric": "mean_information_coverage",
        "value": informativeness["mean_information_coverage"],
    })
    for image_id, data in informativeness["image_summary"].items():
        if data.get("status") == "success":
            rows.append({
                "stage": "informativeness", "temperature": "", "image_id": image_id,
                "metric": "information_coverage", "value": data["information_coverage"],
            })
            rows.append({
                "stage": "informativeness", "temperature": "", "image_id": image_id,
                "metric": "semantic_fact_count", "value": data["consensus_semantic_facts"],
            })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stage", "temperature", "image_id", "metric", "value"],
        )
        writer.writeheader()
        writer.writerows(make_json_serializable(rows))

    print("\n" + "=" * 70)
    print("BASELINE T=0.7")
    print("=" * 70)
    print("Exact Match Rate:", baseline_t07["overall"]["exact_match_rate"])
    print("Unique Output Ratio:", baseline_t07["overall"]["unique_output_ratio"])
    print("Semantic Similarity:", baseline_t07["overall"]["mean_pairwise_semantic_similarity"])

    print("\n" + "=" * 70)
    print("BASELINE T=0.0")
    print("=" * 70)
    print("Exact Match Rate:", baseline_t00["overall"]["exact_match_rate"])
    print("Unique Output Ratio:", baseline_t00["overall"]["unique_output_ratio"])
    print("Semantic Similarity:", baseline_t00["overall"]["mean_pairwise_semantic_similarity"])

    print("\n" + "=" * 70)
    print("SEMANTIC ALIGNMENT")
    print("=" * 70)
    for metric in [
        "entity_consistency", "attribute_consistency", "action_consistency",
        "relation_consistency", "scene_consistency", "overall_consistency",
    ]:
        print(metric + ":", alignment[metric])

    print("\n" + "=" * 70)
    print("DETERMINISTIC RENDERER")
    print("=" * 70)
    print("Images:", renderer["num_images"])
    print("Total renders:", renderer["total_renders"])
    print("Reproducible images:", renderer["reproducible_images"])
    print("Reproducibility rate:", renderer["reproducibility_rate"])

    print("\n" + "=" * 70)
    print("PERSISTENT FROZEN PIPELINE")
    print("=" * 70)
    print("Images:", frozen_reproducibility["num_images"])
    print("Frozen consensus images:", frozen_reproducibility["frozen_consensus_images"])
    print("Reproducible images:", frozen_reproducibility["reproducible_images"])
    print("Reproducibility rate:", frozen_reproducibility["reproducibility_rate"])

    print("\n" + "=" * 70)
    print("SEMANTIC FIDELITY")
    print("=" * 70)
    print("Evaluated images:", semantic_fidelity["evaluated_images"])
    print("Mean semantic fidelity:", semantic_fidelity["mean_semantic_fidelity"])

    print("\n" + "=" * 70)
    print("INFORMATIVENESS")
    print("=" * 70)
    print("Evaluated images:", informativeness["evaluated_images"])
    print("Mean information coverage:", informativeness["mean_information_coverage"])

    print("\nSaved:")
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)
    print("\nDone.")


if __name__ == "__main__":
    main()
