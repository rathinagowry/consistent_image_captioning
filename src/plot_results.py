import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

FINAL_EVALUATION_FILE = RESULTS_DIR / "final_evaluation.json"

PLOT_DIR = RESULTS_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD RESULTS
# ============================================================

if not FINAL_EVALUATION_FILE.exists():
    raise FileNotFoundError(
        f"Could not find:\n{FINAL_EVALUATION_FILE}"
    )

with open(FINAL_EVALUATION_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)


sns.set_theme(style="whitegrid")


# ============================================================
# 1. BASELINE OUTPUT VARIABILITY
# ============================================================

reproducibility = results["reproducibility"]

baseline_t07 = reproducibility["baseline_t0.7"]
baseline_t00 = reproducibility["baseline_t0.0"]

baseline_df = pd.DataFrame({
    "Temperature": ["T = 0.7", "T = 0.0"],

    "Exact Match Rate": [
        baseline_t07["exact_match_rate"],
        baseline_t00["exact_match_rate"]
    ],

    "Unique Output Ratio": [
        baseline_t07["unique_output_ratio"],
        baseline_t00["unique_output_ratio"]
    ]
})

baseline_long = baseline_df.melt(
    id_vars="Temperature",
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(9, 5.5))

sns.barplot(
    data=baseline_long,
    x="Metric",
    y="Score",
    hue="Temperature"
)

plt.ylim(0, 1)
plt.ylabel("Score")
plt.xlabel("")

plt.title(
    "Baseline Output Variability",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "01_baseline_variability.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. SEMANTIC ALIGNMENT
# ============================================================

alignment = results["semantic_alignment"]

alignment_df = pd.DataFrame({
    "Category": [
        "Entity",
        "Attribute",
        "Action",
        "Relation",
        "Scene"
    ],

    "Consistency": [
        alignment["entity_consistency"],
        alignment["attribute_consistency"],
        alignment["action_consistency"],
        alignment["relation_consistency"],
        alignment["scene_consistency"]
    ]
})

plt.figure(figsize=(8, 5.5))

sns.barplot(
    data=alignment_df,
    x="Category",
    y="Consistency"
)

plt.ylim(0, 1)
plt.ylabel("Consistency")
plt.xlabel("")

plt.title(
    "Semantic Alignment Across CSR Components",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "02_semantic_alignment.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. PROPOSED PIPELINE EVALUATION
# ============================================================

persistent_pipeline = (
    reproducibility["persistent_frozen_pipeline"]
)

reproducibility_rate = (
    persistent_pipeline["reproducibility_rate"]
)

semantic_fidelity = (
    results["semantic_fidelity"]
    ["mean_semantic_fidelity"]
)

information_coverage = (
    results["informativeness"]
    ["mean_information_coverage"]
)

pipeline_df = pd.DataFrame({
    "Metric": [
        "Persistent Reproducibility",
        "Semantic Fidelity",
        "Information Coverage"
    ],

    "Score": [
        reproducibility_rate,
        semantic_fidelity,
        information_coverage
    ]
})

plt.figure(figsize=(9, 5.5))

sns.barplot(
    data=pipeline_df,
    x="Metric",
    y="Score"
)

plt.ylim(0, 1)
plt.ylabel("Score")
plt.xlabel("")

plt.title(
    "Proposed Pipeline Evaluation",
    fontsize=15,
    fontweight="bold"
)

plt.xticks(rotation=10)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "03_pipeline_evaluation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. PER-IMAGE SEMANTIC FIDELITY
# ============================================================

fidelity_summary = (
    results["semantic_fidelity"]
    ["image_summary"]
)

fidelity_rows = []

for image_id, data in fidelity_summary.items():

    if data.get("status") != "success":
        continue

    fidelity_rows.append({
        "Image": image_id,
        "Semantic Fidelity": data["semantic_similarity"]
    })

fidelity_df = pd.DataFrame(fidelity_rows)

plt.figure(figsize=(9, 5.5))

sns.barplot(
    data=fidelity_df,
    x="Image",
    y="Semantic Fidelity"
)

plt.ylim(0, 1)

plt.ylabel("Semantic Similarity")
plt.xlabel("Image")

plt.title(
    "Semantic Fidelity Across Test Images",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "04_semantic_fidelity_per_image.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. PER-IMAGE INFORMATION COVERAGE
# ============================================================

information_summary = (
    results["informativeness"]
    ["image_summary"]
)

information_rows = []

for image_id, data in information_summary.items():

    if data.get("status") != "success":
        continue

    information_rows.append({
        "Image": image_id,
        "Information Coverage": data["information_coverage"]
    })

information_df = pd.DataFrame(information_rows)

plt.figure(figsize=(9, 5.5))

sns.barplot(
    data=information_df,
    x="Image",
    y="Information Coverage"
)

plt.ylim(0, 1)

plt.ylabel("Information Coverage")
plt.xlabel("Image")

plt.title(
    "Semantic Information Coverage Across Test Images",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "05_information_coverage_per_image.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE PLOTTING DATA
# ============================================================

baseline_df.to_csv(
    PLOT_DIR / "baseline_plot_data.csv",
    index=False
)

alignment_df.to_csv(
    PLOT_DIR / "alignment_plot_data.csv",
    index=False
)

pipeline_df.to_csv(
    PLOT_DIR / "pipeline_plot_data.csv",
    index=False
)

fidelity_df.to_csv(
    PLOT_DIR / "fidelity_plot_data.csv",
    index=False
)

information_df.to_csv(
    PLOT_DIR / "information_coverage_plot_data.csv",
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 65)
print("PLOTS GENERATED SUCCESSFULLY")
print("=" * 65)

print()
print("Output folder:")
print(PLOT_DIR)

print()
print("Generated graphs:")

print("  1. 01_baseline_variability.png")
print("  2. 02_semantic_alignment.png")
print("  3. 03_pipeline_evaluation.png")
print("  4. 04_semantic_fidelity_per_image.png")
print("  5. 05_information_coverage_per_image.png")

print()
print("KEY RESULTS")
print("-" * 65)

print(
    f"Baseline T=0.7 Exact Match Rate : "
    f"{baseline_t07['exact_match_rate']:.4f}"
)

print(
    f"Baseline T=0.0 Exact Match Rate : "
    f"{baseline_t00['exact_match_rate']:.4f}"
)

print(
    f"Semantic Alignment             : "
    f"{alignment['overall_consistency']:.4f}"
)

print(
    f"Persistent Reproducibility     : "
    f"{reproducibility_rate:.4f}"
)

print(
    f"Mean Semantic Fidelity         : "
    f"{semantic_fidelity:.4f}"
)

print(
    f"Mean Information Coverage      : "
    f"{information_coverage:.4f}"
)

print("=" * 65)