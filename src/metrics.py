import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    JSON_OUTPUT,
)


# CONFIGURATION
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LOAD RESULTS

def load_results():

    with open(
        JSON_OUTPUT,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# EXACT MATCH RATE
def calculate_exact_match_rate(outputs):

    if not outputs:
        return 0.0

    counts = Counter(outputs)

    most_common_count = counts.most_common(1)[0][1]

    return most_common_count / len(outputs)

# UNIQUE OUTPUT RATIO


def calculate_unique_output_ratio(outputs):

    if not outputs:
        return 0.0

    unique_outputs = len(set(outputs))

    return unique_outputs / len(outputs)



# PAIRWISE SEMANTIC SIMILARITY
def calculate_semantic_similarity(
    outputs,
    embedding_model
):

    if len(outputs) < 2:
        return 1.0

    embeddings = embedding_model.encode(
        outputs,
        convert_to_numpy=True
    )

    similarity_matrix = cosine_similarity(
        embeddings
    )

    similarities = []

    for i in range(len(outputs)):

        for j in range(i + 1, len(outputs)):

            similarities.append(
                similarity_matrix[i][j]
            )

    return float(
        np.mean(similarities)
    )



# ANALYZE ONE IMAGE
def analyze_image(
    image_id,
    outputs,
    embedding_model
):

    exact_match_rate = (
        calculate_exact_match_rate(outputs)
    )

    unique_output_ratio = (
        calculate_unique_output_ratio(outputs)
    )

    semantic_similarity = (
        calculate_semantic_similarity(
            outputs,
            embedding_model
        )
    )

    return {
        "image_id": image_id,
        "num_outputs": len(outputs),
        "num_unique_outputs": len(set(outputs)),
        "exact_match_rate": round(
            exact_match_rate,
            4
        ),
        "unique_output_ratio": round(
            unique_output_ratio,
            4
        ),
        "mean_pairwise_semantic_similarity": round(
            semantic_similarity,
            4
        ),
    }


# MAIN ANALYSIS

def calculate_baseline_metrics():

    results = load_results()

    # Keep only successful generations
    successful_results = [
        result
        for result in results
        if result["status"] == "success"
        and result["output"]
    ]

    if not successful_results:

        raise ValueError(
            "No successful generations found."
        )

    # Load embedding model

    print()
    print(
        "Loading embedding model..."
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        "Embedding model loaded."
    )

    # Group outputs by image

    image_outputs = {}

    for result in successful_results:

        image_id = result["image_id"]

        if image_id not in image_outputs:

            image_outputs[image_id] = []

        image_outputs[image_id].append(
            result["output"]
        )

    # Calculate metrics


    metrics = []

    for image_id, outputs in image_outputs.items():

        print()
        print(
            f"Analyzing {image_id}..."
        )

        result = analyze_image(
            image_id=image_id,
            outputs=outputs,
            embedding_model=embedding_model
        )

        metrics.append(result)

        print(
            f"  Outputs       : {result['num_outputs']}"
        )

        print(
            f"  Unique        : {result['num_unique_outputs']}"
        )

        print(
            f"  Exact Match   : "
            f"{result['exact_match_rate']:.2%}"
        )

        print(
            f"  Unique Ratio  : "
            f"{result['unique_output_ratio']:.2%}"
        )

        print(
            f"  Semantic Sim. : "
            f"{result['mean_pairwise_semantic_similarity']:.4f}"
        )
    # Create DataFrame
    dataframe = pd.DataFrame(metrics)

    # Overall averages

    overall = {
        "image_id": "OVERALL",
        "num_outputs": dataframe[
            "num_outputs"
        ].sum(),

        "num_unique_outputs": dataframe[
            "num_unique_outputs"
        ].sum(),

        "exact_match_rate": dataframe[
            "exact_match_rate"
        ].mean(),

        "unique_output_ratio": dataframe[
            "unique_output_ratio"
        ].mean(),

        "mean_pairwise_semantic_similarity":
            dataframe[
                "mean_pairwise_semantic_similarity"
            ].mean(),
    }

    dataframe = pd.concat(
        [
            dataframe,
            pd.DataFrame([overall])
        ],
        ignore_index=True
    )


    # Save results

    output_directory = Path(
        "experiments/baseline_results"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_directory /
        "baseline_metrics.csv"
    )

    dataframe.to_csv(
        output_file,
        index=False
    )



    print()
    print("=" * 60)
    print("BASELINE METRICS")
    print("=" * 60)

    print(
        f"Average Exact Match Rate      : "
        f"{overall['exact_match_rate']:.2%}"
    )

    print(
        f"Average Unique Output Ratio   : "
        f"{overall['unique_output_ratio']:.2%}"
    )

    print(
        f"Average Semantic Similarity   : "
        f"{overall['mean_pairwise_semantic_similarity']:.4f}"
    )

    print("=" * 60)

    print()
    print(
        f"Results saved to: {output_file}"
    )


if __name__ == "__main__":

    calculate_baseline_metrics()