import json
import csv
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
from typing import Any, Dict, List, Tuple, Optional


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

STABILITY_FILE = RESULTS_DIR / "stability_15_results.json"
ALIGNMENT_FILE = RESULTS_DIR / "semantic_alignment_results.json"

OUTPUT_JSON = RESULTS_DIR / "consensus_csrs.json"
OUTPUT_CSV = RESULTS_DIR / "consensus_csrs.csv"


# ============================================================
# CONSENSUS POLICY
# ============================================================

# Strict majority.
#
# For 3 runs:
#       3/3 -> keep
#       2/3 -> keep
#       1/3 -> minority / do not put in main CSR
#
# This is a generic policy, not an image-specific rule.

MIN_SUPPORT_RATIO = 0.5


def minimum_support_count(
    number_of_runs: int,
) -> int:

    # Strict majority.
    return (number_of_runs // 2) + 1


# ============================================================
# FILE UTILITIES
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


def save_json(
    path: Path,
    data: Any,
) -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
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

    if not required.issubset(csr):
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

        if not isinstance(entity_id, str):
            return False

        if entity_id in entity_ids:
            return False

        entity_ids.add(entity_id)

        if not isinstance(
            entity.get("type"),
            str,
        ):
            return False

        if not isinstance(
            entity.get("attributes"),
            list,
        ):
            return False

    for action in csr["actions"]:

        if not isinstance(action, dict):
            return False

        if action.get("subject") not in entity_ids:
            return False

        if not isinstance(
            action.get("action"),
            str,
        ):
            return False

        obj = action.get("object")

        if (
            obj is not None
            and obj not in entity_ids
        ):
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

    return True


# ============================================================
# GROUP RECORDS
# ============================================================

def group_records_by_image(
    records: List[dict],
) -> Dict[str, List[dict]]:

    groups = defaultdict(list)

    for record in records:

        if record.get("status") != "success":
            continue

        csr = record.get("csr")

        if not validate_csr(csr):
            continue

        groups[
            record["image_id"]
        ].append(record)

    for image_id in groups:

        groups[image_id].sort(
            key=lambda record: (
                record["baseline_run_number"],
                record["extraction_run_number"],
            )
        )

    return dict(groups)


# ============================================================
# FIND ALIGNMENT
# ============================================================

def find_alignment(
    image_id: str,
    record_a: dict,
    record_b: dict,
    alignment_results: List[dict],
) -> Optional[dict]:

    a = (
        record_a["baseline_run_number"],
        record_a["extraction_run_number"],
    )

    b = (
        record_b["baseline_run_number"],
        record_b["extraction_run_number"],
    )

    # Pair IDs generated by semantic_aligner.py.
    possible_ids = {
        f"{image_id}_{a[0]}_{a[1]}_{b[1]}",
        f"{image_id}_{b[0]}_{b[1]}_{a[1]}",
    }

    for result in alignment_results:

        if result.get("status") != "success":
            continue

        if result.get("pair_id") in possible_ids:
            return result

    return None


# ============================================================
# UNION FIND
# ============================================================

class UnionFind:

    def __init__(self):

        self.parent = {}
        self.rank = {}

    def add(self, item):

        if item not in self.parent:

            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item):

        if self.parent[item] != item:

            self.parent[item] = self.find(
                self.parent[item]
            )

        return self.parent[item]

    def union(self, a, b):

        self.add(a)
        self.add(b)

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:

            self.parent[root_a] = root_b

        elif self.rank[root_a] > self.rank[root_b]:

            self.parent[root_b] = root_a

        else:

            self.parent[root_b] = root_a
            self.rank[root_a] += 1


# ============================================================
# SAFE CLUSTERING
# ============================================================

def safe_components(
    nodes: List[Tuple],
    edges: List[Tuple],
) -> List[List[Tuple]]:
    """
    Construct connected components.

    Safety condition:
    A semantic cluster must not contain two occurrences
    from the same run.

    If a connected component violates this condition,
    it is split into singleton occurrences rather than
    incorrectly merging two entities/facts from the same run.
    """

    uf = UnionFind()

    for node in nodes:
        uf.add(node)

    for a, b in edges:
        uf.union(a, b)

    raw = defaultdict(list)

    for node in nodes:

        root = uf.find(node)
        raw[root].append(node)

    final_components = []

    for component in raw.values():

        runs = [
            node[0]
            for node in component
        ]

        if len(runs) == len(set(runs)):

            final_components.append(
                sorted(
                    component,
                    key=str,
                )
            )

        else:

            # Conflict:
            # do not make an unsafe semantic merge.
            for node in sorted(
                component,
                key=str,
            ):
                final_components.append(
                    [node]
                )

    final_components.sort(
        key=lambda component: str(component)
    )

    return final_components


# ============================================================
# ENTITY CLUSTERS
# ============================================================

def build_entity_clusters(
    records: List[dict],
    alignment_results: List[dict],
) -> List[List[Tuple]]:

    nodes = []
    edges = []

    # Every entity occurrence is a node.
    for run_index, record in enumerate(records):

        for entity in record["csr"]["entities"]:

            nodes.append(
                (
                    run_index,
                    entity["id"],
                )
            )

    # Pairwise semantic entity matches become edges.
    for i, j in combinations(
        range(len(records)),
        2,
    ):

        alignment = find_alignment(
            records[i]["image_id"],
            records[i],
            records[j],
            alignment_results,
        )

        if alignment is None:
            continue

        matches = alignment.get(
            "alignment",
            {},
        ).get(
            "entity_matches",
            [],
        )

        source_seen = set()
        target_seen = set()

        for match in matches:

            a = match.get("a")
            b = match.get("b")

            if not a or not b:
                continue

            # Enforce one-to-one pairwise mapping.
            if a in source_seen:
                continue

            if b in target_seen:
                continue

            source_seen.add(a)
            target_seen.add(b)

            edges.append(
                (
                    (
                        i,
                        a,
                    ),
                    (
                        j,
                        b,
                    ),
                )
            )

    return safe_components(
        nodes,
        edges,
    )


# ============================================================
# ENTITY ACCESS
# ============================================================

def get_entity(
    record: dict,
    entity_id: str,
) -> Optional[dict]:

    for entity in record["csr"]["entities"]:

        if entity["id"] == entity_id:
            return entity

    return None


# ============================================================
# ENTITY REPRESENTATION
# ============================================================

def choose_mode(
    values: List[str],
) -> Optional[str]:

    if not values:
        return None

    counts = Counter(values)

    maximum = max(
        counts.values()
    )

    candidates = sorted(
        value
        for value, count
        in counts.items()
        if count == maximum
    )

    return candidates[0]


def entity_type(
    cluster: List[Tuple],
    records: List[dict],
) -> str:

    values = []

    for run_index, entity_id in cluster:

        entity = get_entity(
            records[run_index],
            entity_id,
        )

        if entity:
            values.append(
                entity["type"]
            )

    return (
        choose_mode(values)
        or "entity"
    )


def entity_attributes(
    cluster: List[Tuple],
    records: List[dict],
    support_required: int,
) -> Tuple[List[str], Dict[str, int]]:

    attribute_runs = defaultdict(set)

    for run_index, entity_id in cluster:

        entity = get_entity(
            records[run_index],
            entity_id,
        )

        if entity is None:
            continue

        for attribute in set(
            entity.get(
                "attributes",
                [],
            )
        ):

            attribute_runs[
                attribute
            ].add(run_index)

    majority = sorted(
        attribute
        for attribute, runs
        in attribute_runs.items()
        if len(runs) >= support_required
    )

    support = {
        attribute: len(runs)
        for attribute, runs
        in attribute_runs.items()
    }

    return majority, support


# ============================================================
# ACTION SIGNATURE
# ============================================================

def action_signature(
    action: dict,
    run_index: int,
    entity_map: Dict[Tuple, str],
) -> Optional[Tuple]:

    subject = action.get(
        "subject"
    )

    if subject is None:
        return None

    subject_cluster = entity_map.get(
        (
            run_index,
            subject,
        )
    )

    if subject_cluster is None:
        return None

    object_id = action.get(
        "object"
    )

    if object_id is None:

        object_cluster = None

    else:

        object_cluster = entity_map.get(
            (
                run_index,
                object_id,
            )
        )

        if object_cluster is None:
            return None

    return (
        subject_cluster,
        action["action"],
        object_cluster,
    )


# ============================================================
# RELATION SIGNATURE
# ============================================================

def relation_signature(
    relation: dict,
    run_index: int,
    entity_map: Dict[Tuple, str],
) -> Optional[Tuple]:

    subject = relation.get(
        "subject"
    )

    object_id = relation.get(
        "object"
    )

    subject_cluster = entity_map.get(
        (
            run_index,
            subject,
        )
    )

    object_cluster = entity_map.get(
        (
            run_index,
            object_id,
        )
    )

    if (
        subject_cluster is None
        or object_cluster is None
    ):
        return None

    return (
        subject_cluster,
        relation["relation"],
        object_cluster,
    )


# ============================================================
# ACTION CONSENSUS
# ============================================================

def build_actions(
    records: List[dict],
    entity_map: Dict[Tuple, str],
    support_required: int,
) -> Tuple[List[dict], List[dict]]:

    grouped = defaultdict(list)

    for run_index, record in enumerate(records):

        for action_index, action in enumerate(
            record["csr"]["actions"]
        ):

            signature = action_signature(
                action,
                run_index,
                entity_map,
            )

            if signature is None:
                continue

            grouped[
                signature
            ].append(
                {
                    "run_index": run_index,
                    "action_index": action_index,
                    "action": action,
                }
            )

    consensus = []
    minority = []

    for signature, occurrences in sorted(
        grouped.items(),
        key=lambda item: str(item[0]),
    ):

        runs = {
            occurrence["run_index"]
            for occurrence in occurrences
        }

        subject, action_name, object_id = signature

        output = {
            "subject": subject,
            "action": action_name,
            "object": object_id,
        }

        provenance = {
            "fact": output,
            "support": len(runs),
            "runs": sorted(runs),
        }

        if len(runs) >= support_required:

            consensus.append(
                output
            )

        else:

            minority.append(
                provenance
            )

    return consensus, minority


# ============================================================
# RELATION CONSENSUS
# ============================================================

def build_relations(
    records: List[dict],
    entity_map: Dict[Tuple, str],
    support_required: int,
) -> Tuple[List[dict], List[dict]]:

    grouped = defaultdict(list)

    for run_index, record in enumerate(records):

        for relation in record["csr"]["relations"]:

            signature = relation_signature(
                relation,
                run_index,
                entity_map,
            )

            if signature is None:
                continue

            grouped[
                signature
            ].append(run_index)

    consensus = []
    minority = []

    for signature, run_indices in sorted(
        grouped.items(),
        key=lambda item: str(item[0]),
    ):

        runs = set(
            run_indices
        )

        subject, relation_name, object_id = signature

        output = {
            "subject": subject,
            "relation": relation_name,
            "object": object_id,
        }

        provenance = {
            "fact": output,
            "support": len(runs),
            "runs": sorted(runs),
        }

        if len(runs) >= support_required:

            consensus.append(
                output
            )

        else:

            minority.append(
                provenance
            )

    return consensus, minority


# ============================================================
# SCENE CONSENSUS
# ============================================================

def build_scene(
    records: List[dict],
    support_required: int,
) -> Tuple[dict, dict]:

    location_runs = defaultdict(set)
    environment_runs = defaultdict(set)

    for run_index, record in enumerate(records):

        scene = record["csr"]["scene"]

        location = scene.get(
            "location"
        )

        if location is not None:

            location_runs[
                location
            ].add(run_index)

        for item in set(
            scene.get(
                "environment",
                [],
            )
        ):

            environment_runs[
                item
            ].add(run_index)

    # Majority location.
    location_candidates = [
        (
            location,
            len(runs),
        )
        for location, runs
        in location_runs.items()
        if len(runs) >= support_required
    ]

    if location_candidates:

        max_support = max(
            support
            for _, support
            in location_candidates
        )

        location = sorted(
            value
            for value, support
            in location_candidates
            if support == max_support
        )[0]

    else:

        location = None

    environment = sorted(
        item
        for item, runs
        in environment_runs.items()
        if len(runs) >= support_required
    )

    minority_environment = [
        {
            "value": item,
            "support": len(runs),
            "runs": sorted(runs),
        }
        for item, runs
        in environment_runs.items()
        if len(runs) < support_required
    ]

    scene = {
        "location": location,
        "environment": environment,
    }

    metadata = {
        "location_support": {
            location: len(runs)
            for location, runs
            in location_runs.items()
        },
        "minority_environment": sorted(
            minority_environment,
            key=lambda x: x["value"],
        ),
    }

    return scene, metadata


# ============================================================
# BUILD ONE CONSENSUS CSR
# ============================================================

def build_consensus(
    image_id: str,
    records: List[dict],
    alignment_results: List[dict],
) -> Tuple[dict, dict]:

    number_of_runs = len(records)

    support_required = minimum_support_count(
        number_of_runs
    )

    # --------------------------------------------------------
    # Entity clusters
    # --------------------------------------------------------

    entity_clusters = build_entity_clusters(
        records,
        alignment_results,
    )

    # --------------------------------------------------------
    # Sort entity clusters deterministically
    # --------------------------------------------------------

    entity_clusters.sort(
        key=lambda cluster: (
            entity_type(
                cluster,
                records,
            ),
            str(cluster),
        )
    )

    # --------------------------------------------------------
    # Entity IDs
    # --------------------------------------------------------

    entity_map = {}

    for index, cluster in enumerate(
        entity_clusters,
        start=1,
    ):

        consensus_id = f"e{index}"

        for node in cluster:

            entity_map[node] = consensus_id

    # --------------------------------------------------------
    # Build entities
    # --------------------------------------------------------

    entities = []

    minority_entities = []

    for index, cluster in enumerate(
        entity_clusters,
        start=1,
    ):

        support = len(
            {
                run_index
                for run_index, _
                in cluster
            }
        )

        entity = {
            "id": f"e{index}",
            "type": entity_type(
                cluster,
                records,
            ),
            "attributes": [],
        }

        attributes, attribute_support = (
            entity_attributes(
                cluster,
                records,
                support_required,
            )
        )

        entity["attributes"] = attributes

        provenance = {
            "consensus_id":
                f"e{index}",

            "support":
                support,

            "runs":
                sorted(
                    {
                        run_index
                        for run_index, _
                        in cluster
                    }
                ),

            "type_variants":
                sorted(
                    {
                        get_entity(
                            records[run_index],
                            entity_id,
                        )["type"]
                        for run_index, entity_id
                        in cluster
                        if get_entity(
                            records[run_index],
                            entity_id,
                        )
                    }
                ),

            "attribute_support":
                attribute_support,
        }

        if support >= support_required:

            entities.append(
                entity
            )

        else:

            minority_entities.append(
                provenance
            )

    # --------------------------------------------------------
    # Remove entity map entries belonging to minority
    # entities.
    # --------------------------------------------------------

    valid_entity_ids = {
        entity["id"]
        for entity in entities
    }

    filtered_entity_map = {
        node: consensus_id
        for node, consensus_id
        in entity_map.items()
        if consensus_id in valid_entity_ids
    }

    # --------------------------------------------------------
    # Actions
    # --------------------------------------------------------

    actions, minority_actions = build_actions(
        records,
        filtered_entity_map,
        support_required,
    )

    # --------------------------------------------------------
    # Relations
    # --------------------------------------------------------

    relations, minority_relations = build_relations(
        records,
        filtered_entity_map,
        support_required,
    )

    # --------------------------------------------------------
    # Scene
    # --------------------------------------------------------

    scene, scene_metadata = build_scene(
        records,
        support_required,
    )

    # --------------------------------------------------------
    # Final CSR
    # --------------------------------------------------------

    csr = {
        "image_id": image_id,
        "entities": entities,
        "actions": actions,
        "relations": relations,
        "scene": scene,
    }

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    metadata = {
        "consensus_policy": {
            "number_of_runs":
                number_of_runs,

            "minimum_support":
                support_required,

            "rule":
                "strict majority",
        },

        "minority_entities":
            minority_entities,

        "minority_actions":
            minority_actions,

        "minority_relations":
            minority_relations,

        "scene":
            scene_metadata,
    }

    return csr, metadata


# ============================================================
# VALIDATE FINAL CSR
# ============================================================

def validate_consensus(
    csr: dict,
) -> bool:

    if not isinstance(
        csr.get("image_id"),
        str,
    ):
        return False

    return validate_csr(
        {
            "entities":
                csr["entities"],

            "actions":
                csr["actions"],

            "relations":
                csr["relations"],

            "scene":
                csr["scene"],
        }
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DETERMINISTIC CONSENSUS BUILDER")
    print("=" * 70)

    stability_records = load_json(
        STABILITY_FILE
    )

    alignment_results = load_json(
        ALIGNMENT_FILE
    )

    groups = group_records_by_image(
        stability_records
    )

    print(
        f"Loaded CSR records: "
        f"{len(stability_records)}"
    )

    print(
        f"Loaded alignment results: "
        f"{len(alignment_results)}"
    )

    print(
        f"Images: {len(groups)}"
    )

    consensus_results = []
    metadata_results = []

    for image_id in sorted(groups):

        records = groups[
            image_id
        ]

        print("\n" + "-" * 70)
        print(image_id)
        print("-" * 70)

        print(
            f"Runs: {len(records)}"
        )

        support_required = minimum_support_count(
            len(records)
        )

        print(
            f"Majority threshold: "
            f"{support_required}/{len(records)}"
        )

        csr, metadata = build_consensus(
            image_id,
            records,
            alignment_results,
        )

        if not validate_consensus(
            csr
        ):

            raise ValueError(
                f"Invalid consensus CSR: "
                f"{image_id}"
            )

        print(
            f"Entities: "
            f"{len(csr['entities'])}"
        )

        print(
            f"Actions: "
            f"{len(csr['actions'])}"
        )

        print(
            f"Relations: "
            f"{len(csr['relations'])}"
        )

        print(
            f"Minority entities: "
            f"{len(metadata['minority_entities'])}"
        )

        print(
            f"Minority actions: "
            f"{len(metadata['minority_actions'])}"
        )

        print(
            f"Minority relations: "
            f"{len(metadata['minority_relations'])}"
        )

        consensus_results.append(
            csr
        )

        metadata_results.append(
            {
                "image_id": image_id,
                "metadata": metadata,
            }
        )

    # ========================================================
    # SAVE JSON
    # ========================================================

    # Keep the main CSR file clean.
    save_json(
        OUTPUT_JSON,
        consensus_results,
    )

    # Save provenance separately.
    provenance_file = (
        RESULTS_DIR /
        "consensus_provenance.json"
    )

    save_json(
        provenance_file,
        metadata_results,
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
                "entity_count",
                "action_count",
                "relation_count",
                "scene_location",
                "environment",
            ],
        )

        writer.writeheader()

        for csr in consensus_results:

            writer.writerow(
                {
                    "image_id":
                        csr["image_id"],

                    "entity_count":
                        len(
                            csr["entities"]
                        ),

                    "action_count":
                        len(
                            csr["actions"]
                        ),

                    "relation_count":
                        len(
                            csr["relations"]
                        ),

                    "scene_location":
                        csr["scene"]["location"],

                    "environment":
                        "; ".join(
                            csr["scene"][
                                "environment"
                            ]
                        ),
                }
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("CONSENSUS BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Consensus CSRs: "
        f"{len(consensus_results)}"
    )

    print(
        f"\nMain CSR:"
        f"\n{OUTPUT_JSON}"
    )

    print(
        f"\nProvenance:"
        f"\n{provenance_file}"
    )

    print(
        f"\nCSV:"
        f"\n{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()