from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Callable, List

from evaluation_utils import evaluate, match_lenient, match_standard

# Tool name constants
TOOL_BERN_V2 = "BERNv2"
TOOL_HUNFLAIR2 = "HunFlair2"
TOOL_PUBTATOR = "PubTator"
TOOL_SCISPACY = "SciSpacy"
TOOL_BENT = "BENT"


# Dataset name constants
class Corpora(Enum):
    BIOID = "BioID"
    MEDMENTIONS_CTD = "MedMentions CTD"
    TMVAR_V3 = "tmVar (v3)"


# Map to tool to their existing models / annotations directories
TOOL_TO_MODELS = {
    TOOL_BERN_V2: ["bern"],
    TOOL_HUNFLAIR2: ["hunflair2"],
    TOOL_PUBTATOR: ["pubtator"],
    TOOL_SCISPACY: [
        "scispacy",
        "scispacy_en_ner_bionlp13cg_md",
        "scispacy_en_ner_craft_md",
        "scispacy_en_ner_jnlpba_md"
    ],
    TOOL_BENT: ["bent"],
}

# Map dataset names to their file names
CORPORA_TO_FILEPREFIX = {
    Corpora.BIOID: "bioid",
    Corpora.MEDMENTIONS_CTD: "medmentions",
    Corpora.TMVAR_V3: "tmvar_v3",
}

ENTITY_TYPES_TO_CORPORA = {
    "Chemical": [Corpora.MEDMENTIONS_CTD],
    "Disease": [Corpora.MEDMENTIONS_CTD],
    "Gene": [Corpora.TMVAR_V3],
    "Species": [Corpora.BIOID],
}

CORPORA_TO_ENTITY_TYPES = {
    Corpora.BIOID: ["Species"],
    Corpora.MEDMENTIONS_CTD: ["Chemical", "Disease"],
    Corpora.TMVAR_V3: ["Gene"],
}


def compute_result_table(
    annotations_dir: Path,
    tools: List[str],
    matching_func: Callable,
    entity_types: List[str] = ["Chemical", "Disease", "Gene", "Species"],
    ignore_normalization: bool = True,
    macro_average_over_mentions: bool = False,
):
    model_to_dataset_to_result = defaultdict(dict)

    # Run evaluation for each tool (resp. each model) on each data set
    for tool in tools:
        print(f"\tEvaluating {tool}")

        for model in TOOL_TO_MODELS[tool]:
            print(f"\t\tModel: {model}")
            model_ann_dir = annotations_dir / model

            for dataset, file_prefix in CORPORA_TO_FILEPREFIX.items():
                gold_file = annotations_dir / "goldstandard" / f"{file_prefix}.txt"
                if dataset == Corpora.MEDMENTIONS_CTD:
                    gold_file = annotations_dir / "goldstandard" / "medmentions_ctd.txt"

                pred_file = model_ann_dir / f"{file_prefix}.txt"
                if not pred_file.exists():
                    continue

                eval_result = evaluate(
                    gold_file=gold_file,
                    pred_file=pred_file,
                    match_func=matching_func,
                    ignore_normalization_ids=ignore_normalization,
                )

                model_to_dataset_to_result[model][dataset] = eval_result

    # Start printing results table
    print("\n")
    print("\t".join(["Entity type / dataset"] + tools + ["Support"]))

    # Collects results per entity type and tool for computing averages later
    entity_type_to_tool_to_results = defaultdict(lambda: defaultdict(list))
    tool_to_results = defaultdict(list)
    entity_type_to_dataset_to_support = defaultdict(lambda: defaultdict(int))

    # For each entity type ...
    for entity_type in entity_types:
        print(entity_type)

        # ... and each dataset - print results
        for dataset in ENTITY_TYPES_TO_CORPORA[entity_type]:
            all_results = []
            support = -1

            for tool in tools:
                tool_results = []

                for i, model in enumerate(TOOL_TO_MODELS[tool]):
                    if ((model not in model_to_dataset_to_result) or
                            (dataset not in model_to_dataset_to_result[model])):
                        continue

                    result = model_to_dataset_to_result[model][dataset]
                    if macro_average_over_mentions:
                        f1_score = result.mention_macro_avg_f_score(entity_type.lower())
                    else:
                        f1_score = result.f_score(entity_type.lower())

                    tool_results.append(f1_score * 100)

                    if i == len(TOOL_TO_MODELS[tool]) - 1:
                        if macro_average_over_mentions:
                            support = len(
                                result.entity_type_to_mention_count[
                                    entity_type.lower()
                                ].keys()
                            )
                        else:
                            support = result.entity_type_count[entity_type.lower()]

                if len(tool_results) > 0:
                    best_score = max(tool_results)
                    all_results.append(f"{best_score:.2f}")

                    entity_type_to_tool_to_results[entity_type][tool].append(best_score)
                    tool_to_results[tool].append(best_score)

            entity_type_to_dataset_to_support[entity_type][dataset] = support

            lines_values = [dataset.value] + all_results + [f"{support}"]
            print("\t".join(lines_values))

    print("-"*75)
    # Print total average results per tool
    total_avgs = [
        sum(tool_to_results[tool]) / len(tool_to_results[tool]) for tool in tools
    ]
    total_avgs = ["Avg. All"] + [f"{result:.2f}" for result in total_avgs]
    total_avgs.append(
        f"""{sum(
            [
                sum(entity_type_to_dataset_to_support[entity_type].values())
                for entity_type in entity_types
            ]
        )}"""
    )
    print("\t".join(total_avgs) + "\n")


if __name__ == "__main__":
    print("Standard NER evaluation")
    compute_result_table(
        annotations_dir=Path("annotations"),
        tools=[
            TOOL_BERN_V2,
            TOOL_HUNFLAIR2,
            TOOL_PUBTATOR,
            TOOL_SCISPACY,
            TOOL_BENT,
        ],
        matching_func=match_standard(1),
        macro_average_over_mentions=False,
    )

    print("Lenient NER evaluation")
    compute_result_table(
        annotations_dir=Path("annotations"),
        tools=[
            TOOL_BERN_V2,
            TOOL_HUNFLAIR2,
            TOOL_PUBTATOR,
            TOOL_SCISPACY,
            TOOL_BENT,
        ],
        matching_func=match_lenient(1),
        macro_average_over_mentions=False,
    )
