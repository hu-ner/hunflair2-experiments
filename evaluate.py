#!/usr/bin/env python3
"""
Evaluate NER+NEN performance given gold and prediction files in PubTator format
"""
import os
import random
from typing import Dict, List

import pandas as pd
from utils import Metric, load_bern, load_pubtator

DISEASE = "disease"
CHEMICAL = "chemical"
SPECIES = "species"
GENE = "gene"


def load_gold(corpora: Dict[str, List]):
    print("-" * 80)
    print("Load gold annotations")
    print("-" * 80)

    gold_dir = os.path.join(os.getcwd(), "annotations", "goldstandard")
    gold = {}
    for corpus, entity_types in corpora.items():
        path = os.path.join(gold_dir, f"{corpus}.txt")
        print(f"Load gold annotations for corpus {corpus}: {path}")
        gold[corpus] = load_pubtator(path, entity_types=entity_types)
    return gold


def load_preds(models: list[str], corpora: Dict[str, List], path: str):
    print("-" * 80)
    print("Load predicted annotations")
    print("-" * 80)

    preds: dict = {}
    for model in models:
        model_dir = os.path.join(path, model)
        preds[model] = {}

        if model == "bern":
            load_fn = load_bern
        else:
            load_fn = load_pubtator

        for corpus, entity_types in corpora.items():
            corpus_path = os.path.join(model_dir, f"{corpus}.txt")

            if not os.path.exists(corpus_path):
                print(f"WARN: Prediction of `{model}` for `{corpus}` not found")
                continue

            print(
                f"Load model `{model}` annotations for corpus {corpus}: {corpus_path}"
            )
            preds[model][corpus] = load_fn(corpus_path, entity_types=entity_types)

    return preds


class Results:
    def __init__(self):
        self.data = {}
        self.types = {}

    def _add(
        self,
        method: str,
        entity_type: str,
        corpus: str,
        key: str,
        class_name: str,
        mention: str,
    ):
        if method not in self.data:
            self.data[method] = {}
            self.types[method] = {}

        if entity_type not in self.data[method]:
            self.data[method][entity_type] = {}
            self.types[method][entity_type] = Metric("eval")

        if corpus not in self.data[method][entity_type]:
            self.data[method][entity_type][corpus] = Metric("eval")
            # self.data[method][entity_type][corpus]["mentions"] = Counter()
            # self.data[method][entity_type][corpus]["entities"] = defaultdict(Counter)
        # self.data[method][entity_type][corpus]["mentions"][key] += 1
        # self.data[method][entity_type][corpus]["entities"][class_name][key] += 1

        if key == "tp":
            self.data[method][entity_type][corpus].add_tp(
                class_name=class_name, mention=mention
            )
            self.types[method][entity_type].add_tp(
                class_name=class_name, mention=mention
            )
        if key == "fp":
            self.data[method][entity_type][corpus].add_fp(
                class_name=class_name, mention=mention
            )
            self.types[method][entity_type].add_fp(
                class_name=class_name, mention=mention
            )
        if key == "fn":
            self.data[method][entity_type][corpus].add_fn(
                class_name=class_name, mention=mention
            )
            self.types[method][entity_type].add_fn(
                class_name=class_name, mention=mention
            )

    def add_tp(
        self, method: str, entity_type: str, corpus: str, class_name: str, mention: str
    ):
        self._add(
            method=method,
            entity_type=entity_type,
            corpus=corpus,
            class_name=class_name,
            mention=mention,
            key="tp",
        )

    def add_fp(
        self, method: str, entity_type: str, corpus: str, class_name: str, mention: str
    ):
        self._add(
            method=method,
            entity_type=entity_type,
            corpus=corpus,
            class_name=class_name,
            mention=mention,
            key="fp",
        )

    def add_fn(
        self, method: str, entity_type: str, corpus: str, class_name: str, mention: str
    ):
        self._add(
            method=method,
            entity_type=entity_type,
            corpus=corpus,
            class_name=class_name,
            mention=mention,
            key="fn",
        )


def multi_label_match(y_pred: set, y_true: set) -> int:
    """
    As gold mentions can have multiple identifiers.
    We use a **relaxed** version of a match,
    i.e. we consider the prediction correct
    if any of the predicted identifier is equal to any of the gold ones.
    """

    return int(any(yp in y_true for yp in y_pred))


class DocumentLevelMetrics:
    def __init__(self):
        self.p = 0
        self.r = 0
        self.t = 0

    def update(self, y_true: set, y_pred: set):
        assert isinstance(
            y_true, set
        ), "Document-level metrics expect a set of identifiers"
        assert isinstance(
            y_pred, set
        ), "Document-level metrics expect a set of identifiers"
        tps = y_true.intersection(y_pred)

        self.p += len(tps) / len(y_pred) if len(y_pred) > 0 else len(y_pred)
        self.r += len(tps) / len(y_true)
        self.t += 1

    def precision(self):
        return self.p / self.t

    def recall(self):
        return self.r / self.t

    def f_score(self):
        micro_p = self.p / self.t
        micro_r = self.r / self.t

        return (
            (2 * micro_p * micro_r / (micro_p + micro_r))
            if (micro_p + micro_r) > 0
            else (micro_p + micro_r)
        )


def get_offests(start, end):
    for s, e in [
        (start, end),
        (start + 1, end),
        (start - 1, end),
        (start, end + 1),
        (start, end - 1),
    ]:
        yield s, e


def evaluate_mention_level(gold: dict, preds: dict):
    results = Results()
    for method, pred in preds.items():
        for corpus, entity_type_pmid in pred.items():
            for entity_type, pmid_entities in entity_type_pmid.items():
                for pmid, pred_offsets in pmid_entities.items():
                    for (start, end), yp in pred_offsets.items():
                        yt = {}
                        for s, e in get_offests(start=start, end=end):
                            try:
                                yt = gold[corpus][entity_type][pmid][(s, e)]
                                break
                            except KeyError:
                                continue

                        if multi_label_match(
                            y_pred=yp["identifiers"],
                            y_true=yt.get("identifiers", set()),
                        ):
                            results.add_tp(
                                method=method,
                                corpus=corpus,
                                entity_type=entity_type,
                                class_name=",".join(
                                    yp["identifiers"],
                                ),
                                mention=yp["text"],
                            )
                        else:
                            results.add_fp(
                                method=method,
                                corpus=corpus,
                                entity_type=entity_type,
                                class_name=",".join(
                                    yp["identifiers"],
                                ),
                                mention=yp["text"],
                            )

    for corpus, entity_type_pmid in gold.items():
        for entity_type, pmid_entities in entity_type_pmid.items():
            for pmid, gold_offsets in pmid_entities.items():
                for (start, end), yt in gold_offsets.items():
                    for method, pred in preds.items():
                        yp = {}
                        for s, e in get_offests(start=start, end=end):
                            try:
                                yp = pred[corpus][entity_type][pmid][(s, e)]
                                break
                            except KeyError:
                                continue

                        if not multi_label_match(
                            y_pred=yp.get("identifiers", set()),
                            y_true=yt["identifiers"],
                        ):
                            results.add_fn(
                                method=method,
                                corpus=corpus,
                                entity_type=entity_type,
                                class_name=",".join(
                                    yt["identifiers"],
                                ),
                                mention=yt["text"],
                            )

    return results


def get_main_result_table(results: Results, only_f1: bool = True):
    out = {}
    for method, data in results.data.items():
        if method not in data:
            out[method] = {}
        for et, subdata in data.items():
            for corpus, metric in subdata.items():
                key = f"{et}-{corpus}"

                if not only_f1:
                    for k in ["P", "R", "F1"]:
                        if k not in out[method]:
                            out[method][k] = {}

                if method == "SciSpacy" and corpus == "tmvar_v3":
                    if not only_f1:
                        out[method]["P"][key] = 0.0
                        out[method]["R"][key] = 0.0
                        out[method]["F1"][key] = 0.0
                    else:
                        out[method][key] = 0.0
                else:
                    if not only_f1:
                        out[method]["P"][key] = metric.precision() * 100
                        out[method]["R"][key] = metric.recall() * 100
                        out[method]["F1"][key] = metric.f_score() * 100
                    else:
                        try:
                            out[method][key] = metric.f_score()
                        except ZeroDivisionError:
                            out[method][key] = 0.0
    if not only_f1:
        out = {k: pd.DataFrame(v) for k, v in out.items()}

        for k, df in out.items():
            df.loc["Avg. All "] = df.mean(numeric_only=True)
            df.columns = pd.MultiIndex.from_product([[k], df.columns])

        df = pd.concat(list(out.values()), axis=1)
    else:
        df = pd.DataFrame(out)
        df.loc["Avg"] = df.mean(numeric_only=True)
        df = df.apply(lambda x: x * 100)

    return df


def get_micro_macro_difference_table(results: Results):
    out = {}
    for method, entity_metric in results.data.items():
        for et, corpus_metric in entity_metric.items():
            for corpus, metric in corpus_metric.items():
                key = f"{et}-{corpus}"

                if method not in out:
                    out[method] = {}

                for k in ["M-F1", "diff"]:
                    if k not in out[method]:
                        out[method][k] = {}

                if method == "SciSpacy" and corpus == "tmvar_v3":
                    micro_f1 = 0.0
                    macro_f1 = 0.0
                else:
                    micro_f1 = round(metric.micro_avg_f_score() * 100, 2)
                    macro_f1 = round(metric.entity_macro_avg_f_score() * 100, 2)

                out[method]["M-F1"][key] = macro_f1
                out[method]["diff"][key] = macro_f1 - micro_f1

    out = {k: pd.DataFrame(v) for k, v in out.items()}

    for k, df in out.items():
        df.columns = pd.MultiIndex.from_product([[k], df.columns])

    df = pd.concat(list(out.values()), axis=1)
    df.loc["Avg"] = df.mean(numeric_only=True)

    return df


def evaluate_document_level(gold: dict, preds: dict):
    results = {}

    for corpus, entity_type_pmid in gold.items():
        for entity_type, pmid_entities in entity_type_pmid.items():
            for pmid, gold_offsets in pmid_entities.items():
                y_true = set(
                    i for offset, yt in gold_offsets.items() for i in yt["identifiers"]
                )
                for method, pred in preds.items():
                    if method not in results:
                        results[method] = {}

                    if entity_type not in results[method]:
                        results[method][entity_type] = {}

                    if corpus not in results[method][entity_type]:
                        results[method][entity_type][corpus] = DocumentLevelMetrics()

                    if corpus not in pred:
                        continue

                    try:
                        pred_offsets = pred[corpus][entity_type][pmid]
                    except KeyError:
                        pred_offsets = {}

                    y_pred = set(
                        i
                        for offset, yp in pred_offsets.items()
                        for i in yp["identifiers"]
                    )

                    results[method][entity_type][corpus].update(
                        y_pred=y_pred, y_true=y_true
                    )

    return results


def get_document_level_table(results: dict):
    out = {}

    for method, data in results.items():
        if method not in data:
            out[method] = {}

        for et, subdata in data.items():
            for corpus, metric in subdata.items():
                key = f"{et}-{corpus}"

                # for k in ["P", "R", "F1"]:
                #     if k not in out[method]:
                #         out[method][k] = {}

                try:
                    # p = metric.precision()
                    # r = metric.recall()
                    f1 = metric.f_score()
                except ZeroDivisionError:
                    f1 = 0
                    # p, r, f1 = 0, 0, 0

                out[method][key] = f1

                # out[method]["P"][key] = p
                # out[method]["R"][key] = r
                # out[method]["F1"][key] = f1

    df = pd.DataFrame(out)
    df.loc["Avg"] = df.mean(numeric_only=True)
    df = df.apply(lambda x: round(x * 100, 2))

    # out = {k: pd.DataFrame(v) for k, v in out.items()}
    #
    # for k, df in out.items():
    #     # df.loc["Avg. Disease"] = df.loc[["disease-craft", "disease-medmentions"]].mean(
    #     #     numeric_only=True
    #     # )
    #     # df.loc["Avg. Species"] = df.loc[["species-craft", "species-bioid"]].mean(
    #     #     numeric_only=True
    #     # )
    #     df.loc["Avg. All "] = df.mean(numeric_only=True)
    #     df.columns = pd.MultiIndex.from_product([[k], df.columns])
    #
    # df = pd.concat(list(out.values()), axis=1)

    return df


def main():
    """
    Script
    """
    random.seed(42)

    print("*" * 80)
    print("Evaluate cross-corpus joint NER and NEN")
    print("*" * 80)

    CORPORA = {
        "medmentions": [DISEASE, CHEMICAL],
        "tmvar_v3": [GENE],
        "bioid": [SPECIES],
    }

    METHODS = {
        "bern": "BERN2",
        "pubtator": "PubTator",
        "scispacy": "SciSpacy",
        "bent": "bent",
        "hunflair2": "HunFlair2",
    }

    gold = load_gold(corpora=CORPORA)

    preds = load_preds(
        models=list(METHODS),
        corpora=CORPORA,
        path=os.path.join(os.getcwd(), "annotations"),
    )

    ml_results = evaluate_mention_level(gold=gold, preds=preds)
    ml_df = get_main_result_table(ml_results)
    ml_df.rename(METHODS, inplace=True, axis=1)
    ml_df = ml_df.round(2)
    print(ml_df)


if __name__ == "__main__":
    main()
