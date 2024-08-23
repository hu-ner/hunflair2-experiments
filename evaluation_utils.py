import itertools
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from utils import read_documents_from_pubtator


# Copied from older Flair version
class Metric:
    def __init__(self, name, beta=1):
        self.name = name
        self.beta = beta

        self._tps = defaultdict(lambda: defaultdict(int))
        self._fps = defaultdict(lambda: defaultdict(int))
        self._tns = defaultdict(lambda: defaultdict(int))
        self._fns = defaultdict(lambda: defaultdict(int))

        self.entity_type_to_mention_count: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.entity_type_count: Dict[str, int] = defaultdict(int)

    def add_tp(self, class_name, mention=None):
        self._tps[class_name][mention] += 1

    def add_tn(self, class_name, mention=None):
        self._tns[class_name][mention] += 1

    def add_fp(self, class_name, mention=None):
        self._fps[class_name][mention] += 1

    def add_fn(self, class_name, mention=None):
        self._fns[class_name][mention] += 1

    def get_ftnp(self, ftnp, class_name=None, mention=None):
        if class_name is None and mention is None:
            return sum(
                [
                    ftnp[class_name][mention]
                    for class_name in self.get_classes()
                    for mention in self.get_mentions(class_name)
                ]
            )
        elif mention is None:
            return sum([ftnp[class_name][mention] for mention in ftnp[class_name]])
        else:
            return ftnp[class_name][mention]

    def get_tp(self, class_name=None, mention=None):
        return self.get_ftnp(self._tps, class_name, mention)

    def get_tn(self, class_name=None, mention=None):
        return self.get_ftnp(self._tns, class_name, mention)

    def get_fp(self, class_name=None, mention=None):
        return self.get_ftnp(self._fps, class_name, mention)

    def get_fn(self, class_name=None, mention=None):
        return self.get_ftnp(self._fns, class_name, mention)

    def precision(self, class_name=None, mention=None):
        if self.get_tp(class_name, mention) + self.get_fp(class_name, mention) > 0:
            return self.get_tp(class_name, mention) / (
                self.get_tp(class_name, mention) + self.get_fp(class_name, mention)
            )
        return 0.0

    def recall(self, class_name=None, mention=None):
        if self.get_tp(class_name, mention) + self.get_fn(class_name, mention) > 0:
            return self.get_tp(class_name, mention) / (
                self.get_tp(class_name, mention) + self.get_fn(class_name, mention)
            )
        return 0.0

    def f_score(self, class_name=None, mention=None):
        if self.precision(class_name, mention) + self.recall(class_name, mention) > 0:
            return (
                (1 + self.beta * self.beta)
                * (
                    self.precision(class_name, mention)
                    * self.recall(class_name, mention)
                )
                / (
                    self.precision(class_name, mention) * self.beta * self.beta
                    + self.recall(class_name, mention)
                )
            )
        return 0.0

    def accuracy(self, class_name=None, mention=None):
        if (
            self.get_tp(class_name, mention)
            + self.get_fp(class_name, mention)
            + self.get_fn(class_name, mention)
            + self.get_tn(class_name, mention)
            > 0
        ):
            return (
                self.get_tp(class_name, mention) + self.get_tn(class_name, mention)
            ) / (
                self.get_tp(class_name, mention)
                + self.get_fp(class_name, mention)
                + self.get_fn(class_name, mention)
                + self.get_tn(class_name, mention)
            )
        return 0.0

    def mention_macro_avg_f_score(self, class_name=None):
        class_precisions = [
            self.precision(class_name, mention)
            for mention in self.get_mentions(class_name)
        ]
        class_recalls = [
            self.recall(class_name, mention)
            for mention in self.get_mentions(class_name)
        ]
        macro_precision = sum(class_precisions) / len(class_precisions)
        macro_recall = sum(class_recalls) / len(class_recalls)
        # print(macro_precision, macro_recall)
        # print(class_precisions, class_recalls)
        if (
            len(class_precisions) == 0
            or len(class_recalls) == 0
            or (macro_precision + macro_recall) == 0
        ):
            return 0.0
        macro_f_score = (
            (1 + self.beta * self.beta)
            * (macro_precision * macro_recall)
            / (macro_precision * self.beta * self.beta + macro_recall)
        )
        return macro_f_score

    def entity_macro_avg_f_score(self):
        class_precisions = [self.precision(c) for c in self.get_classes()]
        class_recalls = [self.recall(c) for c in self.get_classes()]

        macro_precision = sum(class_precisions) / len(class_precisions)
        macro_recall = sum(class_recalls) / len(class_recalls)
        # print(macro_precision, macro_recall)
        # print(class_precisions, class_recalls)
        if (
            len(class_precisions) == 0
            or len(class_recalls) == 0
            or (macro_precision + macro_recall) == 0
        ):
            return 0.0
        macro_f_score = (
            (1 + self.beta * self.beta)
            * (macro_precision * macro_recall)
            / (macro_precision * self.beta * self.beta + macro_recall)
        )
        return macro_f_score

    def micro_avg_f_score(self):
        return self.f_score(None)

    def macro_avg_f_score(self):
        class_f_scores = [self.f_score(class_name) for class_name in self.get_classes()]
        if len(class_f_scores) == 0:
            return 0.0
        macro_f_score = sum(class_f_scores) / len(class_f_scores)
        return macro_f_score

    def micro_avg_accuracy(self):
        return self.accuracy(None)

    def macro_avg_accuracy(self):
        class_accuracy = [
            self.accuracy(class_name) for class_name in self.get_classes()
        ]

        if len(class_accuracy) > 0:
            return sum(class_accuracy) / len(class_accuracy)

        return 0.0

    def get_classes(self) -> List:
        all_classes = set(
            itertools.chain(
                *[
                    list(keys)
                    for keys in [
                        self._tps.keys(),
                        self._fps.keys(),
                        self._tns.keys(),
                        self._fns.keys(),
                    ]
                ]
            )
        )
        all_classes = [
            class_name for class_name in all_classes if class_name is not None
        ]
        all_classes.sort()
        return all_classes

    def get_mentions(self, class_name) -> List:
        all_mentions = set(
            list(self._tps[class_name].keys())
            + list(self._fps[class_name].keys())
            + list(self._tns[class_name].keys())
            + list(self._fns[class_name].keys())
        )
        all_mentions = [mention for mention in all_mentions if mention is not None]
        all_mentions.sort()
        return all_mentions

    def __str__(self):
        all_classes = self.get_classes()
        all_classes = [None] + all_classes
        all_lines = [
            "{0:<10}\tsup: {1} - tp: {2} - fp: {3} - fn: {4} - tn: {5} - precision: {6:.4f} - recall: {7:.4f} - accuracy: {8:.4f} - f1-score: {9:.4f}".format(
                self.name if class_name is None else class_name,
                self.get_tp(class_name) + self.get_fn(class_name),
                self.get_tp(class_name),
                self.get_fp(class_name),
                self.get_fn(class_name),
                self.get_tn(class_name),
                self.precision(class_name),
                self.recall(class_name),
                self.accuracy(class_name),
                self.f_score(class_name),
            )
            for class_name in all_classes
        ]
        return "\n".join(all_lines)


def copy_dict(
    dictionary: Dict[int, List[Tuple]], ignore_normalization_ids: bool = False
) -> Dict[int, List[Tuple]]:
    copy = {}

    for key, values in dictionary.items():
        if ignore_normalization_ids:
            value_copy = []
            for value in values:
                value_copy.append(
                    tuple([v for i, v in enumerate(value) if i < len(value) - 1])
                )
        else:
            value_copy = [v for v in values]
        copy[key] = value_copy

    return copy


def evaluate(
    gold_file: Path,
    pred_file: Path,
    match_func: Callable[[Tuple, List], Optional[Tuple]],
    ignore_normalization_ids: bool = False,
) -> Metric:
    gold_documents = read_documents_from_pubtator(gold_file)
    gold_annotations = {doc.id: doc.annotations for doc in gold_documents}

    pred_documents = read_documents_from_pubtator(pred_file)
    pred_annotations = {doc.id: doc.annotations for doc in pred_documents}

    metric = Metric("Evaluation", beta=1)

    entity_types = ["cell_line", "chemical", "disease", "gene", "species"]
    for doc in gold_documents:
        for entity in doc.annotations:
            assert entity[4] in entity_types
            if entity[-1] == "NIL":
                continue
            metric.entity_type_to_mention_count[entity[4]][entity[3]] += 1
            metric.entity_type_count[entity[4]] += 1

    copy_gold = copy_dict(gold_annotations, ignore_normalization_ids)
    for document_id, annotations in pred_annotations.items():
        for pred_entry in annotations:
            if ignore_normalization_ids:
                copy_pred_entry = tuple(
                    [v for i, v in enumerate(pred_entry) if i < len(pred_entry) - 1]
                )
            # Documents may not contain any gold entity!
            if document_id in copy_gold:
                matched_gold = match_func(copy_pred_entry, copy_gold[document_id])
            else:
                matched_gold = None

            # Ignore predictions mapping gold NIL values
            matched_gold_nil = match_func(pred_entry, gold_annotations[document_id])
            if matched_gold_nil and matched_gold_nil[-1] == "NIL":
                continue

            if matched_gold:
                copy_gold[document_id].remove(matched_gold)
                metric.add_tp(pred_entry[4], pred_entry[3])
            else:
                metric.add_fp(pred_entry[4], pred_entry[3])

    copy_pred = copy_dict(pred_annotations, ignore_normalization_ids)

    for document_id, annotations in gold_annotations.items():
        for gold_entry in annotations:
            # Ignore NIL values
            if gold_entry[-1] == "NIL":
                continue
            if ignore_normalization_ids:
                gold_entry = tuple(
                    [v for i, v in enumerate(gold_entry) if i < len(gold_entry) - 1]
                )
            if document_id in copy_pred:
                matched_pred = match_func(gold_entry, copy_pred[document_id])
            else:
                matched_pred = None

            if not matched_pred:
                metric.add_fn(gold_entry[4], gold_entry[3])
            else:
                copy_pred[document_id].remove(matched_pred)

    return metric


def match_lenient(threshold: int):
    def _partial_match(entry, candidates):
        for c in candidates:
            if (
                entry[0] == c[0]  # same document?
                and entry[4] == c[4]  # same entity type?
                and (
                    # Substrings
                    (
                        entry[1] >= (c[1] - threshold) and entry[2] <= c[2]
                    )  # Start shifted by <threshold>
                    or (
                        entry[1] >= c[1] and entry[2] <= (c[2] + threshold)
                    )  # End shifted by <threshold>
                    # Superstrings
                    or (
                        entry[1] <= (c[1] + threshold) and entry[2] >= c[2]
                    )  # Start shifted by <threshold>
                    or (
                        entry[1] <= c[1] and entry[2] >= (c[2] - threshold)
                    )  # End shifted by <threshold>
                )
                and (len(entry) < 6 or entry[5] == c[5])  # same normalization id?
            ):
                return c

    return _partial_match


def match_standard(threshold: int):
    def _partial_match(entry, candidates):
        for c in candidates:
            if (
                entry[0] == c[0]  # same document?
                and entry[4] == c[4]  # same entity type?
                and (
                    (
                        abs(entry[1] - c[1]) <= threshold
                    )  # Start offset difference within threshold
                    and (
                        abs(entry[2] - c[2]) <= threshold
                    )  # End offset difference within threshold
                )
                and (len(entry) < 6 or entry[5] == c[5])  # same normalization id?
            ):
                return c

    return _partial_match
