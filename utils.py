#!/usr/bin/evim python3
"""
Utilities to load data
"""

import re
from collections import OrderedDict, defaultdict
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, cast

from bioc import pubtator
from datasets import IterableDatasetDict, load_dataset

DEFAULT_TYPE_MAPPING = {
    "chemical": "chemical",
    "['chemical']": "chemical",
    "simple_chemical": "chemical",
    "cancer": "disease",
    "disease": "disease",
    "['disease']": "disease",
    "gene": "gene",
    "['gene']": "gene",
    "gene_or_gene_product": "gene",
    "species": "species",
    "['species']": "species",
    "cellline": "cell_line",
    "cell_line": "cell_line",
    "protein": "gene",
    # "simple_chemical": "chemical",  # BioNLP ST 2013 CG
    "amino_acid": "chemical",  # BioNLP ST 2013 CG
    # "cancer": "disease",  # BioNLP ST 2013 CG
    # "gene_or_gene_product": "gene",  # BioNLP ST 2013 CG
    "organism": "species",  # BioNLP ST 2013 CG
    "pathological_formation": "disease",  # BioNLP ST 2013 CG
    # "gene": "gene",  # NLM Gene
    "generif": "gene",  # NLM Gene
    "stargene": "gene",  # NLM Gene
    "domain": "gene",  # NLM Gene
    "other": "gene",  # NLM Gene
    # "chemical": "chemical",  # NLM Chem
    "diseaseclass": "disease",  # NCBI Disease
    "specificdisease": "disease",  # NCBI Disease
    "modifier": "disease",  # NCBI Disease
    "geneprotein": "gene",  # Cell Finder
    # "cellline": "cell_line",  # Cell Finder
    # "species": "species",  # Cell Finder
    "geneorgeneproduct": "gene",  # BioRED
    "chemicalentity": "chemical",  # BioRED
    "organismtaxon": "species",  # BioRED
    "diseaseorphenotypicfeature": "disease",  # BioRED
    "pr": "gene",  # CRAFT (local)
    "chebi": "chemical",  # CRAFT (local)
    "ncbitaxon": "species",  # CRAFT (local)
    # "protein": "gene",  # BioID
    "mondo": "disease",  # CRAFT (local)
    "drug": "chemical",  # BERNv2
    # https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemGroups_2018.txt
    "t020": "disease",  # UMLS
    "t190": "disease",  # UMLS
    "t049": "disease",  # UMLS
    "t019": "disease",  # UMLS
    "t033": "disease",  # UMLS
    "t047": "disease",  # UMLS
    "t050": "disease",  # UMLS
    "t037": "disease",  # UMLS
    "t048": "disease",  # UMLS
    "t191": "disease",  # UMLS
    "t046": "disease",  # UMLS
    "t184": "disease",  # UMLS
    "t103": "chemical",  # UMLS
    "t120": "chemical",  # UMLS
    "t104": "chemical",  # UMLS
    "t109": "chemical",  # UMLS
    "t197": "chemical",  # UMLS
    "t116": "chemical",  # UMLS
    "t195": "chemical",  # UMLS
    "t123": "chemical",  # UMLS
    "t122": "chemical",  # UMLS
    "t200": "chemical",  # UMLS
    "t196": "chemical",  # UMLS
    "t126": "chemical",  # UMLS
    "t131": "chemical",  # UMLS
    "t125": "chemical",  # UMLS
    "t129": "chemical",  # UMLS
    "t130": "chemical",  # UMLS
    "t114": "chemical",  # UMLS
    "t121": "chemical",  # UMLS
    "t192": "chemical",  # UMLS
    "t127": "chemical",  # UMLS
}


def clean_identifiers(ids: str) -> list:
    """
    Homogenize identifiers
    """

    return (
        ids.replace("NCBI Gene:", "")
        .replace("NCBIGene:", "")
        .replace("NCBI taxon:", "")
        .replace("NCBITaxon:", "")
        .upper()
        .replace(",", ";")
        .split(";")
    )


def load_pubtator(path: str, entity_types: list[str]) -> dict:
    """
    Load annotations from PubTator into nested dict:
        - pmid:
            - entity_type:
                - offset:
                    - identifiers
    """

    annotations: dict = {}

    with open(path) as fp:
        documents = pubtator.load(fp)
        for d in documents:
            for a in d.annotations:
                if a.type in ["NCBITaxon", "organism"]:
                    a.type = "species"

                entity_type = a.type.lower()

                if entity_type not in entity_types:
                    continue

                if entity_type not in annotations:
                    annotations[entity_type] = {}

                if a.pmid not in annotations[entity_type]:
                    annotations[entity_type][a.pmid] = {}

                identifiers = clean_identifiers(a.id)

                annotations[entity_type][a.pmid][(a.start, a.end)] = {
                    "identifiers": identifiers,
                    "text": a.text,
                }

    return annotations


def load_bern(path: str, entity_types: list[str]) -> dict:
    """
    Load annotations from BERN into nested dict:
        - pmid:
            - entity_type:
                - offset:
                    - identifiers
    """

    annotations: dict = {}

    with open(path) as fp:
        for line in fp:
            line = line.strip()

            if line == "":
                continue

            elements = line.split("\t")

            if len(elements) == 6:
                pmid, start, end, text, entity_type, ids = elements
            elif len(elements) == 5:
                pmid, start, end, text, entity_type = elements
                ids = "-1"

            if entity_type not in entity_types:
                continue

            if entity_type not in annotations:
                annotations[entity_type] = {}

            if pmid not in annotations[entity_type]:
                annotations[entity_type][pmid] = {}

            identifiers = clean_identifiers(ids)

            annotations[entity_type][pmid][(int(start), int(end))] = {
                "identifiers": identifiers,
                "text": text,
            }

    return annotations


class Document:
    def __init__(self, id: int, title: str, abstract: str, annotations: List[Tuple]):
        self.id = id
        self.title = title
        self.abstract = abstract
        self.annotations = annotations

        self.text = title if title is not None else ""
        if abstract is not None:
            self.text = self.text + " " + abstract if title is not None else abstract


def get_documents_from_bigbio(
    data_set: str, schema: str = "bigbio_kb", exclude_splits=None, download_mode=None
) -> Dict[int, Document]:
    if exclude_splits is None:
        exclude_splits = []

    print(f"Loading data set {data_set} from BigBio")

    # Handle special cases
    if data_set == "medmentions":
        schema = "st21pv_" + schema  # MedMentions uses a different schema

    data = cast(
        IterableDatasetDict,
        load_dataset(
            path=f"bigbio/{data_set}",
            name=f"{data_set}_{schema}",
            download_mode=download_mode,
        ),
    )

    documents = OrderedDict()
    doc_id = 0

    for split in sorted(data.keys()):
        if split in exclude_splits:
            print(f"Excluding split {split}")
            continue

        for doc in data[split]:
            if "text" in doc:
                # Special case PDR source annotations
                text = doc["text"].replace("\n", " ").strip()
                title = text
                abstract = None
            else:
                if len(doc["passages"]) == 1:
                    text = doc["passages"][0]["text"][0].replace("\n", " ").strip()
                    title = text
                    abstract = None

                elif len(doc["passages"]) == 2:
                    title = None
                    abstract = None
                    for passage in doc["passages"]:
                        if passage["type"] == "title":
                            title = passage["text"][0].replace("\n", " ").strip()
                        elif passage["type"] == "abstract":
                            abstract = passage["text"][0].replace("\n", " ").strip()
                        else:
                            raise AssertionError()
                else:
                    # Used for nlmchem dataset
                    title = None
                    abstract = None
                    fulltext = ""
                    for i, passage in enumerate(doc["passages"]):
                        if i == 0:
                            fulltext += passage["text"][0]
                        else:
                            fulltext += " " + passage["text"][0]
                    abstract = fulltext.strip().replace("\n", " ")
                    # TODO: Check this case for nlmchem dataset where assert is raised
                    # raise AssertionError()

            doc_text = title
            doc_text += " " + abstract if abstract is not None else ""

            entities = []
            for entity in doc["entities"]:
                # We ignore non-consecutive entities!
                if len(entity["offsets"]) > 1:
                    continue

                start, end = entity["offsets"][0]
                mention = entity["text"][0]
                type = entity["type"]
                db_ids = ",".join(
                    [
                        entry["db_name"] + ":" + str(entry["db_id"])
                        for entry in entity["normalized"]
                    ]
                )

                if mention != doc_text[start:end].strip():
                    print(
                        f"Offset mismatch in {doc_id}: |{mention}| vs. |{doc_text[start:end]}|"
                    )

                entities.append((start, end, mention, type, db_ids))

            documents[doc_id] = Document(doc_id, title, abstract, entities)

            doc_id += 1

    return documents


def read_documents_from_pubtator(
    input_file: Path, type_mapping: Dict[str, str] = DEFAULT_TYPE_MAPPING
) -> List[Document]:
    id_to_annotations = defaultdict(list)
    id_to_title = {}
    id_to_abstract = {}

    with input_file.open("r", encoding="utf8") as input_text:
        for line in input_text:
            line = line.rstrip()
            p_title = re.compile("^([0-9]+)\|t\|(.*)$")
            p_abstract = re.compile("^([0-9]+)\|a\|(.*)$")
            p_annotation = re.compile(
                "^([0-9]+)	([0-9]+)	([0-9]+)	([^\t]+)	([^\t]+)(.*)"
            )

            if p_title.search(line):  # title
                m = p_title.match(line)
                pmid = int(m.group(1))
                id_to_title[pmid] = m.group(2)

            elif p_abstract.search(line):  # abstract
                m = p_abstract.match(line)
                pmid = int(m.group(1))
                abstract = m.group(2).strip()
                if len(abstract) > 0:
                    id_to_abstract[pmid] = abstract

            elif p_annotation.search(line):  # annotation
                m = p_annotation.match(line)
                pmid = int(m.group(1))
                start = int(m.group(2))
                last = int(m.group(3))
                mention = m.group(4)
                type = m.group(5).lower()
                id = m.group(6).strip()

                if type in type_mapping:
                    type = type_mapping[type]
                    id_to_annotations[pmid].append(
                        (pmid, start, last, mention.strip(), type, id)
                    )

    all_pmids = sorted(
        set(
            list(id_to_title.keys())
            + list(id_to_abstract.keys())
            + list(id_to_annotations.keys())
        )
    )

    documents = [
        Document(
            id=pmid,
            title=id_to_title[pmid] if pmid in id_to_title else None,
            abstract=id_to_abstract[pmid] if pmid in id_to_abstract else None,
            annotations=id_to_annotations[pmid],
        )
        for pmid in all_pmids
    ]

    return documents


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
        # if self.precision(class_name, mention) + self.recall(class_name, mention) > 0:
        return (
            (1 + self.beta * self.beta)
            * (self.precision(class_name, mention) * self.recall(class_name, mention))
            / (
                self.precision(class_name, mention) * self.beta * self.beta
                + self.recall(class_name, mention)
            )
        )
        # return 0.0

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
