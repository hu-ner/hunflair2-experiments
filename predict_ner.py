import argparse
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, cast

from datasets import IterableDatasetDict, load_dataset
from flair.models import SequenceTagger
from flair.models.prefixed_tagger import (
    PrefixedSequenceTagger,
)
from flair.nn import Classifier
from flair.splitter import SciSpacySentenceSplitter
from tqdm import tqdm


class Predictor(ABC):
    def __init__(self, data_set: str, output_file: Path, test_only: bool = False):
        self.documents: Dict[int, Dict[str, str]] = {}
        self.annotations: Dict[str, List] = defaultdict(list)
        self.data_set = data_set
        self.output_file = output_file
        self.test_only = test_only
        self._read_documents()
        self._tag_documents()
        self._write_annotations()

    def _read_documents(self):
        # Handle special cases
        if self.data_set == "medmentions":
            schema = f"{self.data_set}_st21pv_bigbio_kb"  # MedMentions uses a different schema
        else:
            schema = f"{self.data_set}_bigbio_kb"

        data = cast(
            IterableDatasetDict,
            load_dataset(f"bigbio/{self.data_set}", name=schema),
        )

        doc_id = 0
        for split in sorted(data.keys()):
            if self.test_only and split != "test":
                continue

            print(f"Reading documents for {split}")
            for doc in data[split]:
                self.documents[doc_id] = {}
                if len(doc["passages"]) == 1:
                    self.documents[doc_id]["title"] = (
                        doc["passages"][0]["text"][0].replace("\n", " ").strip()
                    )
                elif len(doc["passages"]) == 2:
                    for passage in doc["passages"]:
                        if passage["type"] == "title":
                            self.documents[doc_id]["title"] = (
                                passage["text"][0].replace("\n", " ").strip()
                            )
                        elif passage["type"] == "abstract":
                            self.documents[doc_id]["abstract"] = (
                                passage["text"][0].replace("\n", " ").strip()
                            )
                        else:
                            raise AssertionError()
                else:
                    # Used for nlmchem dataset
                    for passage in doc["passages"]:
                        if "fulltext" not in self.documents[doc_id]:
                            self.documents[doc_id]["fulltext"] = (
                                passage["text"][0].replace("\n", " ").strip()
                            )
                        else:
                            self.documents[doc_id]["fulltext"] += (
                                " " + passage["text"][0].replace("\n", " ").strip()
                            )
                    # TODO: Check this case for nlmchem dataset where assert is raised
                    # raise AssertionError()
                doc_id += 1

    @abstractmethod
    def _tag_documents(self):
        ...

    def _write_annotations(self):
        """Writes annotations to a file in the PubTator format
        required by the evaluation script."""
        with open(str(self.output_file), "w") as writer:
            print(f"Writing annotations to file {str(self.output_file)}")
            doc_id = 0
            for document_id in tqdm(self.documents.keys(), total=len(self.documents)):
                if "title" in self.documents[document_id]:
                    title = self.documents[document_id]["title"]
                    writer.write(f"{doc_id}|t|{title}\n")
                if "abstract" in self.documents[document_id]:
                    abstract = (
                        self.documents[document_id]["abstract"]
                        .replace("\n", " ")
                        .strip()
                    )
                    writer.write(f"{document_id}|a|{abstract}\n")
                if "fulltext" in self.documents[document_id]:
                    fulltext = (
                        self.documents[document_id]["fulltext"]
                        .replace("\n", " ")
                        .strip()
                    )
                    writer.write(f"{document_id}|f|{fulltext}\n")
                for entity in self.annotations[document_id]:
                    start = entity[0]
                    end = entity[1]
                    mention = entity[2]
                    entity_type = entity[3]
                    db_id = entity[4]
                    line_values = [
                        str(doc_id),
                        start,
                        end,
                        mention,
                        entity_type,
                        db_id,
                    ]
                    writer.write("\t".join(line_values) + "\n")
                doc_id += 1


class HunFlairPredictor(Predictor):
    def __init__(
        self,
        data_set: str,
        model: str,
        output_file: Path,
        multi_task_learning: bool,
        entity_types: list,
        test_only: bool = False,
        single: bool = False,
    ):
        self.sentence_splitter = SciSpacySentenceSplitter()
        self.multi_task_learning = multi_task_learning
        self.entity_types = sorted(entity_types)
        if single:
            self.model = SequenceTagger.load(model)
        elif multi_task_learning:
            print("Loading multi-task learning model")
            self.model = PrefixedSequenceTagger.load(model)
        else:
            self.model = Classifier.load(model)
        super().__init__(data_set, output_file, test_only)

    def _tag_documents(self):
        print("Tagging documents")

        for document_id in tqdm(self.documents.keys(), total=len(self.documents)):
            if "title" in self.documents[document_id]:
                text = self.documents[document_id]["title"]
            if "abstract" in self.documents[document_id]:
                text += " "
                text += self.documents[document_id]["abstract"]
            if "fulltext" in self.documents[document_id]:
                text = self.documents[document_id]["fulltext"]

            sentences = self.sentence_splitter.split(text)
            for sentence in sentences:
                self.model.predict(sentence)

                for annotation_layer in sentence.annotation_layers.keys():
                    for entity in sentence.get_spans(annotation_layer):
                        start_position = sentence.start_position + entity.start_position
                        end_position = sentence.start_position + entity.end_position
                        self.annotations[document_id] += [
                            (
                                str(start_position),
                                str(end_position),
                                entity.text,
                                entity.tag.lower(),
                                "",
                            )
                        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", default="bionlp_st_2013_cg")
    parser.add_argument(
        "--model",
        default="/home/hunflair2/best-model.pt",
    )
    parser.add_argument(
        "--output_file",
        default="data/bionlp_st_2013_cg/hunflair2.txt",
    )
    parser.add_argument("--multi_task_learning", action="store_true", default=True)
    parser.add_argument(
        "--entity_types",
        nargs="*",
        default=["diseases"],
    )
    parser.add_argument("--predict_test_only", action="store_true", default=False)
    parser.add_argument("--single", action="store_true", default=False)
    args = parser.parse_args()

    output_file = Path(args.output_file)
    os.makedirs(str(output_file.parent), exist_ok=True)

    HunFlairPredictor(
        args.input_dataset,
        args.model,
        output_file,
        args.multi_task_learning,
        sorted(args.entity_types),
        args.predict_test_only,
        args.single,
    )
