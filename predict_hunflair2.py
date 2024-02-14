import argparse
import time
from typing import Dict, List, Tuple

from bioc import pubtator
from flair.data import Sentence
from flair.models.prefixed_tagger import PrefixedSequenceTagger
from flair.models.entity_mention_linking import EntityMentionLinker
from flair.splitter import SciSpacySentenceSplitter, SentenceSplitter

ENTITY_TYPES = ("disease", "chemical", "gene", "species")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run Hunflair2 on documents in PubTator format")
    parser.add_argument(
        "--input",
        type=str,
        default="./annotations/raw/tmvar_v3_text.txt",
        help="Raw (w/o annotation) file in PubTator format",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./annotations/hunflair2/tmvar_v3.txt",
        help="File with Hunflair2 annotations",
    )
    parser.add_argument("--entity_types", nargs="*", default=["gene"])
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of sentences to precess in one go",
    )
    return parser.parse_args()


def load_documents(path: str) -> Dict[str, pubtator.PubTator]:
    documents = {}
    with open(path) as fp:
        for d in pubtator.load(fp):
            documents[d.pmid] = d
    return documents


def get_document_text(document: pubtator.PubTator) -> str:
    text = ""
    if document.title is not None:
        text += document.title
    if document.abstract is not None:
        text += " "
        text += document.abstract
    return text


def get_pmid_sentence_list(
    documents: Dict[str, pubtator.PubTator], splitter: SentenceSplitter
) -> List[Tuple[str, Sentence]]:
    pmid_sentence_list = []
    for pmid, document in documents.items():
        for s in splitter.split(get_document_text(document)):
            pmid_sentence_list.append((pmid, s))

    return pmid_sentence_list


def main(args: argparse.Namespace):
    assert all(
        et in ENTITY_TYPES for et in args.entity_types
    ), f"There are invalid entity types. All must be one one of: {ENTITY_TYPES}"

    print("Start predicting with Hunflair2:")
    print(f"- input file: {args.input}")
    print(f"- output file: {args.output}")
    print("- load NER model")
    splitter = SciSpacySentenceSplitter()
    tagger = PrefixedSequenceTagger.load("hunflair/hunflair2-ner")
    print(f"- load EL models: {args.entity_types}")
    linkers = [EntityMentionLinker.load(f"{et}-linker") for et in args.entity_types]
    documents = load_documents(args.input)
    print("- split documents into sentences")
    pmid_sentence = get_pmid_sentence_list(documents=documents, splitter=splitter)
    pmids, sentences = zip(*pmid_sentence)
    sentences = list(sentences)

    print("- start tagging")
    start = time.time()
    tagger.predict(sentences, mini_batch_size=args.batch_size)
    for linker in linkers:
        linker.predict(sentences, batch_size=args.batch_size)
    elapsed = round(time.time() - start, 2)
    print(f"- tagging took: {elapsed}s")

    print("- write output file")
    for pmid, sentence in zip(pmids, sentences):
        for span in sentence.get_spans("ner"):
            for link in span.get_labels("link"):
                annotation = pubtator.PubTatorAnn(
                    pmid=pmid,
                    text=span.text,
                    start=span.start_position,
                    end=span.end_position,
                    type=span.tag.lower(),
                    id=link.value,
                )
                documents[pmid].add_annotation(annotation)
    print("- done")

    breakpoint()


if __name__ == "__main__":
    args = parse_args()
    main(args)
