# HunFlair2

This repository contains the source code to reproduce the experiments conducted in

## Setup

Setup environment and install necessary packages / models:

## Reproduce paper results

To reproduce the main result from the paper simply run:

```bash
python evaluate.py
```

This will load the pre-computed predictions of several tools along with the gold labels.

### Corpora and tools

Results are computed on the following corpora:

- [MedMentions](https://github.com/chanzuckerberg/MedMentions)
- [BioID](https://biocreative.bioinformatics.udel.edu/resources/corpora/bcvi-bio-id-track/)
- [tmVar (v3)](https://github.com/ncbi/tmVar3/blob/main/README.md)

We obtiained the prediction and stored them in the [pubtator format](https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/Format.html)
in the `annotations` folder for the following tools:

- [PubTator](https://www.ncbi.nlm.nih.gov/research/pubtator/api.html)
- [BERN2](https://github.com/dmis-lab/BERN2)
- [SciSpacy](https://github.com/allenai/scispacy)
- [bent](https://github.com/lasigeBioTM/bent)

See the paper for more details.

## Run HunFlair2

You can obtain the predictions from our HunFlair2 tool with the following command:

```bash
python predict_hunflair2.py \
    --input ./annotations/raw/bioid_text.txt \
    --output ./annotations/hunflair2/bioid_text.txt \
    --entity_types species
```

This will load a file in the PubTator formatand store the predictions into the `--output` file
