# DrugGPT: Transformer-Based Generative Models for Drug Discovery ## A Comparative Study of Tokenization Methods

This repository contains the source code, preprocessed dataset, for the paper:

> **"Transformer-Based Generative Models for Drug Discovery:
> A Comparative Study of Tokenization Methods"**
> Kaoutar M'RHAR, Mohamed-Amine CHADI, Hajar Mousannif
> *Journal of Chemical Information and Modeling*, 2026
> DOI: [to be added upon acceptance]

---

## Repository Structure

| File | Description |
|------|-------------|
| `character_level.ipynb` | Character-level tokenization — full pipeline |
| `atom_level.ipynb` | Atom-level tokenization — full pipeline |
| `functional_group.ipynb` | Functional-group tokenization — full pipeline |
| `fragment_brics.py` | Fragment-BRICS tokenization — full pipeline |
| `bpe_tokenization.py` | Byte Pair Encoding — full pipeline |
| `spe_tokenization.py` | SMILES Pair Encoding — full pipeline |
| `chembl21_preprocessed.smi` | Preprocessed ChEMBL21 dataset |
| `utils.py` | Shared utility functions for evaluations metrics |

Each notebook and script contains the complete pipeline:
DrugGPT model architecture, training, molecule generation,
and evaluation for that tokenization strategy.

---

## Dataset

The original ChEMBL21 dataset is publicly available at:
https://doi.org/10.6019/CHEMBL.database.21

The preprocessed file `chembl21_preprocessed.smi`
was obtained after three steps:
1. Removal of invalid SMILES (RDKit sanitization)
2. Removal of duplicate entries
3. Addition of start token `<` and end token `>`

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## How to Run

Open any notebook in Jupyter and run all cells sequentially.
Each notebook is fully self-contained.

```bash
jupyter notebook functional_group.ipynb
```

For `.py` files:
```bash
python fragment_brics.py
python bpe_tokenization.py
python spe_tokenization.py
```

## Contact

For questions please open a GitHub Issue or contact:
[your institutional email]
