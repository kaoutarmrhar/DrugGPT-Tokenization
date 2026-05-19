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
| `tokenization/character_level.ipynb` | Character-level tokenization — full pipeline |
| `tokenization/atom_level.ipynb` | Atom-level tokenization — full pipeline |
| `tokenization/functional_group.ipynb` | Functional-group tokenization — full pipeline |
| `tokenization/fragment_brics.py` | Fragment-BRICS tokenization — full pipeline |
| `tokenization/bpe_tokenization.py` | Byte Pair Encoding — full pipeline |
| `tokenization/spe_tokenization.py` | SMILES Pair Encoding — full pipeline |
| `data/chembl21_preprocessed.smi` | Preprocessed ChEMBL21 dataset |
| `utils/utils.py` | Shared utility functions |
| `results/evaluation_results.csv` | Full results table (mean ± SD) |

Each notebook and script contains the complete pipeline:
DrugGPT model architecture, training, molecule generation,
and evaluation for that tokenization strategy.

---

## Dataset

The original ChEMBL21 dataset is publicly available at:
https://doi.org/10.6019/CHEMBL.database.21

The preprocessed file `data/chembl21_preprocessed.smi`
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
jupyter notebook tokenization/functional_group.ipynb
```

For `.py` files:
```bash
python tokenization/fragment_brics.py
python tokenization/bpe_tokenization.py
python tokenization/spe_tokenization.py
```

---

## Results

| Metric | Character | Atom | BRICS | **Functional** | BPE | SPE |
|--------|-----------|------|-------|----------------|-----|-----|
| Validity | 80.00±1.5% | 90.00±1.9% | 100.00±0.0% | **100.00±0.0%** | 10.00±2.8% | 80.00±3.6% |
| Uniqueness | 100.00% | 100.00% | 100.00% | **100.00%** | 100.00% | 100.00% |
| Novelty | 100.00% | 100.00% | 100.00% | **100.00%** | 100.00% | 100.00% |
| Diversity | 0.92±0.03 | 0.92±0.05 | 0.82±0.01 | **0.92±0.06** | 0.00 | 0.91±0.07 |
| Drug-likeness | 62.50±2.4% | 66.67±1.8% | 88.89±1.2% | **90.00±1.0%** | 0.00% | 87.50±1.5% |

Functional-group tokenization achieves the best overall
performance: highest drug-likeness (90.00%) with perfect
validity (100.00%) and high diversity (0.92).

---

## License

MIT License — see `LICENSE` for details.

---

## Citation

```bibtex
@article{mrhar2026druggpt,
  title={Transformer-Based Generative Models for Drug 
         Discovery: A Comparative Study of Tokenization 
         Methods},
  author={M'RHAR, Kaoutar and CHADI, Mohamed-Amine 
          and Mousannif, Hajar},
  journal={Journal of Chemical Information and Modeling},
  year={2026},
  doi={to be added upon acceptance}
}
```

---

## Contact

For questions please open a GitHub Issue or contact:
[your institutional email]
