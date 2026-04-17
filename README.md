# PRISM

PRISM is a molecular machine learning project for benchmark datasets such as
BBBP, ESOL, FreeSolv, HIV, MUV, QM7, QM9, SIDER, Tox21, and PDBBind-related tasks.

## Abstract

> TODO: Add your paper/project abstract here.
>
> Suggested length: 150-250 words covering motivation, method, and key results.

## Framework Overview

<!-- TODO: Replace with your framework figure -->
<!-- Example: ![PRISM Framework](./results/figures/framework.png) -->

![Framework Placeholder](https://via.placeholder.com/1200x500?text=PRISM+Framework+Diagram+Placeholder)

Figure: PRISM framework diagram (placeholder).

## Highlights

- Multi-dataset training and evaluation entrypoints through task-specific scripts.
- Modular model implementations under `model/`.
- Config-driven experiments with reusable YAML files under `configs/`.
- Reproducible structure for data, checkpoints, metrics, and results.

## Project Structure

- `main_bbbp.py`, `main_esol.py`, `main_free.py`, `main_hiv.py`, `main_muv.py`,
  `main_qm7.py`, `main_qm9.py`, `main_sider.py`, `main_tox.py`: task entry scripts.
- `main_bace_recept.py`, `main_pdbbind.py`, `main_peptides_struct.py`: additional
  benchmark/task pipelines.
- `model/`: core model components and architectures.
- `utils/`: utility modules (config loading, graph path logic, splitters).
- `configs/`: experiment configuration files.
- `data/`: local datasets (ignored in Git if configured via `.gitignore`).
- `checkpoints/`: saved model weights.
- `metrics/`: evaluation metrics/log artifacts.
- `results/`: output predictions, tables, and figures.

## Quick Start

### 1) Environment

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
# TODO: add your exact dependencies (requirements.txt or conda env)
pip install -U pip
```

### 2) Prepare Data

Place required datasets under `data/` following each script's expected format.

```text
data/
  bbbp.csv
  esol.csv
  ...
```

### 3) Run an Example Task

```bash
python main_bbbp.py
```

You can run other tasks similarly:

```bash
python main_esol.py
python main_qm9.py
python main_hiv.py
```

## Configuration

Experiment settings are managed in `configs/`.

- TODO: document key config fields (model, optimizer, scheduler, split, seed).
- TODO: add one full example config and command mapping.

## Results

> TODO: Add your benchmark tables and key metrics here.
>
> Suggested sections:
> - Main benchmark comparison
> - Ablation study
> - Efficiency (time/memory) if available

## Citation

If you use this repository, please cite:

```bibtex
@misc{prism,
  title  = {PRISM},
  author = {TODO: author list},
  year   = {2026},
  note   = {TODO: arXiv / venue / URL}
}
```

## Acknowledgements

> TODO: Add acknowledgements, dataset sources, and external libraries.
