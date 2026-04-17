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

