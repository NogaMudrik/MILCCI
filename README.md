## Multi-Integration of Labels Across Categories for Component Identification in Multi-trial Time Series (MILCCI), ICML 2026

[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2602.04270-b31b1b.svg)](https://arxiv.org/abs/2602.04270)

**Multi-axis Interpretable Latent Component and Condition Inference**

MILCCI decomposes a 3D tensor **Y** (e.g., neurons × time × trials) into
condition-varying spatial maps **A** and temporal traces **Φ**, with
similarity regularization along multiple label axes.

Each trial carries a multi-axis label (e.g., stimulus identity × block
number × trial order). MILCCI assigns a subset of ensembles to each
axis and enforces that the spatial map **A** is shared across trials
that agree on that axis, while allowing variation along other axes.

## Citation

If you use MILCCI in your research, please cite:

```
@article{mudrik2026multi,
  title={Multi-Integration of Labels across Categories for Component Identification (MILCCI)},
  author={Mudrik, Noga and Chen, Yuxi and Mishne, Gal and Charles, Adam S},
  journal={ArXiv},
  pages={arXiv--2602},
  year={2026}
}
```
Mudrik, N., Chen, Y., Mishne, G., & Charles, A. S. (2026). Multi-Integration of Labels across Categories for Component Identification (MILCCI). ArXiv, arXiv-2602.
Soon at ICML 2026.

## Installation

```bash
pip install git+https://github.com/NogaMudrik/MILCCI.git
```

**Requirements:** Python ≥ 3.7, NumPy < 2.0, SciPy ≥ 1.7, scikit-learn ≥ 1.0.

## Quick start

```python
import milcci

# generate synthetic data
synth = milcci.generate_synthetic_data(
    N=50, T=100,
    n_ensembles_each=[2, 2],
    axis_values=[[0, 1, 2], [0, 1]],
    noise_std=0.2,
)

# run MILCCI
result = milcci.fit(
    data=synth['Y'],
    labels=synth['labels'],
    numbers2tuples=synth['numbers2tuples'],
    n_ensembles=4,
    n_ensembles_each=[2, 2],
    nu=[0.01] * 4,
    lambda_similarity=100,
    decor_A=2,
    num_repeats=15,
    split_A=True,
)


```

## Usage

### `milcci.fit(data, labels, numbers2tuples, ...)`

Main decomposition function. Returns a dict with:

| Key       | Shape              | Description                          |
|-----------|--------------------|--------------------------------------|
| `Phi`     | (T, P, M)          | Temporal traces                      |
| `A`       | (N, P, K)          | Spatial maps per unique condition    |
| `A_full`  | (N, P, M)          | Spatial maps expanded to all trials  |
| `params`  | dict               | Run parameters and metadata          |

### Key parameters

| Parameter            | Description                                        |
|----------------------|----------------------------------------------------|
| `n_ensembles`        | Total number of components                         |
| `n_ensembles_each`   | Components per label axis (must sum to n_ensembles) |
| `nu`                 | Per-ensemble similarity strength                   |
| `lambda_similarity`  | Global regularization weight                       |
| `cont_axis_list`     | Which axes are continuous (e.g. trial number)      |
| `split_A`            | Infer separate A per axis-value (recommended)      |
| `style_infer_phi`    | `'LS'` or `'dynamic_prior'` (LDS-regularized)     |




## Running the demo

```bash
python examples/demo_synthetic.py
```

Or open `examples/demo_notebook.ipynb` in Jupyter for an interactive walkthrough.

## Repository structure

```
MILCCI/
├── milcci/
│   ├── __init__.py         # public API
│   ├── core.py             # fit(), reconstruct()
│   ├── solvers.py          # least-squares / NNLS solvers
│   ├── phi_inference.py    # Phi update (LS + LDS prior)
│   ├── regularization.py   # basis patterns, nu matrices
│   ├── evaluation.py       # R^2, correlation metrics
│   ├── synthetic.py        # GP-based synthetic data generator
│   ├── plotting.py         # visualization functions
│   └── utils.py            # small utilities
├── examples/
│   ├── demo_synthetic.py
│   └── demo_notebook.ipynb
├── tests/
│   └── test_milcci.py
├── setup.py
├── LICENSE
├── .gitignore
└── README.md
```

## License

MIT
