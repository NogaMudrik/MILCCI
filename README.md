# MILCCI

**Multi-axis Interpretable Latent Component and Condition Inference**

MILCCI decomposes a 3D tensor **Y** (neurons × time × trials) into
condition-varying spatial maps **A** and temporal traces **Φ**, with
similarity regularization along multiple label axes.

Each trial carries a multi-axis label (e.g., stimulus identity × block
number × trial order).  MILCCI assigns a subset of ensembles to each
axis and enforces that the spatial map **A** is shared across trials
that agree on that axis, while allowing variation along other axes.

## Citation

If you use MILCCI in your research, please cite:

```
Mudrik, N., Chen, Y., Mishne, G., & Charles, A. S. (2026). Multi-Integration of Labels across Categories for Component Identification (MILCCI). ArXiv, arXiv-2602.
```


## Installation

```bash
pip install -e .
```

**Requirements:** Python ≥ 3.8, NumPy < 2.0, SciPy ≥ 1.7, scikit-learn ≥ 1.0.

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

# evaluate
r2 = milcci.global_r2(synth['Y'], result['A_full'], result['Phi'])
print('R^2 = %.4f' % r2)
```

## API

### `milcci.fit(data, labels, numbers2tuples, ...)`

Main decomposition function.  Returns a dict with:

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

### Evaluation

```python
milcci.global_r2(Y, A_full, Phi)             # scalar R^2
milcci.per_trial_r2(Y, A_full, Phi)           # per-trial R^2 vector
milcci.reconstruction_correlation(Y, A_full, Phi)  # Pearson rho
```

### Synthetic data

```python
milcci.generate_synthetic_data(N, T, n_ensembles_each, axis_values, noise_std)
```

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
