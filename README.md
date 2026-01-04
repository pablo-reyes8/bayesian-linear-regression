# Bayesian Linear Regression

![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/scratch-bayesian-salary-regression)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/scratch-bayesian-salary-regression)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/scratch-bayesian-salary-regression)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/scratch-bayesian-salary-regression?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/scratch-bayesian-salary-regression?style=social)

A full Bayesian linear regression workflow for salary prediction, with conjugate and non-conjugate sampling, diagnostics, and posterior predictive checks. The notebooks show the end to end analysis, while the `src/` modules keep the core sampling and plotting logic reusable.

## Overview

This project covers:

- Data preprocessing and design matrix construction
- Conjugate Gaussian-Inverse-Gamma regression with optimized Gibbs sampling
- Non-conjugate priors via Metropolis-Hastings updates
- Diagnostics: trace plots, ACF, ESS, R-hat, and posterior summaries
- Posterior predictive checks (PPC) with density, KDE, HDI, and residual diagnostics
- Log-scale modeling support, including calibrated priors for sigma2

## Notebooks

- `notebooks/Linear_Regression.ipynb` - baseline conjugate model and full diagnostics
- `notebooks/model_train_no_conjugate.ipynb` - MH for slopes, conjugate sigma2, and log(y) variants

## Posterior Predictive Checks (PPC)

Utilities live in `src/posterior_predictive_check.py` and now support transformations so the observed data and replicated draws share the same scale.

Example for log(y):

```python
idata_ppc = attach_posterior_predictive_y(idata, X_design, seed=123, var_name="y")

plot_ppc_density_y(
    idata=idata_ppc,
    y=y,
    obs_transform=np.log,
    var_name="y",
    fmt_thousands=False,
)

plot_ppc_residuals(
    idata=idata_ppc,
    X=X_design,
    y=y,
    y_transform=np.log,
    standardize=True,
    use_posterior_predictive=True,
)
```

## Log-scale sigma2 calibration

When modeling log(y), the prior for sigma2 must match the log-scale variance. A large `b0` in an Inv-Gamma prior will dominate the posterior and explode the PPC variance.

Recommended calibration:

```python
y_log = np.log(y)
s2_emp = y_log.var(ddof=1)

a0 = 3.0
b0 = (a0 - 1) * s2_emp  # prior mean approx s2_emp

beta_post, sigma_post, acc, info = MCMC_LM_beta_nonconj_sigma_conj(
    X_design, y_log,
    a0=a0, b0=b0,
    sigma2_init=s2_emp,
    # ... other settings
)
```

## Quick start

```bash
pip install numpy scipy pandas matplotlib seaborn statsmodels arviz
```

Run the notebooks in order to reproduce the full analysis.

## Testing

A basic pytest suite is available in `tests/`.

```bash
pytest -q
```

## Docker

Build and run a local container:

```bash
docker build -t bayesian-lr .
docker run -it --rm -v "$(pwd)":/app bayesian-lr bash
```

Then run:

```bash
pytest -q
```

## Repository layout

- `src/` - samplers, diagnostics, PPC utilities
- `notebooks/` - analysis notebooks
- `data/` - raw and processed datasets
- `experiments/` - ad hoc experiments and extensions
- `tests/` - pytest coverage for core utilities

## Contributing

Issues and PRs are welcome. If you add new samplers or diagnostics, please include tests.

## License

MIT License
