# Bayesian Linear Regression (From Scratch)

![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/scratch-bayesian-salary-regression)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/scratch-bayesian-salary-regression)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/scratch-bayesian-salary-regression)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/scratch-bayesian-salary-regression?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/scratch-bayesian-salary-regression?style=social)

A professional, from-scratch implementation of Bayesian linear regression. The dataset is a toy example; the real goal is to build the full Bayesian pipeline, understand the mechanics, and extend the model step by step.

## Index

- [What this repo is](#what-this-repo-is)
- [Core model (short math)](#core-model-short-math)
- [What we actually implement](#what-we-actually-implement)
- [Posterior predictive checks (PPC)](#posterior-predictive-checks-ppc)
- [Posterior analysis utilities](#posterior-analysis-utilities)
- [Log-scale modeling and sigma2 calibration](#log-scale-modeling-and-sigma2-calibration)
- [How to run (Python snippets)](#how-to-run-python-snippets)
- [CLI scripts](#cli-scripts)
- [Notebooks](#notebooks)
- [Quick start](#quick-start)
- [Testing](#testing)
- [Docker](#docker)
- [Repository layout](#repository-layout)
- [Roadmap (examples)](#roadmap-examples)
- [License](#license)

## What this repo is

- A clean, minimal implementation of Bayesian linear regression (not a library).
- A learning and experimentation base to add priors, samplers, and likelihood variants.
- A reproducible workflow: sampling, diagnostics, and posterior predictive checks (PPCs).

## Core model (short math)

**Likelihood**

$y \in \mathbb{R}^n,\; X \in \mathbb{R}^{n \times p}$

$y \mid \beta, \sigma^2 \sim \mathcal{N}(X\beta, \sigma^2 I)$

**Baseline conjugate priors**

$\beta \sim \mathcal{N}(m_0, V_0),\; \sigma^2 \sim \mathrm{Inv\text{-}Gamma}(a_0, b_0)$

**Gibbs updates (sketch)**

$\beta \mid \sigma^2, y \sim \mathcal{N}(m_n, V_n)$

$\sigma^2 \mid \beta, y \sim \mathrm{Inv\text{-}Gamma}(a_n, b_n)$

**Non-conjugate priors**

We replace the Gaussian prior on $\beta$ with Laplace / Student-t / Cauchy / spike-slab and update $\beta$ via MH:

$\alpha = \min\left(1, \frac{p(y \mid \beta') p(\beta')}{p(y \mid \beta) p(\beta)}\right)$

## What we actually implement

- Conjugate Bayesian linear regression with optimized Gibbs updates.
- Metropolis-Hastings updates for non-conjugate priors on $\beta$.
- Full MH for $\beta$ and $\sigma^2$ with non-conjugate priors and adaptive covariance.
- Diagnostics: trace plots, ACF, ESS, R-hat, posterior summaries.
- Posterior predictive checks: density/KDE, HDI shading, and residual checks.
- Extensions: log-scale modeling, heavier tails, mixtures, heteroskedasticity (as separate experiments).

## Posterior predictive checks (PPC)

We simulate replicated data and compare it to the observed data and residuals:

$y^{rep} \sim \mathcal{N}(X\beta, \sigma^2 I)$

$e^{rep} = y^{rep} - X\beta,\quad e^{obs} = y - X\mathbb{E}[\beta]$

## Posterior analysis utilities

We provide reusable analysis helpers in `src/analize_chain_convergence.py` and `src/analize_mcmc.py`:

- `build_idata_from_chains` to assemble ArviZ `InferenceData` objects
- `arviz_mcmc_report` for a compact diagnostics report (ESS, R-hat, MCSE)
- `plot_beta_acf` and `plot_beta_traces` for autocorrelation and mixing checks
- `summarize_beta_posterior` and `posterior_probabilities` for HDIs and sign tests

## Log-scale modeling and sigma2 calibration

When using $\log(y)$, the variance prior must be on the log scale. If $b_0$ is too large, the posterior variance inflates and PPC becomes flat.

Recommended calibration:

```python
y_log = np.log(y)
s2_emp = y_log.var(ddof=1)

a0 = 3.0
b0 = (a0 - 1) * s2_emp  # prior mean approx s2_emp
```

PPC helpers support transformations so observed data and replicated draws are aligned:

```python
plot_ppc_density_y(
    idata=idata_ppc,
    y=y,
    obs_transform=np.log,
)

plot_ppc_residuals(
    idata=idata_ppc,
    X=X_design,
    y=y,
    y_transform=np.log,
)
```

## How to run (Python snippets)

Build a design matrix (standardize predictors + add intercept):

```python
import numpy as np

X0 = X.copy()
muX = X0.mean(axis=0)
sdX = X0.std(axis=0, ddof=1)
sdX[sdX == 0] = 1.0

X_std = (X0 - muX) / sdX
X_design = np.hstack([np.ones((len(y), 1)), X_std])
```

Conjugate Gibbs sampler (two chains):

```python
from src.model_estimations.gibbs_sampling_conjugate_prior import Gibbs_Sampling
from src.analize_chain_convergence import build_idata_from_chains

n, p = X_design.shape
m0 = np.zeros(p)
V0 = np.eye(p) * 10.0
a0, b0 = 2.0, 2.0

beta1, sigma1 = Gibbs_Sampling(m0, V0, a0, b0, n, p, X_design, y, n_draws=20_000, burn_in=5_000, seed=1)
beta2, sigma2 = Gibbs_Sampling(m0, V0, a0, b0, n, p, X_design, y, n_draws=20_000, burn_in=5_000, seed=2)

idata = build_idata_from_chains(
    beta_chains=[beta1, beta2],
    sigma2_chains=[sigma1, sigma2],
)
```

Non-conjugate slopes (MH) + conjugate sigma2:

```python
from src.model_estimations.metropolis_conjugate_variance import MCMC_LM_beta_nonconj_sigma_conj

beta_post, sigma_post, acc, info = MCMC_LM_beta_nonconj_sigma_conj(
    X_design, y,
    a0=3.0, b0=1.0,
    n_draws=50_000,
    burn_in=10_000,
    thinning=1,
    seed=42,
    prior="laplace",
    prior_kwargs={"b": 50.0},
    proposal_scale=0.2,
    adapt=True,
    target_accept=0.25,
    adapt_every=100,
    adapt_start=400,
    return_info=True,
)
```

Posterior summaries and diagnostics:

```python
from src.analize_mcmc import summarize_beta_posterior, posterior_probabilities
from src.analize_chain_convergence import arviz_mcmc_report

tests = [
    {"name": "P(>0)", "fn": lambda s: s > 0},
    {"name": "P(|.|<0.1)", "fn": lambda s: np.abs(s) < 0.1},
]

summary = summarize_beta_posterior(beta1, tests=tests)
report = arviz_mcmc_report(idata, var_names=("beta", "sigma2"), hdi_prob=0.95)
print(report["text"])
```

PPC with replicated data:

```python
from src.posterior_predictive_check import attach_posterior_predictive_y, plot_ppc_density_y

idata_ppc = attach_posterior_predictive_y(idata, X_design, seed=123, var_name="y")
plot_ppc_density_y(idata=idata_ppc, y=y, var_name="y")
```

## CLI scripts

All CLI entry points live in `SCRIPTS/` and save `posterior.npz` + `metadata.json`.

Examples:

```bash
python3 SCRIPTS/run_gibbs_conjugate.py --data data/df_clean.csv --output-dir outputs/gibbs
python3 SCRIPTS/run_mh_conjugate.py --data data/df_clean.csv --output-dir outputs/mh_conjugate
python3 SCRIPTS/run_mh_adaptive.py --data data/df_clean.csv --output-dir outputs/mh_adaptive
python3 SCRIPTS/run_full_mh.py --data data/df_clean.csv --output-dir outputs/full_mh
python3 SCRIPTS/run_inference.py --posterior outputs/full_mh/posterior.npz --data data/df_clean.csv --ppc
```

Notes:

- `--features` expects a comma-separated list like `cost,LSAT,GPA,age,llibvol,lcost,rank`.
- `--prior-kwargs` and `--sigma2-prior-kwargs` accept JSON or `@path/to/file.json`.

## Notebooks

- `notebooks/Linear_Regression.ipynb` - conjugate baseline and diagnostics
- `notebooks/model_train.ipynb` - conjugate pipeline and posterior analysis
- `notebooks/model_train_no_conjugate.ipynb` - MH for slopes, conjugate sigma2, log-scale variants
- `notebooks/model_train_full_mh.ipynb` - full MH for beta and sigma2 (non-conjugate)

## Quick start

```bash
pip install numpy scipy pandas matplotlib seaborn statsmodels arviz
```

Run the notebooks in order to reproduce the analysis.

## Testing

```bash
pytest -q
```

## Docker

```bash
docker build -t bayesian-lr .
docker run -it --rm -v "$(pwd)":/app bayesian-lr bash
```

## Repository layout

- `src/` - samplers, diagnostics, PPC utilities
- `SCRIPTS/` - CLI entry points for running samplers and inference
- `notebooks/` - analysis notebooks
- `data/` - toy datasets
- `experiments/` - extensions and variants
- `tests/` - pytest coverage for core utilities

## Roadmap (examples)

- Robust likelihoods (Student-t)
- Mixture models
- Heteroskedastic models
- Alternative priors and shrinkage
- Additional calibration diagnostics and scoring

## License

MIT License
