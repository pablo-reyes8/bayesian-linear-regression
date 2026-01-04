# Bayesian Linear Regression (From Scratch)

![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/scratch-bayesian-salary-regression)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/scratch-bayesian-salary-regression)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/scratch-bayesian-salary-regression)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/scratch-bayesian-salary-regression?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/scratch-bayesian-salary-regression?style=social)

A professional, from-scratch implementation of Bayesian linear regression. The dataset is a toy example; the real goal is to build the full Bayesian pipeline, understand the mechanics, and extend the model step by step.

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
- Diagnostics: trace plots, ACF, ESS, R-hat, posterior summaries.
- Posterior predictive checks: density/KDE, HDI shading, and residual checks.
- Extensions: log-scale modeling, heavier tails, mixtures, heteroskedasticity (as separate experiments).

## Posterior predictive checks (PPC)

We simulate replicated data and compare it to the observed data and residuals:

$y^{rep} \sim \mathcal{N}(X\beta, \sigma^2 I)$

$e^{rep} = y^{rep} - X\beta,\quad e^{obs} = y - X\mathbb{E}[\beta]$

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

## Notebooks

- `notebooks/Linear_Regression.ipynb` - conjugate baseline and diagnostics
- `notebooks/model_train_no_conjugate.ipynb` - MH for slopes, conjugate sigma2, log-scale variants

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
