# Bayesian Linear Regression 

![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/scratch-bayesian-salary-regression)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/scratch-bayesian-salary-regression)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/scratch-bayesian-salary-regression)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/scratch-bayesian-salary-regression?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/scratch-bayesian-salary-regression?style=social)

A comprehensive end-to-end implementation of a Bayesian linear regression model to predict individual salaries using conjugate Gaussian–Inverse-Gamma priors, optimized Gibbs sampling, and full posterior diagnostics.

## 📖 Overview

This project provides a self-contained Jupyter notebook that:

- Loads and preprocesses salary data with predictors: cost, LSAT, GPA, age, library volume, log(cost), and institutional rank  
- Specifies weakly-informative priors for regression coefficients and error variance  
- Implements an optimized Gibbs sampler (Cholesky sampling, pre-inversion of constant matrices)  
- Runs multiple chains and assesses convergence via trace plots, ACF, Effective Sample Size (ESS), and Gelman–Rubin $\hat R$  
- Summarizes posterior distributions (means, medians, SDs, 95% HDIs, sign probabilities)  
- Explores joint parameter dependencies (pairplots, correlation heatmaps)  
- Performs Posterior Predictive Checks (histogram, KDE, HDI shading)  
- Demonstrates model refinements (Student-$t$ likelihood, log-transform, mixture models, heteroskedasticity)

## ✨ Key Features

- **Conjugate Bayesian setup**: closed-form updates for $\beta$ and $\sigma^2$  
- **Optimized Gibbs sampler**: Cholesky draws and precomputed inverses for speed  
- **Robust diagnostics**: trace, ACF, ESS, $\hat R$ and ArviZ integration  
- **Rich posterior summaries**: HDIs, posterior $P(\beta>0)$, forest plots  
- **Flexible PPCs**: histograms, KDE, rug plots, HDI shading  
- **Extension recipes**: code snippets for Student-$t$ errors, mixtures, transformations  

## 🚀 Quick Start

1. **Dependencies**  
   - Python 3.8+  
   - NumPy, SciPy, Pandas  
   - Matplotlib, Seaborn  
   - Statsmodels (for ACF)  
   - ArviZ (for ESS, $\hat R$)
  
2. **Environment Setup**  
   ```bash
   pip install numpy scipy pandas matplotlib seaborn statsmodels arviz
   ```
3. **Open & Execute**
   - Navigate to Linear_Regression.ipynb
   - Run all cells in sequential order to reproduce data loading, model specification, Gibbs sampling, diagnostics, posterior summaries, PPCs, and extensions.

## 📈 Results & Interpretation

- **Predictor Effects**  
  - **Strong positive**: Cost, GPA, Library volume (95% HDI excludes zero, P(β>0)>0.99)  
  - **Strong negative**: Log(cost), Institutional rank (95% HDI excludes zero, P(β>0)<0.01)  
  - **Ambiguous**: LSAT, Age (HDIs straddle zero, moderate P(β>0))

- **Sampling Diagnostics**  
  - High Effective Sample Size (ESS > 70 000)  
  - Gelman–Rubin $\hat R = 1.00$  
  - ACF near zero beyond lag 0 → almost independent draws

- **Predictive Fit**  
  - Posterior Predictive Checks reveal that the Normal-error model underestimates multimodality and heavy tails in the observed salary distribution.  
  - Model refinements (Student-t errors, mixture components, transformations) are recommended to capture sharp peaks and extreme values.

---

## 🤝 Contributing

Contributions and suggestions are welcome! Please:

- Open an issue to propose enhancements or report bugs  
- Submit pull requests with clear descriptions of changes  
- Include unit tests for any new sampler or diagnostic functions

---

## 📜 License

Released under the **MIT License**

   
