import numpy as np
import matplotlib

matplotlib.use("Agg")

from src.analize_chain_convergence import build_idata_from_chains
from src.posterior_predictive_check import (
    attach_posterior_predictive_y,
    plot_ppc_density_y,
    plot_ppc_residuals,
)


def _make_idata(rng, draws=80, chains=2, p=2):
    beta_chains = [rng.normal(size=(draws, p)) for _ in range(chains)]
    sigma2_chains = [rng.uniform(0.05, 0.2, size=draws) for _ in range(chains)]
    return build_idata_from_chains(beta_chains=beta_chains, sigma2_chains=sigma2_chains)


def test_attach_posterior_predictive_y_shape():
    rng = np.random.default_rng(0)
    n, p = 12, 2
    X = rng.normal(size=(n, p))
    idata = _make_idata(rng, draws=60, chains=2, p=p)

    idata_ppc = attach_posterior_predictive_y(idata, X, seed=0, var_name="y")
    y_rep = idata_ppc.posterior_predictive["y"].values

    assert y_rep.shape == (2, 60, n)


def test_plot_ppc_density_y_log_transform():
    rng = np.random.default_rng(1)
    y = rng.uniform(5.0, 20.0, size=50)
    y_rep_log = rng.normal(loc=np.log(y).mean(), scale=0.2, size=500)

    fig, ax = plot_ppc_density_y(
        y=y,
        y_rep=y_rep_log,
        obs_transform=np.log,
        show=False,
        fmt_thousands=False,
    )

    lo = min(np.log(y).min(), y_rep_log.min())
    hi = max(np.log(y).max(), y_rep_log.max())
    xlim = ax.get_xlim()

    assert np.isclose(xlim[0], lo * 0.9)
    assert np.isclose(xlim[1], hi * 1.1)
    fig.clf()


def test_plot_ppc_residuals_log_transform():
    rng = np.random.default_rng(2)
    n, p = 15, 2
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    beta_true = np.array([2.0, 0.3])
    y_log = X @ beta_true + rng.normal(scale=0.1, size=n)
    y = np.exp(y_log)

    idata = _make_idata(rng, draws=100, chains=2, p=p)

    out = plot_ppc_residuals(
        idata=idata,
        X=X,
        y=y,
        use_posterior_predictive=False,
        y_transform=np.log,
        subsample_draws=50,
        kde_points=50,
        bins=15,
        show=False,
    )

    assert np.isfinite(out["e_obs"]).all()
    assert np.isfinite(out["e_rep"]).all()
    assert out["e_rep"].size > 0
    out["fig"].clf()
