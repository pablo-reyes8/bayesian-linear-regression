import math
import numpy as np
import matplotlib.pyplot as plt

def plot_beta_posteriors(
    beta_post,
    max_plots=None,          # plot first n coefs
    which=None,              # explicit list of indices (overrides max_plots)
    ncols=4,
    figsize=None,
    bins=40,
    density=True,
    alpha=0.7,
    ci=(2.5, 97.5),          # percentile interval
    show_mean=True,
    mean_fmt="{:.3f}",
    show_ci=True,
    ci_color="red",
    mean_color="red",
    add_legend=True,
    suptitle=None,
    show=True,
    savepath=None,
    dpi=200):
    """
    Plot posterior histograms (optionally density) for beta coefficients.

    Parameters
    ----------
    beta_post : array-like, (n_draws, p)
        Posterior draws for beta.
    max_plots : int or None
        Plot first max_plots coefficients (ignored if which is provided).
    which : list[int] or None
        Plot specific coefficient indices.
    ncols : int
        Columns in subplot grid.
    figsize : tuple or None
        Figure size; auto if None.
    bins, density, alpha : histogram parameters
    ci : tuple(float,float)
        Percentile interval to show (e.g., (2.5,97.5)).
    show_mean : bool
        Draw vertical line at posterior mean.
    show_ci : bool
        Draw vertical lines for CI bounds.
    add_legend : bool
        Add legend to each subplot.
    suptitle : str or None
        Figure-level title.
    show : bool
        Whether to plt.show().
    savepath : str or None
        Save figure to path if provided.
    dpi : int
        DPI for saving.

    Returns
    -------
    fig, axes : matplotlib Figure and array of Axes
    """
    beta_post = np.asarray(beta_post)
    if beta_post.ndim != 2:
        raise ValueError(f"beta_post must be 2D (n_draws, p). Got shape={beta_post.shape}")

    n_draws, p = beta_post.shape

    # Decide which coefficients to plot
    if which is not None:
        idx = [int(i) for i in which]
        for i in idx:
            if i < 0 or i >= p:
                raise ValueError(f"Coefficient index out of range: {i} (p={p})")
    else:
        n_to_plot = p if max_plots is None else min(int(max_plots), p)
        idx = list(range(n_to_plot))

    k = len(idx)
    if k == 0:
        raise ValueError("No coefficients selected to plot.")

    nrows = math.ceil(k / ncols)
    if figsize is None:
        figsize = (4.8 * ncols, 2.8 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for j, coef_i in enumerate(idx):
        ax = axes[j]
        samples = beta_post[:, coef_i]

        ax.hist(samples, bins=bins, alpha=alpha, density=density)

        # Percentile CI
        if show_ci and ci is not None:
            lo, hi = np.percentile(samples, list(ci))
            ax.axvline(lo, linestyle=":", color=ci_color, label=f"{ci[0]}%")
            ax.axvline(hi, linestyle=":", color=ci_color, label=f"{ci[1]}%")

        # Mean
        if show_mean:
            mean_val = float(np.mean(samples))
            ax.axvline(
                mean_val,
                linestyle="--",
                linewidth=2,
                color=mean_color,
                label=f"Mean = {mean_fmt.format(mean_val)}",)

        ax.set_title(f"β[{coef_i}] Posterior")
        ax.set_xlabel(f"β[{coef_i}]")
        ax.set_ylabel("Density" if density else "Frequency")

        if add_legend:
            ax.legend()

    for j in range(k, len(axes)):
        axes[j].axis("off")

    if suptitle is not None:
        fig.suptitle(suptitle, y=1.02)

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes