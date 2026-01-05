from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SCRIPTS.cli_utils import (
    build_design_matrix,
    load_dataset,
    load_metadata,
    load_posterior,
    parse_columns_arg,
    standardize_matrix,
)
from src.analize_chain_convergence import build_idata_from_chains, arviz_mcmc_report
from src.analize_mcmc import summarize_beta_posterior
from src.posterior_predictive_check import (
    attach_posterior_predictive_y,
    plot_ppc_density_y,
    plot_ppc_residuals,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run posterior summaries and PPC from saved posterior draws.")
    parser.add_argument("--posterior", required=True, help="Path to posterior.npz.")
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to metadata.json (defaults to sibling of posterior).",
    )
    parser.add_argument("--data", default=None, help="CSV data for PPC.")
    parser.add_argument("--target", default=None, help="Target column name.")
    parser.add_argument(
        "--features",
        default=None,
        help="Comma-separated feature columns. Defaults to metadata or dataset defaults.")
    parser.add_argument(
        "--standardize",
        dest="standardize",
        action="store_true",
        help="Standardize predictors (z-score).")
    parser.add_argument(
        "--no-standardize",
        dest="standardize",
        action="store_false",
        help="Do not standardize predictors.")
    parser.set_defaults(standardize=None)
    parser.add_argument(
        "--log-y",
        dest="log_y",
        action="store_true",
        help="Model log(y) instead of y.")
    parser.add_argument(
        "--no-log-y",
        dest="log_y",
        action="store_false",
        help="Use y as-is.")
    parser.set_defaults(log_y=None)

    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for summaries and plots.")

    parser.add_argument(
        "--ppc",
        dest="ppc",
        action="store_true",
        help="Run posterior predictive checks (requires --data).",
    )
    parser.add_argument(
        "--no-ppc",
        dest="ppc",
        action="store_false",
        help="Skip posterior predictive checks.",
    )
    parser.set_defaults(ppc=False)
    parser.add_argument("--ppc-seed", type=int, default=123)
    parser.add_argument("--ppc-hdi", type=float, default=0.94)
    parser.add_argument("--ppc-subsample", type=int, default=5000)
    return parser


def _resolve_setting(arg_value, meta_value, default):
    if arg_value is not None:
        return arg_value
    if meta_value is not None:
        return meta_value
    return default


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    posterior_path = Path(args.posterior)
    if not posterior_path.exists():
        raise FileNotFoundError(f"Posterior file not found: {posterior_path}")

    metadata_path = args.metadata
    if metadata_path is None:
        candidate = posterior_path.parent / "metadata.json"
        metadata_path = str(candidate) if candidate.exists() else None

    metadata = load_metadata(metadata_path)

    beta_post, sigma_post = load_posterior(posterior_path)

    features = parse_columns_arg(args.features)
    if features is None:
        features = metadata.get("features")

    target = args.target or metadata.get("target") or "salary"

    standardize = _resolve_setting(args.standardize, metadata.get("standardize"), False)
    log_y = _resolve_setting(args.log_y, metadata.get("log_y"), False)

    coef_names = metadata.get("coef_names")
    p = beta_post.shape[1]
    if not coef_names:
        if features is None:
            coef_names = ["intercept"] + [f"x{idx}" for idx in range(1, p)]
        else:
            coef_names = ["intercept"] + list(features)
    if len(coef_names) != p:
        coef_names = ["intercept"] + [f"x{idx}" for idx in range(1, p)]

    out_dir = Path(args.out_dir) if args.out_dir else posterior_path.parent / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    idata = build_idata_from_chains(
        beta_chains=[beta_post],
        sigma2_chains=[sigma_post],
        coef_names=coef_names,
    )

    summary_df = summarize_beta_posterior(beta_post, ci=args.ci)
    summary_path = out_dir / "beta_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    report = arviz_mcmc_report(idata, var_names=("beta", "sigma2"), hdi_prob=args.ci)
    report_path = out_dir / "mcmc_report.txt"
    report_path.write_text(report["text"], encoding="utf-8")
    report_summary_path = out_dir / "mcmc_summary.csv"
    report["summary"].to_csv(report_summary_path)

    if args.ppc:
        if args.data is None:
            raise ValueError("--ppc requires --data.")
        X, y, features = load_dataset(args.data, target, features=features)

        x_mean = metadata.get("x_mean")
        x_std = metadata.get("x_std")
        if standardize:
            use_stored = x_mean is not None and x_std is not None
            if use_stored:
                x_mean_arr = np.asarray(x_mean, dtype=float)
                x_std_arr = np.asarray(x_std, dtype=float)
                if x_mean_arr.shape[0] == X.shape[1] and x_std_arr.shape[0] == X.shape[1]:
                    X, _, _ = standardize_matrix(X, mu=x_mean_arr, sd=x_std_arr)
                else:
                    X, _, _ = standardize_matrix(X)
            else:
                X, _, _ = standardize_matrix(X)

        X_design = build_design_matrix(X, add_intercept=True)

        idata_ppc = attach_posterior_predictive_y(
            idata,
            X_design,
            seed=args.ppc_seed,
            var_name="y",
        )

        density_path = out_dir / "ppc_density.png"
        fig, _ = plot_ppc_density_y(
            idata=idata_ppc,
            y=y,
            var_name="y",
            hdi_prob=args.ppc_hdi,
            show=False,
            savepath=density_path,
            obs_transform=np.log if log_y else None,
        )
        plt.close(fig)

        residual_path = out_dir / "ppc_residuals.png"
        out = plot_ppc_residuals(
            idata=idata_ppc,
            X=X_design,
            y=y,
            var_name="y",
            hdi_prob=args.ppc_hdi,
            show=False,
            savepath=residual_path,
            y_transform=np.log if log_y else None,
            subsample_draws=args.ppc_subsample,
        )
        plt.close(out["fig"])

    print(f"Saved beta summary to {summary_path}")
    print(f"Saved MCMC report to {report_path}")
    print(f"Saved MCMC summary to {report_summary_path}")
    if args.ppc:
        print(f"Saved PPC density to {density_path}")
        print(f"Saved PPC residuals to {residual_path}")


if __name__ == "__main__":
    main()
