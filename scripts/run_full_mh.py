from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SCRIPTS.cli_utils import (
    build_design_matrix,
    jsonable,
    load_dataset,
    parse_columns_arg,
    parse_json_arg,
    save_posterior,
    standardize_matrix,
    transform_y,
)
from src.model_estimations.full_metropolis_hastings_regression import (
    MCMC_LM_beta_nonconj_sigma2_nonconj_adaptcov_slopes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full MH for beta and sigma2 (non-conjugate).")
    parser.add_argument("--data", default="data/df_clean.csv", help="Path to CSV data.")
    parser.add_argument("--target", default="salary", help="Target column name.")
    parser.add_argument(
        "--features",
        default=None,
        help="Comma-separated feature columns. Defaults to dataset defaults.")
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
    parser.set_defaults(standardize=True)
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
    parser.set_defaults(log_y=True)

    parser.add_argument("--n-draws", type=int, default=50000)
    parser.add_argument("--burn-in", type=int, default=10000)
    parser.add_argument("--thinning", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--prior", default="laplace")
    parser.add_argument(
        "--prior-kwargs",
        default=None,
        help="JSON string or @path to JSON file with prior kwargs.")
    parser.add_argument(
        "--prior-b",
        type=float,
        default=50.0,
        help="Laplace scale if prior-kwargs not provided.")

    parser.add_argument("--sigma2-prior", default="lognormal")
    parser.add_argument(
        "--sigma2-prior-kwargs",
        default=None,
        help="JSON string or @path to JSON file with sigma2 prior kwargs.")
    parser.add_argument("--sigma2-tau", type=float, default=0.8)

    parser.add_argument(
        "--gibbs-intercept",
        dest="gibbs_intercept",
        action="store_true",
        help="Gibbs update for intercept when prior is flat on intercept.")
    parser.add_argument(
        "--no-gibbs-intercept",
        dest="gibbs_intercept",
        action="store_false",
        help="Disable Gibbs update for intercept.")
    parser.set_defaults(gibbs_intercept=True)

    parser.add_argument("--proposal-scale", type=float, default=0.2)
    parser.add_argument("--ridge", type=float, default=1e-8)

    parser.add_argument(
        "--adapt-scale",
        dest="adapt_scale",
        action="store_true",
        help="Enable adaptation of proposal scale during burn-in.")
    parser.add_argument(
        "--no-adapt-scale",
        dest="adapt_scale",
        action="store_false",
        help="Disable adaptation of proposal scale.")
    parser.set_defaults(adapt_scale=True)
    parser.add_argument("--target-accept", type=float, default=0.25)
    parser.add_argument("--adapt-every", type=int, default=100)
    parser.add_argument("--adapt-start", type=int, default=400)
    parser.add_argument("--adapt-gain", type=float, default=2.0)

    parser.add_argument(
        "--adapt-cov",
        dest="adapt_cov",
        action="store_true",
        help="Enable adaptive covariance during burn-in.")
    parser.add_argument(
        "--no-adapt-cov",
        dest="adapt_cov",
        action="store_false",
        help="Disable adaptive covariance during burn-in.")
    parser.set_defaults(adapt_cov=True)
    parser.add_argument("--cov-start", default="xtx")
    parser.add_argument("--cov-ridge", type=float, default=1e-6)
    parser.add_argument("--cov-update-every", type=int, default=200)
    parser.add_argument("--cov-start-at", type=int, default=500)

    parser.add_argument("--sigma2-prop-scale", type=float, default=0.25)
    parser.add_argument(
        "--adapt-sigma2-scale",
        dest="adapt_sigma2_scale",
        action="store_true",
        help="Enable adaptation for sigma2 RW step.")
    parser.add_argument(
        "--no-adapt-sigma2-scale",
        dest="adapt_sigma2_scale",
        action="store_false",
        help="Disable adaptation for sigma2 RW step.")
    parser.set_defaults(adapt_sigma2_scale=True)
    parser.add_argument("--target-accept-sigma2", type=float, default=0.44)
    parser.add_argument("--adapt-sigma2-every", type=int, default=100)
    parser.add_argument("--adapt-sigma2-start", type=int, default=400)
    parser.add_argument("--adapt-sigma2-gain", type=float, default=1.0)
    parser.add_argument("--sigma2-prop-min", type=float, default=1e-6)
    parser.add_argument("--sigma2-prop-max", type=float, default=10.0)

    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="Print sampler progress.")
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable progress logging.")
    parser.set_defaults(progress=True)
    parser.add_argument("--progress-every", type=int, default=1000)

    parser.add_argument(
        "--output-dir",
        default="outputs/full_mh",
        help="Directory to save posterior.npz and metadata.json.")
    return parser


def _prepare_prior_kwargs(prior: str, raw_kwargs: dict, default_b: float) -> dict:
    kwargs = dict(raw_kwargs or {})
    if prior.lower() == "laplace" and not kwargs:
        kwargs = {"b": float(default_b)}
    if prior.lower() == "normal":
        if "m0" not in kwargs or "V0" not in kwargs:
            raise ValueError("Normal prior requires prior-kwargs with m0 and V0.")
        kwargs["m0"] = np.asarray(kwargs["m0"], dtype=float)
        kwargs["V0"] = np.asarray(kwargs["V0"], dtype=float)
    return kwargs


def _prepare_sigma2_prior_kwargs(
    prior: str,
    raw_kwargs: dict,
    s2_emp: float,
    tau_default: float,
) -> dict:
    kwargs = dict(raw_kwargs or {})
    if prior.lower() == "lognormal" and not kwargs:
        kwargs = {"mu": float(np.log(s2_emp)), "tau": float(tau_default)}
    return kwargs


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    feature_list = parse_columns_arg(args.features)
    X, y, features = load_dataset(args.data, args.target, features=feature_list)

    x_mean = None
    x_std = None
    if args.standardize:
        X, x_mean, x_std = standardize_matrix(X)

    y = transform_y(y, log_y=args.log_y)
    X_design = build_design_matrix(X, add_intercept=True)

    s2_emp = float(np.var(y, ddof=1))

    prior_kwargs = _prepare_prior_kwargs(
        args.prior,
        parse_json_arg(args.prior_kwargs),
        args.prior_b,
    )
    sigma2_prior_kwargs = _prepare_sigma2_prior_kwargs(
        args.sigma2_prior,
        parse_json_arg(args.sigma2_prior_kwargs),
        s2_emp,
        args.sigma2_tau,
    )

    beta_post, sigma_post, info = MCMC_LM_beta_nonconj_sigma2_nonconj_adaptcov_slopes(
        X_design,
        y,
        n_draws=args.n_draws,
        burn_in=args.burn_in,
        thinning=args.thinning,
        seed=args.seed,
        prior=args.prior,
        prior_kwargs=prior_kwargs,
        intercept_idx=0,
        gibbs_intercept=args.gibbs_intercept,
        sigma2_prior=args.sigma2_prior,
        sigma2_prior_kwargs=sigma2_prior_kwargs,
        beta_init=None,
        sigma2_init=s2_emp,
        proposal_scale=args.proposal_scale,
        ridge=args.ridge,
        adapt_scale=args.adapt_scale,
        target_accept=args.target_accept,
        adapt_every=args.adapt_every,
        adapt_start=args.adapt_start,
        adapt_gain=args.adapt_gain,
        adapt_cov=args.adapt_cov,
        cov_start=args.cov_start,
        cov_ridge=args.cov_ridge,
        cov_update_every=args.cov_update_every,
        cov_start_at=args.cov_start_at,
        sigma2_prop_scale=args.sigma2_prop_scale,
        adapt_sigma2_scale=args.adapt_sigma2_scale,
        target_accept_sigma2=args.target_accept_sigma2,
        adapt_sigma2_every=args.adapt_sigma2_every,
        adapt_sigma2_start=args.adapt_sigma2_start,
        adapt_sigma2_gain=args.adapt_sigma2_gain,
        sigma2_prop_min=args.sigma2_prop_min,
        sigma2_prop_max=args.sigma2_prop_max,
        progress=args.progress,
        progress_every=args.progress_every,
        return_info=True,
    )

    sampler_info = {
        "acc_rate_slopes": float(info.get("acc_rate_slopes")),
        "acc_rate_sigma2": float(info.get("acc_rate_sigma2")),
        "final_slopes_scale": float(info.get("final_slopes_scale")),
        "final_logsig2_step": float(info.get("final_logsig2_step")),
        "runtime_sec": float(info.get("runtime_sec")),
    }

    metadata = {
        "model": "full_mh_beta_sigma2_nonconjugate",
        "data_path": str(args.data),
        "target": args.target,
        "features": features,
        "coef_names": ["intercept"] + features,
        "standardize": bool(args.standardize),
        "x_mean": x_mean.tolist() if x_mean is not None else None,
        "x_std": x_std.tolist() if x_std is not None else None,
        "log_y": bool(args.log_y),
        "n_draws": int(args.n_draws),
        "burn_in": int(args.burn_in),
        "thinning": int(args.thinning),
        "seed": int(args.seed),
        "prior": {"name": args.prior, "kwargs": jsonable(prior_kwargs)},
        "sigma2_prior": {"name": args.sigma2_prior, "kwargs": jsonable(sigma2_prior_kwargs), "s2_emp": float(s2_emp)},
        "sampler_info": sampler_info,
    }

    posterior_path, metadata_path = save_posterior(
        args.output_dir, beta_post, sigma_post, metadata)

    print(f"Saved posterior to {posterior_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"beta_post shape: {beta_post.shape}")
    print(f"sigma_post shape: {sigma_post.shape}")
    print(f"acc_rate_slopes: {sampler_info['acc_rate_slopes']:.3f}")
    print(f"acc_rate_sigma2: {sampler_info['acc_rate_sigma2']:.3f}")


if __name__ == "__main__":
    main()
