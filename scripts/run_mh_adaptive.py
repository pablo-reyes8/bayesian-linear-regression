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
from src.model_estimations.adaptativeMH_conjugate_variance import (
    MCMC_LM_beta_nonconj_sigma_conj_adaptcov_slopes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run adaptive MH for beta with conjugate sigma2 (adaptive covariance).")
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

    parser.add_argument("--a0", type=float, default=3.0, help="Inv-Gamma shape.")
    parser.add_argument(
        "--b0",
        type=float,
        default=None,
        help="Inv-Gamma scale. If omitted, calibrate from log(y) variance.")

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

    parser.add_argument("--proposal-scale", type=float, default=1.0)
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
    parser.add_argument("--adapt-max-scale", type=float, default=10.0)
    parser.add_argument("--adapt-min-scale", type=float, default=1e-6)

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
        default="outputs/mh_adaptive",
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
    if args.b0 is None:
        if args.a0 <= 1:
            raise ValueError("a0 must be > 1 to calibrate b0 from variance.")
        b0 = (args.a0 - 1.0) * s2_emp
    else:
        b0 = float(args.b0)

    prior_kwargs = _prepare_prior_kwargs(
        args.prior,
        parse_json_arg(args.prior_kwargs),
        args.prior_b,
    )

    beta_post, sigma_post, acc_rate, info = MCMC_LM_beta_nonconj_sigma_conj_adaptcov_slopes(
        X_design,
        y,
        a0=args.a0,
        b0=b0,
        n_draws=args.n_draws,
        burn_in=args.burn_in,
        thinning=args.thinning,
        seed=args.seed,
        prior=args.prior,
        prior_kwargs=prior_kwargs,
        sigma2_init=s2_emp,
        proposal_scale=args.proposal_scale,
        ridge=args.ridge,
        adapt_scale=args.adapt_scale,
        target_accept=args.target_accept,
        adapt_every=args.adapt_every,
        adapt_start=args.adapt_start,
        adapt_gain=args.adapt_gain,
        adapt_max_scale=args.adapt_max_scale,
        adapt_min_scale=args.adapt_min_scale,
        adapt_cov=args.adapt_cov,
        cov_start=args.cov_start,
        cov_ridge=args.cov_ridge,
        cov_update_every=args.cov_update_every,
        cov_start_at=args.cov_start_at,
        progress=args.progress,
        progress_every=args.progress_every,
        return_info=True,
    )

    sampler_info = {
        "acc_rate_slopes": float(acc_rate),
        "final_proposal_scale": float(info.get("final_proposal_scale")),
        "runtime_sec": float(info.get("runtime_sec")),
    }

    metadata = {
        "model": "mh_beta_conjugate_sigma_adaptcov",
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
        "sigma2_prior": {"a0": float(args.a0), "b0": float(b0), "s2_emp": float(s2_emp)},
        "sampler_info": sampler_info,
    }

    posterior_path, metadata_path = save_posterior(
        args.output_dir, beta_post, sigma_post, metadata)

    print(f"Saved posterior to {posterior_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"beta_post shape: {beta_post.shape}")
    print(f"sigma_post shape: {sigma_post.shape}")
    print(f"acc_rate_slopes: {acc_rate:.3f}")


if __name__ == "__main__":
    main()
