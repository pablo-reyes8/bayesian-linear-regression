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
    load_dataset,
    parse_columns_arg,
    save_posterior,
    standardize_matrix,
    transform_y,
)
from src.model_estimations.gibbs_sampling_conjugate_prior import Gibbs_Sampling


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run conjugate Gibbs sampling for Bayesian linear regression.")
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
    parser.set_defaults(standardize=False)
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
    parser.set_defaults(log_y=False)

    parser.add_argument("--n-draws", type=int, default=50000)
    parser.add_argument("--burn-in", type=int, default=10000)
    parser.add_argument("--thinning", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--m0", type=float, default=0.0, help="Prior mean scalar for beta.")
    parser.add_argument("--V0", type=float, default=1000.0, help="Prior variance scalar for beta.")
    parser.add_argument("--a0", type=float, default=1.1, help="Inv-Gamma shape.")
    parser.add_argument("--b0", type=float, default=5000.0, help="Inv-Gamma scale.")

    parser.add_argument(
        "--output-dir",
        default="outputs/gibbs_conjugate",
        help="Directory to save posterior.npz and metadata.json.")
    return parser


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

    n, p = X_design.shape
    m0 = np.full(p, args.m0)
    V0 = np.eye(p) * args.V0

    beta_post, sigma_post = Gibbs_Sampling(
        m0=m0,
        V0=V0,
        a0=args.a0,
        b0=args.b0,
        n=n,
        p=p,
        X=X_design,
        y=y,
        n_draws=args.n_draws,
        burn_in=args.burn_in,
        thinning=args.thinning,
        seed=args.seed,
    )

    metadata = {
        "model": "gibbs_conjugate",
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
        "prior": {"m0": float(args.m0), "V0": float(args.V0), "a0": float(args.a0), "b0": float(args.b0)},
    }

    posterior_path, metadata_path = save_posterior(
        args.output_dir, beta_post, sigma_post, metadata)

    print(f"Saved posterior to {posterior_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"beta_post shape: {beta_post.shape}")
    print(f"sigma_post shape: {sigma_post.shape}")


if __name__ == "__main__":
    main()
