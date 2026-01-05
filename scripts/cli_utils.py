from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TARGET = "salary"
DEFAULT_FEATURES = ["cost", "LSAT", "GPA", "age", "llibvol", "lcost", "rank"]


def parse_columns_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def default_features_from_df(df: pd.DataFrame, target: str) -> list[str]:
    defaults = [col for col in DEFAULT_FEATURES if col in df.columns]
    if defaults:
        return defaults
    return [col for col in df.columns if col != target and not col.startswith("Unnamed")]


def load_dataset(
    csv_path: str | Path,
    target: str,
    features: list[str] | None = None,
    drop_unnamed: bool = True,
    drop_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)

    if drop_unnamed:
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    if drop_cols:
        drop_present = [col for col in drop_cols if col in df.columns]
        if drop_present:
            df = df.drop(columns=drop_present)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")

    if features is None:
        features = default_features_from_df(df, target)

    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found: {missing}. Columns: {list(df.columns)}")

    X = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    return X, y, features


def standardize_matrix(
    X: np.ndarray,
    mu: np.ndarray | None = None,
    sd: np.ndarray | None = None,) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mu is None:
        mu = X.mean(axis=0)
    if sd is None:
        sd = X.std(axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    X_std = (X - mu) / sd
    return X_std, mu, sd


def build_design_matrix(X: np.ndarray, add_intercept: bool = True) -> np.ndarray:
    if not add_intercept:
        return X
    n = X.shape[0]
    return np.hstack([np.ones((n, 1)), X])


def transform_y(y: np.ndarray, log_y: bool = False) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    if log_y:
        if np.any(y <= 0):
            raise ValueError("log_y requested but y contains non-positive values.")
        y = np.log(y)
    return y


def parse_json_arg(value: str | None, default: dict | None = None) -> dict:
    if value is None:
        return {} if default is None else default
    value = value.strip()
    if not value:
        return {} if default is None else default
    if value.startswith("@"):
        path = Path(value[1:])
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def save_posterior(
    output_dir: str | Path,
    beta_post: np.ndarray,
    sigma_post: np.ndarray,
    metadata: dict,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    posterior_path = output_dir / "posterior.npz"
    np.savez(posterior_path, beta=beta_post, sigma2=sigma_post)

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    return posterior_path, metadata_path


def load_metadata(path: str | Path | None) -> dict:
    if path is None:
        return {}
    metadata_path = Path(path)
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_posterior(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["beta"], data["sigma2"]


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(val) for val in value]
    return value
