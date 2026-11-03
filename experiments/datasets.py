"""
Dataset loading for the paper's six benchmarks (Table I) plus synthetic sets for smoke tests.

Every loader returns a `Dataset` with numpy arrays. Classification targets are integer labels,
regression targets are float arrays of shape (n, n_targets) left in their original units, since
the paper reports MSE in the target's units (e.g. frame rate, dB).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import LabelEncoder, QuantileTransformer


@dataclass
class Dataset:
  name: str
  task: str  # 'regression' or 'classification'
  X: np.ndarray
  y: np.ndarray

  @property
  def inp_dim(self) -> int:
    return self.X.shape[1]

  @property
  def out_dim(self) -> int:
    if self.task == 'classification':
      return int(self.y.max()) + 1
    return self.y.shape[1]


def _scale_features(X: np.ndarray) -> np.ndarray:
  # Same transform as load_data.py, fitted on the whole dataset like the original code.
  scaler = QuantileTransformer(output_distribution='normal', random_state=0)
  return scaler.fit_transform(X).astype(np.float32)


def _load_openml(loader: str) -> Tuple[np.ndarray, np.ndarray]:
  # Imported lazily so that the telecom/synthetic sets do not need openml installed.
  from load_data import load_covertype, load_higgs
  if loader == 'covertype':
    X, y = load_covertype()
    y = np.argmax(y, axis=1)
  elif loader == 'higgs_large':
    X, y = load_higgs(small=False)
  elif loader == 'higgs':
    X, y = load_higgs(small=True)
  else:
    raise ValueError(f'Unknown openml loader {loader}')
  return X.astype(np.float32), y.astype(np.int64)


def _load_table(path: str, n_targets: Optional[int], target_columns: Optional[list]) -> Tuple[np.ndarray, np.ndarray]:
  """Loads a .csv/.parquet/.npz file. Targets are `target_columns`, or else the last `n_targets` columns."""
  if not os.path.exists(path):
    raise FileNotFoundError(
      f'{path} not found. The telecom datasets are not public; export them to this path '
      '(see experiments/README.md) or set TABGNS_DATA to the directory that holds them.')
  if path.endswith('.npz'):
    data = np.load(path)
    return data['X'], data['y']
  df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
  if target_columns:
    y = df[list(target_columns)]
    X = df.drop(columns=list(target_columns))
  else:
    assert n_targets, 'Set either dataset.n_targets or dataset.target_columns'
    X, y = df.iloc[:, :-n_targets], df.iloc[:, -n_targets:]
  return X.to_numpy(), y.to_numpy()


def load_dataset(cfg: dict) -> Dataset:
  kind, task = cfg['kind'], cfg['task']
  if kind == 'openml':
    X, y = _load_openml(cfg['loader'])
  elif kind == 'table':
    X, y = _load_table(cfg['path'], cfg.get('n_targets'), cfg.get('target_columns'))
    mask = ~(pd.isna(X).any(axis=1) | pd.isna(y).reshape(len(y), -1).any(axis=1))
    X, y = X[mask], y[mask]
    X = _scale_features(X.astype(np.float64))
  elif kind == 'synthetic':
    if task == 'regression':
      X, y = make_regression(n_samples=cfg['n_samples'], n_features=cfg['n_features'],
                             n_targets=cfg['n_targets'], noise=cfg.get('noise', 1.0), random_state=0)
    else:
      X, y = make_classification(n_samples=cfg['n_samples'], n_features=cfg['n_features'],
                                 n_informative=cfg['n_features'] // 2, n_classes=cfg['n_classes'],
                                 random_state=0)
    X = _scale_features(X)
  else:
    raise ValueError(f'Unknown dataset kind {kind}')

  if task == 'classification':
    y = LabelEncoder().fit_transform(np.asarray(y).reshape(-1)).astype(np.int64)
  else:
    y = np.asarray(y, dtype=np.float32).reshape(len(y), -1)
  return Dataset(name=cfg['name'], task=task, X=np.asarray(X, dtype=np.float32), y=y)


def split(ds: Dataset, seed: int):
  """60/20/20 train/valid/test split, as in load_data.split_train_valid_test."""
  from sklearn.model_selection import train_test_split
  x_train, x_test, y_train, y_test = train_test_split(ds.X, ds.y, test_size=0.2, random_state=seed)
  x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=seed)
  return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
