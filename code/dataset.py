"""Dataset loading utilities for OOD detection experiments.

This module is intentionally lightweight so the team can adapt it to tabular,
text, or embedding-based datasets with minimal changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler | None = None


class DataPipeline:
    """Simple pipeline for loading, splitting, and normalizing ID data."""

    def __init__(self, data_path: str | Path, label_column: int = -1, random_state: int = 42):
        self.data_path = Path(data_path)
        self.label_column = label_column
        self.random_state = random_state

    def load_numpy_csv(self, delimiter: str = ',') -> Tuple[np.ndarray, np.ndarray]:
        data = np.loadtxt(self.data_path, delimiter=delimiter, dtype=np.float32)
        X = np.delete(data, self.label_column, axis=1)
        y = data[:, self.label_column].astype(np.int64)
        return X, y

    def split_id_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> DatasetBundle:
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )
        relative_val = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=relative_val,
            random_state=self.random_state,
            stratify=y_train_full,
        )
        return DatasetBundle(X_train, y_train, X_val, y_val, X_test, y_test)

    def normalize(self, bundle: DatasetBundle) -> DatasetBundle:
        scaler = StandardScaler()
        bundle.X_train = scaler.fit_transform(bundle.X_train)
        bundle.X_val = scaler.transform(bundle.X_val)
        bundle.X_test = scaler.transform(bundle.X_test)
        bundle.scaler = scaler
        return bundle

    def run(self) -> DatasetBundle:
        X, y = self.load_numpy_csv()
        bundle = self.split_id_data(X, y)
        bundle = self.normalize(bundle)
        return bundle


if __name__ == '__main__':
    pipeline = DataPipeline('data/Arrhythmia_raw_clean.csv')
    bundle = pipeline.run()
    print('Train:', bundle.X_train.shape, bundle.y_train.shape)
    print('Val:  ', bundle.X_val.shape, bundle.y_val.shape)
    print('Test: ', bundle.X_test.shape, bundle.y_test.shape)
