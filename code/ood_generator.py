"""Synthetic OOD generation utilities.

Common strategies:
- Gaussian noise injection
- Feature shuffling
- Feature masking / corruption
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class OODConfig:
    method: Literal['noise', 'shuffle', 'mask'] = 'noise'
    noise_std: float = 0.5
    mask_ratio: float = 0.2
    random_state: int = 42


class OODGenerator:
    def __init__(self, config: OODConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_state)

    def add_noise(self, X: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(loc=0.0, scale=self.config.noise_std, size=X.shape)
        return X + noise

    def shuffle_features(self, X: np.ndarray) -> np.ndarray:
        X_ood = X.copy()
        for col in range(X_ood.shape[1]):
            self.rng.shuffle(X_ood[:, col])
        return X_ood

    def mask_features(self, X: np.ndarray) -> np.ndarray:
        X_ood = X.copy()
        mask = self.rng.random(X_ood.shape) < self.config.mask_ratio
        X_ood[mask] = 0.0
        return X_ood

    def generate(self, X: np.ndarray) -> np.ndarray:
        if self.config.method == 'noise':
            return self.add_noise(X)
        if self.config.method == 'shuffle':
            return self.shuffle_features(X)
        if self.config.method == 'mask':
            return self.mask_features(X)
        raise ValueError(f'Unsupported OOD method: {self.config.method}')


if __name__ == '__main__':
    X = np.random.randn(8, 4).astype(np.float32)
    config = OODConfig(method='shuffle')
    generator = OODGenerator(config)
    print(generator.generate(X))
