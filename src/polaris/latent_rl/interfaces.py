from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ChunkPolicy(Protocol):
    """Black-box frozen policy: (obs, initial noise) -> action chunk."""

    noise_shape: tuple[int, int]  # (action_horizon, action_dim) of the model's flow noise

    def query(self, obs: dict, noise: np.ndarray | None = None) -> np.ndarray: ...


@runtime_checkable
class BatchChunkPolicy(Protocol):
    """ChunkPolicy that answers many observations in one call: noise (N, H, D) -> chunks (N, H, A)."""

    noise_shape: tuple[int, int]

    def query_batch(self, obs_list: list[dict], noise: np.ndarray | None = None) -> np.ndarray: ...


@runtime_checkable
class ObsEncoder(Protocol):
    """Maps an env-specific obs dict to a flat float feature vector."""

    feat_dim: int

    def __call__(self, obs: dict) -> np.ndarray: ...
