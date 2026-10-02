"""Pure metric bookkeeping for RL training (no logging backend)."""

from collections import defaultdict

import numpy as np

from polaris.rl.chunk_env import ChunkTransition


class EpisodeStats:
    """Accumulates one episode's ChunkTransitions into scalar summaries."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.terms: dict[str, float] = defaultdict(float)
        self.total = 0.0
        self.chunks = self.steps = 0
        self.last: ChunkTransition | None = None

    def add(self, t: ChunkTransition):
        for name, value in t.reward_terms.items():
            self.terms[name] += value
        self.total += t.reward
        self.chunks += 1
        self.steps += t.steps
        self.last = t

    def summary(self) -> dict[str, float]:
        success = self.last.success if self.last else False
        reached = self.last.reached if self.last else {}
        progress = self.last.info.get("rubric", {}).get("progress") if self.last else None
        return {
            "return": self.total,
            **{f"return/{name}": value for name, value in self.terms.items()},
            # one 0/1 success metric per sparse reward term (+ overall); their means are success rates
            "success/all": float(success),
            **{f"success/{name}": flag for name, flag in reached.items()},
            **({"final_progress": float(progress)} if progress is not None else {}),
            "length_chunks": self.chunks,
            "length_steps": self.steps,
        }


class MeanAccumulator:
    """Collects dicts of scalars and returns per-key means since the last pop."""

    def __init__(self):
        self.values: dict[str, list[float]] = defaultdict(list)

    def add(self, metrics: dict[str, float]):
        for k, v in metrics.items():
            self.values[k].append(float(v))

    def pop_means(self) -> dict[str, float]:
        means = {k: float(np.mean(v)) for k, v in self.values.items()}
        self.values.clear()
        return means


class RollingMean:
    """Moving average over the last `window` values of each key (e.g. recent success rate).

    `add` only returns keys whose window is full, so early values (n=1 -> exactly 0 or 1) are never reported."""

    def __init__(self, window: int = 10):
        self.window = window
        self.values: dict[str, list[float]] = defaultdict(list)

    def add(self, metrics: dict[str, float]) -> dict[str, float]:
        for k, v in metrics.items():
            self.values[k] = (self.values[k] + [float(v)])[-self.window :]
        return {k: float(np.mean(v)) for k, v in self.values.items() if len(v) >= self.window}
