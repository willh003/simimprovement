import numpy as np

from polaris.rl.chunk_env import ChunkEnv
from polaris.rl.stats import EpisodeStats


def evaluate(chunk_env: ChunkEnv, conditions: list[dict], n_videos: int = 0) -> tuple[dict[str, float], list[list[np.ndarray]]]:
    """Run one episode per initial condition; record frames for the first `n_videos`.

    Returns mean episode metrics (return, return/<term>, success/<metric> rates, ...) and the recorded videos.
    """
    summaries, videos = [], []
    stats = EpisodeStats()
    for i, cond in enumerate(conditions):
        record = i < n_videos
        chunk_env.reset(object_positions=cond)
        stats.reset()
        frames = []
        while True:
            t = chunk_env.step(record=record)
            stats.add(t)
            frames += t.frames
            if t.episode_over:
                break
        summaries.append(stats.summary())
        if record:
            videos.append(frames)
    metrics = {k: float(np.mean([s[k] for s in summaries])) for k in summaries[0]}
    return metrics, videos
