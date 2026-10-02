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


def evaluate_vec(venv, act_fn, n_videos: int = 0) -> tuple[dict[str, float], list[list[np.ndarray]]]:
    """Run one episode in every env of a VecChunkEnv (envs that finish early keep stepping but are ignored).

    `act_fn(feat (N, F)) -> z (N, steer_dim)`. Records the first `n_videos` envs. Returns mean episode metrics over
    all N episodes and the videos.
    """
    feat = venv.reset()
    n = venv.num_envs
    n_videos = min(n_videos, n)
    stats = [EpisodeStats() for _ in range(n)]
    videos = [[] for _ in range(n_videos)]
    finished = np.zeros(n, bool)
    while not finished.all():
        t = venv.step(act_fn(feat), record=0 if finished[:n_videos].all() else n_videos)
        for i in np.flatnonzero(~finished):
            stats[i].add(t.unbatch(i))
        for j in range(n_videos):
            if not finished[j]:
                videos[j] += t.frames[j]
        finished |= t.episode_over
        feat = t.next_feat
    summaries = [s.summary() for s in stats]
    metrics = {k: float(np.mean([s[k] for s in summaries])) for k in summaries[0]}
    metrics["episodes"] = n
    return metrics, videos
