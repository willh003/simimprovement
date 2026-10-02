"""Thin wandb wrapper: the only module in the RL stack that touches wandb."""

import shutil
from pathlib import Path

import mediapy
import numpy as np
import wandb

from polaris.rl.local_logger import LocalMetricsLogger


def _ensure_ffmpeg():
    """mediapy needs an ffmpeg binary; fall back to the one bundled with imageio-ffmpeg."""
    if shutil.which("ffmpeg") is None:
        import imageio_ffmpeg

        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())


class WandbLogger:
    def __init__(self, project: str, run_dir: str | Path, config: dict | None = None, entity: str | None = None,
                 name: str | None = None, mode: str = "online"):
        self.video_dir = Path(run_dir) / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        _ensure_ffmpeg()
        self.run = wandb.init(project=project, entity=entity, name=name, mode=mode, config=config, dir=str(run_dir))
        self.local = LocalMetricsLogger(run_dir)

    def log(self, metrics: dict, step: int, prefix: str = ""):
        self.local.log(metrics, step, prefix)
        self.run.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=step)

    def log_video(self, key: str, frames: list[np.ndarray], step: int, fps: int = 15):
        """Write an mp4 locally (kept under run_dir/videos) and upload it."""
        if not frames:
            return
        path = self.video_dir / f"{key.replace('/', '_')}_{step}.mp4"
        mediapy.write_video(path, frames, fps=fps)
        self.run.log({key: wandb.Video(str(path), format="mp4")}, step=step)

    def finish(self):
        self.run.finish()
