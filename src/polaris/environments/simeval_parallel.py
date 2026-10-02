"""Build a sim-evals env with many parallel envs (tiled cameras); shared by run_pi_parallel.py and train_steering.py.

Needs a running Isaac app (AppLauncher) before import.
"""

import functools

import gymnasium as gym
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg

CAM_NAMES = ["external_cam", "external_cam_2", "wrist_cam"]


def to_tiled(cam_cfg, scale: float) -> TiledCameraCfg:
    """Convert a `CameraCfg` into a `TiledCameraCfg` (same prim path/spawn/offset), resolution multiplied by `scale`.

    Tiled cameras render all envs of a camera into one batched render product, which lifts the ~48-camera cap of
    `CameraCfg` (docs/claude/parallel_envs_bench.md).
    """
    return TiledCameraCfg(
        prim_path=cam_cfg.prim_path,
        height=int(cam_cfg.height * scale),
        width=int(cam_cfg.width * scale),
        data_types=cam_cfg.data_types,
        spawn=cam_cfg.spawn,
        offset=TiledCameraCfg.OffsetCfg(pos=cam_cfg.offset.pos, rot=cam_cfg.offset.rot, convention=cam_cfg.offset.convention),
    )


def fired(term_func):
    """Termination function: reward term `term_func` > 0 (e.g. the object rests in the container).

    `wraps` keeps `term_func`'s signature, which Isaac checks against the term params.
    """

    @functools.wraps(term_func)
    def done(env, **params):
        return term_func(env, **params) > 0

    return done


def make_parallel_env(environment: str, num_envs: int, cam_scale: float = 0.5, keep_cam2: bool = False,
                      success_term: str | None = None):
    """gym env with `num_envs` envs and tiled cameras; returns (env, camera names kept).

    `external_cam_2` (unused by the policy client) is dropped unless `keep_cam2`. With `success_term`, the episode also
    terminates (and the env auto-resets) when that reward term fires.
    """
    import polaris.environments  # noqa: F401  (registers the envs)
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(environment, device="cuda", num_envs=num_envs, use_fabric=True)
    cams = [c for c in CAM_NAMES if keep_cam2 or c != "external_cam_2"]
    if not keep_cam2:
        env_cfg.scene.external_cam_2 = None
        env_cfg.observations.policy.external_cam_2 = None
    for name in cams:
        setattr(env_cfg.scene, name, to_tiled(getattr(env_cfg.scene, name), cam_scale))
    if success_term is not None:
        rew = getattr(env_cfg.rewards, success_term)
        env_cfg.terminations.success = DoneTerm(func=fired(rew.func), params=dict(rew.params))
    return gym.make(environment, cfg=env_cfg), cams
