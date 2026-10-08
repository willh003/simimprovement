"""Privileged (ground-truth) sim state for the steering actor and critic. The frozen base policy never sees it.

No Isaac imports at module level; `env` is the gym env of `make_parallel_env`.
"""

import numpy as np
import torch

PRIV_DIM = 16
POS_CLIP = 5.0  # metres, env-local frame; the workspace is ~1 m, so anything beyond is a physics glitch
VEL_CLIP = 5.0  # m/s  # obj pos (3), obj quat (4), obj lin vel (3), container/goal pos (3), obj - container (3)


class SimEvalsPrivileged:
    """Per-env ground truth of a sim-evals task: the manipulated object and the container (= goal location).

    Object / container are the `object_cfg` / `container_cfg` of the success reward term. Positions are in each env's
    local frame (world pos - env origin), so they are comparable across envs. Layout (PRIV_DIM floats):
    [obj_pos(3), obj_quat_wxyz(4), obj_lin_vel(3), goal_pos(3), obj_pos - goal_pos(3)].
    """

    dim = PRIV_DIM
    bad_envs: np.ndarray = np.zeros(0, int)  # envs whose raw state at the last call was non-finite or out of range
    raw_bad: np.ndarray | None = None  # their raw (unsanitized) rows
    nonfinite_envs: np.ndarray = np.zeros(0, int)  # subset of bad_envs with a NaN/inf (not just out of range)

    def __init__(self, env, success_term: str):
        self.u = env.unwrapped
        params = self.u.reward_manager.get_term_cfg(success_term).params
        self.obj, self.cont = self.u.scene[params["object_cfg"].name], self.u.scene[params["container_cfg"].name]

    def _raw(self) -> torch.Tensor:
        origins = self.u.scene.env_origins
        obj_pos, goal = self.obj.data.root_pos_w - origins, self.cont.data.root_pos_w - origins
        return torch.cat([obj_pos, self.obj.data.root_quat_w, self.obj.data.root_lin_vel_w, goal, obj_pos - goal], dim=-1)

    @torch.no_grad()
    def __call__(self) -> np.ndarray:
        """(N, PRIV_DIM) float32 for the current sim state, sanitized: NaN/inf -> 0 and clipped to POS_CLIP / VEL_CLIP, so a
        physics glitch cannot send the critic a wild input. Envs with a bad raw state are listed in `bad_envs` / `raw_bad`."""
        out = self._raw().float().cpu().numpy()
        lim = np.array([POS_CLIP] * 3 + [1.0] * 4 + [VEL_CLIP] * 3 + [POS_CLIP] * 3 + [2 * POS_CLIP] * 3, np.float32)
        bad = ~np.isfinite(out).all(-1) | (np.abs(np.nan_to_num(out, nan=0.0, posinf=1e9, neginf=-1e9)) > lim).any(-1)
        self.bad_envs, self.raw_bad = np.flatnonzero(bad), out[bad]
        self.nonfinite_envs = np.flatnonzero(~np.isfinite(out).all(-1))
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -lim, lim)


class MultiTaskPrivileged(SimEvalsPrivileged):
    """`SimEvalsPrivileged` for the MultiTask env: each env's object / container are those of its own task (env i runs task
    i % num_tasks), and a one-hot of the task is appended (PRIV_DIM + num_tasks floats) so the critic knows which task it is."""

    def __init__(self, env, success_term: str):
        from polaris.environments.simeval_droid import task_ids

        self.u = env.unwrapped
        specs = self.u.reward_manager.get_term_cfg(success_term).params["specs"]
        self.pairs = [(self.u.scene[s["obj"]], self.u.scene[s["cont"]]) for s in specs]
        self.tid = task_ids(self.u, len(specs))
        self.dim = PRIV_DIM + len(specs)

    def _raw(self) -> torch.Tensor:
        origins = self.u.scene.env_origins
        out = None
        for k, (obj, cont) in enumerate(self.pairs):
            obj_pos, goal = obj.data.root_pos_w - origins, cont.data.root_pos_w - origins
            row = torch.cat([obj_pos, obj.data.root_quat_w, obj.data.root_lin_vel_w, goal, obj_pos - goal], dim=-1)
            out = row if out is None else torch.where((self.tid == k)[:, None], row, out)
        return out

    def __call__(self) -> np.ndarray:
        base = super().__call__()
        onehot = np.eye(len(self.pairs), dtype=np.float32)[self.tid.cpu().numpy()]
        return np.concatenate([base, onehot], axis=1)
