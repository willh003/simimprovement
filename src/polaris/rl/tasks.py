"""Env backends for ChunkEnv: how to reset/step an env and read reward terms + success from it.

No Isaac imports, so these are testable with fake envs.
"""

from dataclasses import dataclass, field

import numpy as np

SIM_EVALS_INSTRUCTIONS = {
    "CubeBowl": "put the cube in the bowl",
    "CanMug": "put the can in the mug",
    "BananaBin": "put banana in the bin",
}
SIM_EVALS_SUCCESS_TERMS = {  # reward term that defines success
    "CubeBowl": "obj_in_container",
    "CanMug": "can_in_mug",
    "BananaBin": "obj_in_container",
}


def is_sim_evals(env_id: str) -> bool:
    return env_id in SIM_EVALS_INSTRUCTIONS


@dataclass
class StepResult:
    obs: dict
    terms: dict[str, float]  # this env step's reward terms (summed over the chunk by ChunkEnv)
    success: bool
    reached: dict[str, float]  # per-term max-ever 0/1 flags, used for success/<name> metrics
    terminated: bool
    truncated: bool
    info: dict = field(default_factory=dict)


class RubricSplatTask:
    """ManagerBasedRLSplatEnv + Rubric: one sparse term per criterion (1/N the first time it is reached)."""

    default_success_bonus = 1.0

    def __init__(self):
        self.prev_reached: dict[str, float] = {}

    def reset(self, env, **kwargs) -> dict:
        obs, _ = env.reset(**kwargs)
        self.prev_reached = {}
        return obs

    def step(self, env, action, expensive: bool) -> StepResult:
        obs, _, term, trunc, info = env.step(action, expensive=expensive)
        prefix = "reached/"
        reached = {k[len(prefix):]: v for k, v in info["rubric"]["metrics"].items() if k.startswith(prefix)}
        terms = {name: (flag - self.prev_reached.get(name, 0.0)) / len(reached) for name, flag in reached.items()}
        self.prev_reached = reached
        return StepResult(obs, terms, bool(info["rubric"]["success"]), reached, bool(term[0]), bool(trunc[0]), info)


class SimEvalsTask:
    """sim-evals DROID env (plain Isaac render): reward terms come from the env's reward manager.

    Success = `success_term` fired (> 0) on this step. The env has no initial-condition randomization,
    so reset() takes no object poses.
    """

    default_success_bonus = 0.0  # the success term already pays 1

    def __init__(self, success_term: str = "can_in_mug"):
        self.success_term = success_term
        self.reached: dict[str, float] = {}
        self.first_reset = True

    def reset(self, env, **kwargs) -> dict:
        obs, _ = env.reset()
        if self.first_reset:
            obs, _ = env.reset()  # second render cycle loads materials correctly (as in sim-evals run_eval.py)
            self.first_reset = False
        self.reached = {}
        return obs

    def step(self, env, action, expensive: bool) -> StepResult:
        obs, rew, term, trunc, info = env.step(action)
        raw = {name: float(vals[0]) for name, vals in env.unwrapped.reward_manager.get_active_iterable_terms(0)}
        # get_active_iterable_terms is scaled differently from the env reward (by a common factor);
        # rescale so the terms sum to the env reward actually returned by step().
        total = sum(raw.values())
        scale = float(rew[0]) / total if total else 0.0
        terms = {name: v * scale for name, v in raw.items()}
        for name, v in terms.items():
            self.reached[name] = max(self.reached.get(name, 0.0), float(v > 0))
        return StepResult(obs, terms, terms.get(self.success_term, 0.0) > 0, dict(self.reached), bool(term[0]), bool(trunc[0]), info)


@dataclass
class VecStepResult:
    """One env step of N parallel envs; all arrays are (N,)."""

    obs: dict
    terms: dict[str, np.ndarray]
    success: np.ndarray  # bool
    reached: dict[str, np.ndarray]  # per-term max-ever 0/1 flags of the episode each env was in during this step
    terminated: np.ndarray  # bool; includes success when the env has a success termination
    truncated: np.ndarray  # bool (time-out)


class VecSimEvalsTask:
    """Vectorized SimEvalsTask for a ManagerBasedRLEnv with N envs (auto-reset inside env.step).

    Reward terms per env come straight from the reward manager's (N, n_terms) buffer, which holds `func * weight`;
    times `step_dt` they sum exactly to the env reward. Success = `success_term` > 0 on this step. For the episode to
    end on success the env needs a success termination (see `environments.simeval_parallel.make_parallel_env`).
    """

    default_success_bonus = 0.0

    def __init__(self, success_term: str = "can_in_mug"):
        self.success_term = success_term
        self.reached: dict[str, np.ndarray] = {}
        self.first_reset = True

    def reset(self, env) -> dict:
        obs, _ = env.reset()
        if self.first_reset:
            obs, _ = env.reset()  # second render cycle loads materials correctly
            self.first_reset = False
        self.reached = {}
        return obs

    def step(self, env, action) -> VecStepResult:
        obs, _, term, trunc, _ = env.step(action)
        u = env.unwrapped
        rm = u.reward_manager
        vals = rm._step_reward.cpu().numpy() * u.step_dt
        terms = {name: vals[:, i] for i, name in enumerate(rm.active_terms)}
        for name, v in terms.items():
            self.reached[name] = np.maximum(self.reached.get(name, 0.0), (v > 0).astype(np.float64))
        reached = {k: v.copy() for k, v in self.reached.items()}
        term, trunc = _np_bool(term), _np_bool(trunc)
        for v in self.reached.values():  # envs that just ended were auto-reset: new episode, fresh flags
            v[term | trunc] = 0.0
        return VecStepResult(obs, terms, terms[self.success_term] > 0, reached, term, trunc)

    @staticmethod
    def hold_action(obs) -> np.ndarray:
        """(N, 8) joint-position action holding the current arm pose, gripper open."""
        joint = obs["policy"]["arm_joint_pos"].detach().cpu().numpy()
        return np.concatenate([joint, np.zeros((len(joint), 1), joint.dtype)], axis=1)

    @staticmethod
    def refresh_obs(env) -> dict:
        """Re-render and recompute obs, so envs reset on this step don't show the pre-reset camera frame.

        (Isaac renders before resetting; Isaac's own `num_rerenders_on_reset` would pay this on every reset.)
        """
        u = env.unwrapped
        u.sim.render()
        for sensor in u.scene.sensors.values():
            sensor.update(0.0, force_recompute=True)
        u.obs_buf = u.observation_manager.compute()
        return u.obs_buf


def _np_bool(x) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x, dtype=bool).reshape(-1)
