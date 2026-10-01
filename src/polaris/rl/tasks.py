"""Env backends for ChunkEnv: how to reset/step an env and read reward terms + success from it.

No Isaac imports, so these are testable with fake envs.
"""

from dataclasses import dataclass, field

SIM_EVALS_ENV_PREFIX = "DROID-SimEvals"
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
    return env_id.startswith(SIM_EVALS_ENV_PREFIX)


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
