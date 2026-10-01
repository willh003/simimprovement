# Smoke test: sim-evals DROID env under the container's Isaac Lab

Checks that the `DROID` env in `sim_evals` builds, resets and steps with random
actions. It does not use the policy client, so no policy server is needed.

Use `/isaac-sim/python.sh` as the interpreter. It has torch and Isaac Lab, and
the other dependencies `run_eval.py` needs are already installed in it.
`sim_evals` is added through `PYTHONPATH`, so no install is required.

## 1. Write the script

```bash
cd /workspace/polaris/third_party/sim-evals
export PYTHONPATH=$PWD/src:$PYTHONPATH
cat > /tmp/smoke.py <<'EOF'
import argparse, torch, gymnasium as gym
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a, _ = p.parse_known_args(); a.enable_cameras = True; a.headless = True
app = AppLauncher(a).app
import sim_evals.environments
from isaaclab_tasks.utils import parse_env_cfg
cfg = parse_env_cfg("DROID", device=a.device, num_envs=1, use_fabric=True)
cfg.set_scene(1)
env = gym.make("DROID", cfg=cfg)
obs, _ = env.reset(); obs, _ = env.reset()
print("OBS KEYS", {k: list(v.keys()) if hasattr(v, "keys") else type(v) for k, v in obs.items()})
for _ in range(5):
    obs, *_ = env.step(torch.tensor(env.action_space.sample()))
print("SMOKE OK")
env.close(); app.close()
EOF
```

## 2. Run it

The hard timeout keeps a hang from running forever.

```bash
timeout 600 /isaac-sim/python.sh /tmp/smoke.py 2>&1 | tee /tmp/smoke.log | grep -E "SMOKE OK|OBS KEYS|Error|Traceback"
```

Expected: an `OBS KEYS ...` line followed by `SMOKE OK`. If it fails, check
`/tmp/smoke.log` for the full traceback.

## 3. Then run the real eval

Start the policy server first (see the docstring in `run_eval.py`), then:

```bash
timeout 900 /isaac-sim/python.sh run_eval.py --episodes 1 2>&1 | tee /tmp/run.log | tail -30
```
