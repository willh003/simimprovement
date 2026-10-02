"""Benchmark openpi batched inference throughput vs batch size (dummy inputs). Needs a server on --host/--port.

  third_party/openpi/.venv/bin/python scripts/bench_pi_batch.py --sizes 1 8 32 128 256 512 --out docs/claude/images/pi_batch.json
Stops at the first batch size that fails (e.g. server OOM) and records it.
"""
import argparse, functools, json, time

import numpy as np
import websockets.sync.client as _wc

_wc.connect = functools.partial(_wc.connect, ping_interval=None, ping_timeout=None)  # first call per size compiles
from openpi_client.websocket_client_policy import WebsocketClientPolicy

p = argparse.ArgumentParser()
p.add_argument("--host", default="localhost")
p.add_argument("--port", type=int, default=8000)
p.add_argument("--sizes", type=int, nargs="+", default=[1, 8, 32, 128, 256, 512])
p.add_argument("--chunk", type=int, default=None, help="max obs per message (client splits larger batches)")
p.add_argument("--reps", type=int, default=3)
p.add_argument("--out", default=None)
a = p.parse_args()

rng = np.random.default_rng(0)
img = lambda: rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
mk = lambda i: {"observation/exterior_image_1_left": img(), "observation/wrist_image_left": img(),
                "observation/joint_position": rng.normal(size=7).astype(np.float32),
                "observation/gripper_position": np.array([0.5], dtype=np.float32), "prompt": "pick up the banana"}

c = WebsocketClientPolicy(a.host, a.port)
rows = []
for n in a.sizes:
    obs = [mk(i) for i in range(n)]
    try:
        t = time.time(); c.infer_batch(obs, a.chunk); warm = time.time() - t  # includes compile for this size
        ts, infer_ms = [], []
        for _ in range(a.reps):
            t = time.time(); r = c.infer_batch(obs, a.chunk); ts.append(time.time() - t)
            infer_ms.append(r[0]["policy_timing"]["infer_ms"])
    except Exception as e:
        rows.append({"n": n, "error": str(e)[-400:]}); print(f"N={n}: FAILED {str(e)[-300:]}"); break
    tb = float(np.median(ts))
    rows.append({"n": n, "chunk": a.chunk, "call_s": tb, "obs_per_s": n / tb, "model_ms": float(np.median(infer_ms)), "warmup_s": warm})
    print(f"N={n:4d} call {tb*1000:8.0f} ms model {np.median(infer_ms):8.0f} ms  {n/tb:6.1f} obs/s  (first call {warm:.1f}s)", flush=True)
    if a.out: json.dump(rows, open(a.out, "w"), indent=1)
