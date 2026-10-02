"""Check openpi batched inference against single inference and time both. Needs a server on localhost:8000.

  third_party/openpi/.venv/bin/python scripts/test_pi_batch.py
"""
import functools, websockets.sync.client as _wc
_wc.connect = functools.partial(_wc.connect, ping_interval=None)
import time, numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy
c = WebsocketClientPolicy("localhost", 8000)
rng = np.random.default_rng(0)
def mk(i):
    return {"observation/exterior_image_1_left": rng.integers(0,255,(224,224,3),dtype=np.uint8),
            "observation/wrist_image_left": rng.integers(0,255,(224,224,3),dtype=np.uint8),
            "observation/joint_position": rng.normal(size=7).astype(np.float32),
            "observation/gripper_position": np.array([0.1*i % 1], dtype=np.float32),
            "prompt": ["pick up the banana", "put the cube in the bin"][i % 2]}
B = 4
obs = [mk(i) for i in range(B)]
noise = [rng.normal(size=(15, 32)).astype(np.float32) for _ in range(B)]
for o in obs: c.infer(o)  # warmup
singles = [c.infer({**o, "noise": z}) for o, z in zip(obs, noise)]
batch = c.infer_batch([{**o, "noise": z} for o, z in zip(obs, noise)])
print("shapes", batch[0]["actions"].shape, singles[0]["actions"].shape)
print("max abs diff vs single (same noise):", max(np.abs(a["actions"]-b["actions"]).max() for a,b in zip(singles,batch)))
print("samples differ from each other:", np.abs(batch[0]["actions"]-batch[1]["actions"]).max() > 1e-3)
print("empty batch:", c.infer_batch([]))
for n in (1, 4, 8, 16, 32):
    ol = [mk(i) for i in range(n)]
    c.infer_batch(ol)  # compile for this size
    t = time.time(); c.infer_batch(ol); tb = time.time() - t
    t = time.time(); [c.infer(o) for o in ol]; ts = time.time() - t
    print(f"N={n:3d} batched {tb*1000:7.0f} ms ({n/tb:5.1f} obs/s) | sequential {ts*1000:7.0f} ms ({n/ts:5.1f} obs/s)")
