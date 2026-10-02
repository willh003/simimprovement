"""Probe an openpi policy server with dummy inputs (no Isaac env): latency, batching, concurrency scaling.

    isaacpy scripts/bench_server.py --host g3115 --port 8000

Sections: (1) server metadata, (2) single-client latency (with/without noise), (3) latency vs sent image size,
(4) does the server accept a batch dimension, (5) concurrency sweep: K clients, each with its own websocket,
each sending `reps` requests; reports per-request latency and aggregate queries/s. Prints one `RESULT {json}` per row.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import tyro
from openpi_client import websocket_client_policy


@dataclass
class Args:
    host: str = "localhost"
    port: int = 8000
    single_reps: int = 20
    reps: int = 6  # requests per client in the concurrency sweep (after 2 warm-up requests)
    clients: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    noise_shape: tuple[int, int] = (15, 32)
    sizes: tuple[tuple[int, int], ...] = ((224, 224), (360, 640), (720, 1280))  # (H, W) of images sent


def make_request(rng, hw=(224, 224), noise_shape=None, batch=None):
    def img():
        shape = (batch, *hw, 3) if batch else (*hw, 3)
        return rng.integers(0, 255, shape, dtype=np.uint8)

    req = {
        "observation/exterior_image_1_left": img(),
        "observation/wrist_image_left": img(),
        "observation/joint_position": rng.standard_normal((batch, 7) if batch else 7).astype(np.float64),
        "observation/gripper_position": rng.random((batch, 1) if batch else 1).astype(np.float64),
        "prompt": "put banana in the bin",
    }
    if noise_shape is not None:
        req["noise"] = rng.standard_normal(noise_shape).astype(np.float32)
    return req


def stats(lat):
    lat = np.asarray(lat) * 1000
    return dict(mean_ms=round(float(lat.mean()), 1), p50_ms=round(float(np.percentile(lat, 50)), 1), p95_ms=round(float(np.percentile(lat, 95)), 1))


def emit(**kw):
    print("RESULT " + json.dumps(kw), flush=True)


def main(a: Args):
    rng = np.random.default_rng(0)
    client = websocket_client_policy.WebsocketClientPolicy(host=a.host, port=a.port)
    emit(section="metadata", metadata=str(client.get_server_metadata()))

    for use_noise in (False, True):
        req = make_request(rng, noise_shape=a.noise_shape if use_noise else None)
        out = client.infer(req)  # warm-up (jit compile on first call)
        lat = []
        for _ in range(a.single_reps):
            t = time.perf_counter()
            out = client.infer(req)
            lat.append(time.perf_counter() - t)
        emit(section="single", noise=use_noise, actions_shape=list(np.asarray(out["actions"]).shape),
             server_timing=out.get("server_timing"), **stats(lat))

    for hw in a.sizes:
        req = make_request(rng, hw=hw, noise_shape=a.noise_shape)
        mb = sum(v.nbytes for v in req.values() if isinstance(v, np.ndarray)) / 2**20
        client.infer(req)
        lat = []
        for _ in range(8):
            t = time.perf_counter()
            client.infer(req)
            lat.append(time.perf_counter() - t)
        emit(section="image_size", hw=list(hw), payload_mib=round(mb, 2), **stats(lat))

    for b in (2, 4):
        try:
            out = client.infer(make_request(rng, noise_shape=None, batch=b))
            emit(section="batch", batch=b, ok=True, actions_shape=list(np.asarray(out["actions"]).shape))
        except Exception as e:  # noqa: BLE001
            emit(section="batch", batch=b, ok=False, error=repr(e)[:300])
            client = websocket_client_policy.WebsocketClientPolicy(host=a.host, port=a.port)  # connection is dead after an error

    for k in a.clients:
        clients = [websocket_client_policy.WebsocketClientPolicy(host=a.host, port=a.port) for _ in range(k)]
        reqs = [make_request(rng, noise_shape=a.noise_shape) for _ in range(k)]

        def work(i):
            lat = []
            for r in range(a.reps + 2):
                t = time.perf_counter()
                clients[i].infer(reqs[i])
                if r >= 2:
                    lat.append(time.perf_counter() - t)
            return lat

        with ThreadPoolExecutor(max_workers=k) as pool:
            t0 = time.perf_counter()
            lats = list(pool.map(work, range(k)))
            wall = time.perf_counter() - t0
        flat = [x for l in lats for x in l]
        # wall includes the 2 warm-up requests per client, so queries/s is slightly pessimistic
        emit(section="concurrency", clients=k, queries_per_s=round(k * (a.reps + 2) / wall, 2), **stats(flat))
        for c in clients:
            c._ws.close()

    os._exit(0)


if __name__ == "__main__":
    main(tyro.cli(Args))
