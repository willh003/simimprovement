# openpi policy server (pi0.5 DROID) and its batched API

Training and eval clients (`train_steering.py`, `run_pi.py`, `run_pi_parallel.py`, `run_eval.py`) are websocket clients of an openpi
server. It runs on the **host** (not in the apptainer) from `third_party/openpi`'s own uv env. Apptainer shares the host network, so the
container reaches a same-node server at `localhost:8000`. Clients retry every 5 s, so start order doesn't matter. Custom port: `--port`
on the server, `--policy.port` (or `--remote-port` for `run_eval.py`) on the client. For slurm runs the launcher handles all of this
(`running.md`); this page is for manual starts.

## Start / stop: `scripts/start_pi_server.sh [JOBID] [--mem FRACTION] [--stop]`
With a JOBID it runs the server on that job's node via `srun --jobid=<JOBID> --overlap` (find it with
`squeue -u $USER -o "%i %P %j %T %N"`); without, on the current node. `--mem` caps the JAX GPU fraction (default 1.0; use <= 0.5 when
sharing the GPU with Isaac, 0.25 for 128 envs + Isaac on one L40: 0.4 OOMs Isaac). Env overrides: `POLICY_CONFIG`, `POLICY_DIR`, `PORT`.
Ready when the log (`/gscratch/weirdlab/will/openpi_cache/serve.log`) shows `server listening on 0.0.0.0:8000`; the first model load takes a few minutes.

Equivalent manual command (from `third_party/openpi`):
```
TQDM_DISABLE=1 SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt OPENPI_DATA_HOME=/gscratch/weirdlab/will/openpi_cache \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 exec uv run --no-sync scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid_jointpos_polaris --policy.dir=gs://openpi-assets/checkpoints/pi05_droid_jointpos
```
Other configs are in `src/openpi/training/misc/polaris_config.py` (`pi0_fast_droid_jointpos_polaris`, `pi0_droid_jointpos_polaris`);
checkpoints under `gs://openpi-assets/checkpoints/polaris/<config>` also exist.

Inside the apptainer (no `/gscratch`, no sbatch) `start_pi_server.sh` fails on the read-only cache; run `.venv/bin/python scripts/serve_policy.py
--port 8000 policy:checkpoint ...` by hand with `OPENPI_DATA_HOME` on a writable dir (first start downloads ~12 GB, ~10 min).

Stop: `ps -u $USER -o pid,etime,cmd | grep serve_policy` on the node (via `srun --jobid --overlap`) and kill both the `uv run` wrapper and
the python child (or `exec` as above so there is one process). "address already in use" means an old server is alive.

## Pitfalls on this cluster
- `uv sync --no-install-package rerun-sdk` once, before using `--no-sync`: `rerun-sdk` has no wheel for the host glibc (el8). Only lerobot pulls it in. The venv is already synced this way.
- `SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt`: otherwise `storage.googleapis.com` fails with `CERTIFICATE_VERIFY_FAILED`.
- `OPENPI_DATA_HOME` under /gscratch: the default `~/.cache/openpi` counts against the home quota. On a non-slurm node /gscratch is read-only and
  `/root` has a ~5 GB quota (checkpoint is 12 GB): use a `/tmp` dir, and delete a partial `openpi-assets` dir before retrying (truncated files load as OUT_OF_RANGE).
- `TQDM_DISABLE=1`: the download progress bar floods the log.
- Compute nodes have internet; if one doesn't, download on a login node with the same `OPENPI_DATA_HOME`.
- A server on node A is not `localhost` on node B: test reachability from the node you are on.

## Verify
The container's Isaac python has `openpi_client` but not `websockets`, so use a raw handshake:
```
/isaac-sim/kit/python/bin/python3 -c "import socket;s=socket.create_connection(('localhost',8000),5);s.send(b'GET / HTTP/1.1\r\nHost: localhost:8000\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n');print(s.recv(200).split(b'\r\n')[0].decode())"
```
`HTTP/1.1 101 Switching Protocols` = healthy. On the node: `ss -ltn | grep :8000`. (The "InvalidMessage" in `serve.log` from a probe is harmless.)
`scripts/test_pi_server.sh` / `scripts/test_pi_batch.py` are test helpers.

Request keys: `observation/exterior_image_1_left`, `observation/wrist_image_left` (224x224), `observation/joint_position`,
`observation/gripper_position`, `prompt`, optional `noise` `(15, 32)`. The reply has `actions` `(15, 8)`.

## Batched inference (added in `third_party/openpi`)
`Policy.infer` handles one obs, so a pre-batched obs fails inside the per-sample transforms. Batched path:
- `Policy.infer_batch(obs_list)` (`src/openpi/policies/policy.py`): input transforms per sample -> stack -> ONE `sample_actions` call ->
  output transform per sample. Optional per-obs `"noise"` `(horizon, dim)`, all or none.
- Server: message `{"batch": [obs, ...]}` -> reply `{"batch": [result, ...]}`; single-obs messages are unchanged.
- Client: `WebsocketClientPolicy.infer_batch(obs_list, max_batch_size=None) -> list[dict]` (splits into several messages when `max_batch_size` is set).
- `scripts/run_pi_parallel.py` / the trainer make one `infer_batch` call per chunk step over one connection.

Operating rules (numbers in `../experiments/openpi_batching_bench.md`):
- Throughput is flat at ~17-20 obs/s from N~8 up (compute bound); bigger batches only add latency. One call of 256 fits an L40 at memory fraction 0.5; 384+ OOMs. Use `max_batch_size` to split.
- JAX recompiles per batch size (20-100+ s): keep the sub-batch size fixed (`--server-batch-size` / `--max-batch-size`), or pad to a few sizes.
- The first request compiles for minutes and blocks the event loop, so clients must disable websocket keepalive (`ping_interval=None`). The vendored
  `WebsocketClientPolicy` does so only when `websockets >= 13`; Isaac Sim bundles 12.0, whose sync client sends no keepalive pings.
