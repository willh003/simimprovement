# Starting an openpi policy server for `run_eval.py`

`run_eval.py` (Isaac container) is a websocket client. It needs an openpi policy server on `localhost:8000`
on the same node. The server runs on the **host** (not in the container) from its own uv env.
The client retries every 5 s, so start order doesn't matter. Custom port: pass `--port` to the server and `--remote-port` to the client.

## 1. Get onto the GPU node
Your shell may be a login node (`hostname` shows `klone-login*`, `nvidia-smi` fails). Find the job and attach:

```
squeue -u $USER -o "%i %P %j %T %N"            # note JOBID and node
srun --jobid=<JOBID> --overlap <command>        # run anything on that node
```
Apptainer shares the host network, so the container reaches the server at `localhost:8000`.

## 2. Start the server (from `third_party/openpi`)
```
cd /gscratch/weirdlab/will/polaris/third_party/openpi
nohup srun --jobid=<JOBID> --overlap bash -c '
  cd /gscratch/weirdlab/will/polaris/third_party/openpi &&
  TQDM_DISABLE=1 \
  SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
  OPENPI_DATA_HOME=/gscratch/weirdlab/will/openpi_cache \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  exec uv run --no-sync scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid_jointpos_polaris \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_droid_jointpos' \
  > /gscratch/weirdlab/will/openpi_cache/serve.log 2>&1 &
```
Ready when the log shows `server listening on 0.0.0.0:8000` (first model load takes a few minutes).
Other configs in `src/openpi/training/misc/polaris_config.py`: `pi0_fast_droid_jointpos_polaris`, `pi0_droid_jointpos_polaris`.
Checkpoints under `gs://openpi-assets/checkpoints/polaris/<config>` also exist.

## Why each piece is there (known pitfalls on this cluster)
- **`uv sync --no-install-package rerun-sdk`** (run once, before using `--no-sync`): `rerun-sdk` has no wheel for the
  host's glibc (el8, manylinux_2_28). Only lerobot pulls it in; serving doesn't use it. Plain `uv run` fails on it.
  The venv at `third_party/openpi/.venv` is already synced this way; redo it only if deps change.
- **`SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt`**: the venv Python can't verify `storage.googleapis.com` otherwise
  (`CERTIFICATE_VERIFY_FAILED`).
- **`OPENPI_DATA_HOME` under /gscratch**: the default `~/.cache/openpi` counts against the home quota. Checkpoint is already cached there.
- **`XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`**: GPU is shared with Isaac Sim. Keep <= 0.5.
- **`TQDM_DISABLE=1`**: the download progress bar otherwise floods the log with whitespace and hides errors.
- Compute nodes have internet; if one doesn't, download on a login node with the same `OPENPI_DATA_HOME`.

## 3. Verify
From the container or the node (stdlib only; the container's Isaac python has `openpi_client` but NOT `websockets`,
so `WebsocketClientPolicy` can't be imported there):
```
/isaac-sim/kit/python/bin/python3 -c "import socket;s=socket.create_connection(('localhost',8000),5);s.send(b'GET / HTTP/1.1\r\nHost: localhost:8000\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n');print(s.recv(200).split(b'\r\n')[0].decode())"
```
Healthy output: `HTTP/1.1 101 Switching Protocols`. Connection refused / timeout means the server isn't up.
Port-only check on the node: `ss -ltn | grep :8000`.

## 4. Stop / restart
```
srun --jobid=<JOBID> --overlap bash -c 'ps -u $USER -o pid,etime,cmd | grep serve_policy | grep -v grep'
srun --jobid=<JOBID> --overlap kill <pids>
```
"address already in use" on 8000 means an old server is still alive (check the ps above) — don't start a second one.
Kill both the `uv run` wrapper and the python child, or use `exec` as above so there's only one process.

## Client payload (only needed if the first request fails)
Keys: `observation/exterior_image_1_left`, `observation/wrist_image_left` (224x224), `observation/joint_position`,
`observation/gripper_position`, `prompt`. Server returns an `actions` array.
