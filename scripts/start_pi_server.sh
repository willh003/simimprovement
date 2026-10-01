#!/usr/bin/env bash
# Start (or stop) an openpi policy server on the node of a running slurm job.
# See OPENPI_SERVER.md for background and pitfalls.
#
# Usage: start_pi_server.sh [JOBID] [--stop]
#   With no JOBID, the server runs on the current node.
#   env overrides: POLICY_CONFIG, POLICY_DIR, PORT, TIMEOUT (seconds to wait for ready)
set -euo pipefail

JOBID=""
MODE="start"
for arg in "$@"; do
  case "$arg" in
    --stop) MODE="--stop" ;;
    -*) echo "usage: $0 [JOBID] [--stop]" >&2; exit 2 ;;
    *) JOBID="$arg" ;;
  esac
done

OPENPI_DIR=/gscratch/weirdlab/will/polaris/third_party/openpi
CACHE_DIR=/gscratch/weirdlab/will/openpi_cache
LOG="$CACHE_DIR/serve.log"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_droid_jointpos_polaris}"
POLICY_DIR="${POLICY_DIR:-gs://openpi-assets/checkpoints/pi05_droid_jointpos}"
PORT="${PORT:-8000}"
TIMEOUT="${TIMEOUT:-900}"

# Run a command on the target node: via srun if a job was given, else locally.
if [[ -n "$JOBID" ]]; then
  RUN=(srun --jobid="$JOBID" --overlap)
else
  RUN=()
fi
on_node() { "${RUN[@]}" "$@"; }

# 1. If a job was given, it must exist and be running.
if [[ -n "$JOBID" ]]; then
  state=$(squeue -j "$JOBID" -h -o "%T" 2>/dev/null || true)
  node=$(squeue -j "$JOBID" -h -o "%N" 2>/dev/null || true)
  if [[ "$state" != "RUNNING" ]]; then
    echo "Job $JOBID is not running (state: ${state:-not found})." >&2
    exit 1
  fi
  echo "Job $JOBID running on $node"
else
  node=$(hostname)
  echo "No JOBID given; using current node $node"
fi

find_pids() {
  on_node bash -c "pgrep -u \$USER -f '[s]cripts/serve_policy.py' || true"
}

# 2. Stop mode.
if [[ "$MODE" == "--stop" ]]; then
  pids=$(find_pids)
  if [[ -z "$pids" ]]; then echo "No server running."; exit 0; fi
  echo "Killing: $pids"
  on_node kill $pids
  exit 0
fi

# 3. Don't start a second server.
existing=$(find_pids)
if [[ -n "$existing" ]]; then
  echo "A server is already running on $node (pids: $existing). Use '$0${JOBID:+ $JOBID} --stop' first." >&2
  exit 1
fi

# 4. Launch in the background, detached from this shell.
mkdir -p "$CACHE_DIR"
: > "$LOG"
setsid nohup "${RUN[@]}" bash -c "
  cd $OPENPI_DIR &&
  TQDM_DISABLE=1 \
  SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
  OPENPI_DATA_HOME=$CACHE_DIR \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  exec uv run --no-sync scripts/serve_policy.py --port $PORT policy:checkpoint \
    --policy.config=$POLICY_CONFIG \
    --policy.dir=$POLICY_DIR" \
  > "$LOG" 2>&1 < /dev/null &
SRUN_PID=$!
echo "Launched ($POLICY_CONFIG), log: $LOG"

# 5. Wait for readiness.
start=$SECONDS
while ! grep -q "server listening on" "$LOG"; do
  if ! kill -0 "$SRUN_PID" 2>/dev/null; then
    echo "Server exited before becoming ready. Last log lines:" >&2
    tail -n 30 "$LOG" >&2
    exit 1
  fi
  if (( SECONDS - start > TIMEOUT )); then
    echo "Timed out after ${TIMEOUT}s waiting for server. Last log lines:" >&2
    tail -n 30 "$LOG" >&2
    exit 1
  fi
  sleep 5
done

# 6. Verify the websocket handshake from the node (stdlib python only).
resp=$(on_node python3 -c "
import socket
s=socket.create_connection(('localhost',$PORT),5)
s.send(b'GET / HTTP/1.1\r\nHost: localhost:$PORT\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n')
print(s.recv(200).split(b'\r\n')[0].decode())
" 2>&1 || true)
echo "Handshake: $resp"
if [[ "$resp" != *"101"* ]]; then
  echo "Server logged ready but handshake failed." >&2
  exit 1
fi
echo "Server ready on $node:$PORT (stop with: $0${JOBID:+ $JOBID} --stop)"
