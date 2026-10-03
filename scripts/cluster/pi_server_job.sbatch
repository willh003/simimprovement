#!/usr/bin/env bash
#SBATCH --account=weirdlab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
# Submitted by run_experiment.py (partition/time/name/output given on the command line). Env: EXP_DIR.
# Starts an openpi server on a free port, then writes "<host>:<port>" to $EXP_DIR/pi_server.addr once it is ready.
set -uo pipefail
OPENPI_DIR=/gscratch/weirdlab/will/polaris/third_party/openpi
CACHE_DIR=/gscratch/weirdlab/will/openpi_cache
POLICY_CONFIG="${POLICY_CONFIG:-pi05_droid_jointpos_polaris}"
POLICY_DIR="${POLICY_DIR:-gs://openpi-assets/checkpoints/pi05_droid_jointpos}"
PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1])")
rm -f "$EXP_DIR/pi_server.addr"
cd "$OPENPI_DIR"
TQDM_DISABLE=1 SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt OPENPI_DATA_HOME=$CACHE_DIR XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run --no-sync scripts/serve_policy.py --port "$PORT" policy:checkpoint \
  --policy.config="$POLICY_CONFIG" --policy.dir="$POLICY_DIR" &
SERVER=$!
handshake() {
  timeout 30 python3 -c "
import socket
s=socket.create_connection(('localhost',$PORT),5); s.settimeout(5)
s.send(b'GET / HTTP/1.1\r\nHost: localhost:$PORT\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n')
print(s.recv(200).split(b'\r\n')[0].decode())" 2>/dev/null || true
}
until [[ "$(handshake)" == *101* ]]; do
  kill -0 $SERVER 2>/dev/null || { echo "server exited before ready"; exit 1; }
  sleep 5
done
echo "$(hostname):$PORT" > "$EXP_DIR/pi_server.addr"
echo "pi server ready at $(hostname):$PORT"
wait $SERVER
