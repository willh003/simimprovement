#!/usr/bin/env bash
#SBATCH --account=weirdlab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
# Submitted by run_experiment.py. Env: EXP_DIR, TRAIN_ARGS (shell-quoted), PI_MODE (wait|none), PI_JOB_ID (optional, cancelled on exit).
set -uo pipefail
cd /gscratch/weirdlab/will/polaris
[[ -n "${PI_JOB_ID:-}" ]] && trap 'scancel "$PI_JOB_ID"' EXIT
POLICY_ARGS=""
if [[ "$PI_MODE" == "wait" ]]; then
  for _ in $(seq 360); do [[ -s "$EXP_DIR/pi_server.addr" ]] && break; sleep 10; done   # up to 1h
  [[ -s "$EXP_DIR/pi_server.addr" ]] || { echo "pi server never became ready"; exit 1; }
  ADDR=$(cat "$EXP_DIR/pi_server.addr")
  POLICY_ARGS="--policy.host ${ADDR%%:*} --policy.port ${ADDR##*:}"
  echo "using pi server $ADDR"
fi
scripts/cluster/apptainer_exec.sh "/isaac-sim/python.sh scripts/train_steering.py $TRAIN_ARGS $POLICY_ARGS"
