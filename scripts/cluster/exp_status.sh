#!/usr/bin/env bash
# Usage: exp_status.sh [NAME]   (no name: list all experiments). Shows job states, pi address, and the last log lines.
D=/gscratch/weirdlab/will/polaris/runs/cluster
if [[ $# -eq 0 ]]; then ls -t "$D"; exit 0; fi
E=$D/$1
[[ -f $E/jobs.json ]] || { echo "no experiment $1"; exit 1; }
ids=$(/gscratch/weirdlab/will/venvs/ogbench/bin/python -c "import json;print(','.join(json.load(open('$E/jobs.json')).values()))")
squeue -j "$ids" -o "%.10i %.9P %.20j %.8T %.10M %R" 2>/dev/null
echo "pi server: $(cat $E/pi_server.addr 2>/dev/null || echo not ready)"
echo "--- pi.log"; tail -n 3 "$E/pi.log" 2>/dev/null
echo "--- train.log"; tail -n 8 "$E/train.log" 2>/dev/null
echo "--- latest iteration"; grep -a "^it [0-9]" "$E/train.log" 2>/dev/null | tail -n 1
