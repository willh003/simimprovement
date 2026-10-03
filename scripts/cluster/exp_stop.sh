#!/usr/bin/env bash
# Usage: exp_stop.sh NAME   cancels the experiment's pi server and training jobs.
E=/gscratch/weirdlab/will/polaris/runs/cluster/$1
scancel $(/gscratch/weirdlab/will/venvs/ogbench/bin/python -c "import json;print(' '.join(json.load(open('$E/jobs.json')).values()))") && echo "cancelled $1"
