#!/usr/bin/env bash
# Non-interactive version of /gscratch/weirdlab/will/polaris_overlay.sh: runs a command inside the UWLab/Isaac apptainer
# (read-only overlay) after sourcing ~/polaris_setup.sh. Must run on a GPU node.
# Usage: apptainer_exec.sh "<shell command>"      e.g. apptainer_exec.sh "/isaac-sim/python.sh scripts/train_steering.py ..."
set -euo pipefail
export UWLAB_BASE=/gscratch/weirdlab/will/UWLab_Docker
OVERLAY_IMG="$UWLAB_BASE/overlay.img"
exec apptainer exec --nv \
  --fakeroot \
  --overlay "${OVERLAY_IMG}:ro" \
  --bind /gscratch/weirdlab/will/UWLab/source:/workspace/uwlab/source \
  --bind /gscratch/weirdlab/will/UWLab/scripts:/workspace/uwlab/scripts \
  --bind /gscratch/weirdlab/will/UWLab/assets:/workspace/uwlab/assets \
  --bind /gscratch/weirdlab/will/UWLab/third_party:/workspace/uwlab/third_party \
  --bind $UWLAB_BASE/isaac-cache-kit-polaris:/isaac-sim/kit/cache \
  --bind $UWLAB_BASE/isaac-sim-data:/isaac-sim/kit/data \
  --bind $UWLAB_BASE/isaac-cache-ov:/root/.cache/ov \
  --bind $UWLAB_BASE/isaac-cache-pip:/root/.cache/pip \
  --bind $UWLAB_BASE/isaac-cache-gl:/root/.cache/nvidia/GLCache \
  --bind $UWLAB_BASE/isaac-cache-compute:/root/.nv/ComputeCache \
  --bind /gscratch/weirdlab/will/polaris:/workspace/polaris \
  --bind $UWLAB_BASE/isaac-cache-torchext:/root/.cache/torch_extensions \
  --bind $UWLAB_BASE/logs:/workspace/uwlab/logs \
  --bind $UWLAB_BASE/outputs:/workspace/uwlab/outputs \
  --bind $UWLAB_BASE/data_storage:/workspace/uwlab/data_storage \
  --bind $UWLAB_BASE/tmp/isaaclab:/tmp/isaaclab \
  --bind $UWLAB_BASE/tmp:/tmp \
  --bind /gscratch/weirdlab/will/UWLab/configs:/workspace/uwlab/configs \
  --pwd /workspace/polaris \
  $UWLAB_BASE/uw-lab_latest.sif \
  bash -c "source /mmfs1/home/wph52/polaris_setup.sh && $1"
