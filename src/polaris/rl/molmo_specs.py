"""Specs of the generated MolmoSpaces pick-and-place envs (`scripts/make_molmo_scenes.py` sidecar json files).

No Isaac imports. Each assets/simevals/molmo_<obj>_<cont>.json becomes env id `Molmo-<obj>-<cont>`.
"""
import json
import os
from pathlib import Path

ASSET_PATH = Path(os.environ.get("SIM_EVALS_ASSETS", Path(__file__).resolve().parents[3] / "assets/simevals"))


def load_molmo_specs() -> dict[str, dict]:
    """env id -> spec dict (scene, object / container keywords, thresholds, success_z_max, min_dist, instruction)."""
    specs = (json.loads(p.read_text()) for p in sorted(ASSET_PATH.glob("molmo_*.json")))
    return {s["id"]: s for s in specs}


MOLMO_SPECS = load_molmo_specs()
