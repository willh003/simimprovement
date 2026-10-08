"""Install the MolmoSpaces objects listed in assets/molmo/manifest.json into assets/molmo/cache (symlinked under isaac-assets/).

Run with a python that has `molmospaces-resources` (NOT the Isaac python; a throwaway venv is enough:
`uv venv v && uv pip install --python v/bin/python molmospaces-resources==0.0.3a2`). Thor objects are installed by the
resource manager's setup itself (~3.8 GB, always); objaverse ones are fetched per UID (entries with "source": "objaverse").
Prints the resolved .usda path of every asset.
"""
import argparse
import json
from pathlib import Path

from molmospaces_resources import R2RemoteStorage, setup_resource_manager

BASE = Path(__file__).resolve().parents[1] / "assets/molmo"
VERSIONS = {"objects": {"thor": "20260128", "objaverse": "20260128"}}

rm = setup_resource_manager(R2RemoteStorage("isaac-thor-resources"), symlink_dir=BASE / "isaac-assets", versions=VERSIONS, cache_dir=BASE / "cache")
ap = argparse.ArgumentParser()
ap.add_argument("--uids", nargs="*", help="extra objaverse UIDs (bare, without the obja_ prefix) to install; skips the manifest")
args = ap.parse_args()
if args.uids:
    for uid in args.uids:
        name = f"obja_{uid}"
        path = BASE / "cache/objects/objaverse" / VERSIONS["objects"]["objaverse"] / name / f"{name}.usda"
        if not path.exists():
            rm.install_packages("objects", {"objaverse": rm.find_archives("objects", "objaverse", [f"{name}/{name}.usda"])})
        print(f"{uid}: {'ok' if path.exists() else 'MISSING'} {path.relative_to(BASE)}")
    raise SystemExit
manifest = json.loads((BASE / "manifest.json").read_text())
for group in ("objects", "containers"):
    for key, a in manifest[group].items():
        src, uid = a["source"], a["id"]
        root = BASE / "cache/objects" / src / VERSIONS["objects"][src]
        path = root / uid / f"{uid}.usda"
        if not path.exists() and src != "thor":
            archives = rm.find_archives("objects", src, [f"{uid}/{uid}.usda"])
            rm.install_packages("objects", {src: archives})
        assert path.exists(), path
        print(f"{group}/{key}: {path.relative_to(BASE)}")
