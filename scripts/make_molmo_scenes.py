"""Generate one pick-and-place scene USD per (object, container) pair of assets/molmo/manifest.json.

`isaacpy scripts/make_molmo_scenes.py` writes assets/simevals/molmo_<obj>_<cont>.usd (+ .json sidecar read by
polaris.rl.molmo_specs: name keywords, thresholds, instruction). Each scene = scene1.usd (table, light) minus the
cube/bowl, plus /World/obj_<obj> and /World/cont_<cont>: rigid-body Xforms (translate/orient, as the env code expects) with a
child Xform referencing the MolmoSpaces asset (rotated upright (or on its side with `lay_flat`), origin moved to the bottom-centre of its bbox, so the body z is
its height above the table). Nested rigid-body / articulation / PhysicsScene prims of the asset are disabled; assets without
collision shapes (or `"collision": "mesh"` in the manifest) get convex hull (object) / convex decomposition (container) colliders.
See docs/claude/current/envs_molmo.md.
"""
import argparse
import json
import os
import sys

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "assets/molmo"
SIMEVALS = Path(os.environ.get("SIM_EVALS_ASSETS", ROOT / "assets/simevals"))
OBJ_VERSION = {"thor": "20260128", "objaverse": "20260128"}


def asset_path(a: dict) -> Path:
    return MOLMO / "cache/objects" / a["source"] / OBJ_VERSION[a["source"]] / a["id"] / f"{a['id']}.usda"


def add_body(stage, name: str, a: dict, pos, role: str):
    """Create /World/<name> (rigid body) with the referenced asset below it; return the asset's bbox size in the body frame."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    body = UsdGeom.Xform.Define(stage, f"/World/{name}")
    child = UsdGeom.Xform.Define(stage, f"/World/{name}/asset")
    child.GetPrim().GetReferences().AddReference(os.path.relpath(asset_path(a), SIMEVALS))
    rot = Gf.Rotation(Gf.Vec3d(1, 0, 0), a.get("rot_x_deg", 0))
    if a.get("lay_flat"):  # long (upright) axis -> horizontal, so tall thin things do not topple at spawn
        rot = rot * Gf.Rotation(Gf.Vec3d(0, 1, 0), 90)
    if a.get("roll_deg"):  # extra roll about the (now horizontal) x axis, e.g. to rest a spatula on its flat side
        rot = rot * Gf.Rotation(Gf.Vec3d(1, 0, 0), a["roll_deg"])
    scale = a.get("scale", 1.0)
    child.AddTranslateOp()  # set below once the bbox is known
    child.AddOrientOp().Set(Gf.Quatf(rot.GetQuat()))
    child.AddScaleOp().Set(Gf.Vec3f(scale))
    # disable nested physics bodies of the asset (the body Xform above is the single rigid body)
    for p in Usd.PrimRange(child.GetPrim()):
        if p.GetTypeName() == "PhysicsScene":
            p.SetActive(False)
        else:
            for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.ArticulationRootAPI):
                if p.HasAPI(api):
                    p.RemoveAPI(api)
    meshes = [p for p in Usd.PrimRange(child.GetPrim()) if p.IsA(UsdGeom.Mesh)]
    if a.get("collision") in ("mesh", "walls"):  # replace the asset's primitive colliders
        for p in list(Usd.PrimRange(child.GetPrim())):
            if p.HasAPI(UsdPhysics.CollisionAPI) and not p.IsA(UsdGeom.Mesh):
                p.SetActive(False)
        meshes = [p for p in meshes if p.IsActive()]
    if a.get("collision") != "walls" and (a.get("collision") == "mesh" or not any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(child.GetPrim()))):
        for m in meshes:
            UsdPhysics.CollisionAPI.Apply(m)
            UsdPhysics.MeshCollisionAPI.Apply(m).CreateApproximationAttr(a.get("approx") or ("convexDecomposition" if role == "container" else "convexHull"))
    # bbox in the body frame, then shift so the origin is the bottom centre. The xy centre is that of the bottom 25% slice of the
    # mesh vertices (the "core": handles / spouts of pots, pans, mugs stick out higher up and would pull the centre off the axis)
    xc = UsdGeom.XformCache()
    pts = []
    for m in meshes:
        v = np.array(UsdGeom.Mesh(m).GetPointsAttr().Get(), dtype=float)
        if len(v):
            mat = np.array(xc.ComputeRelativeTransform(m, body.GetPrim())[0]).T  # row-vector convention -> column
            pts.append(v @ mat[:3, :3].T + mat[:3, 3])
    pts = np.concatenate(pts)
    lo, hi = pts.min(0), pts.max(0)
    core = pts[pts[:, 2] < lo[2] + 0.25 * (hi[2] - lo[2])]
    cmin, cmax = core[:, :2].min(0), core[:, :2].max(0)
    cx, cy = (cmin + cmax) / 2
    cz = (lo[2] + hi[2]) / 2 if role == "object" else lo[2]  # object origin = bbox centre, container origin = bottom centre
    child.GetOrderedXformOps()[0].Set(Gf.Vec3d(-cx, -cy, -cz))
    core_size = [float(cmax[0] - cmin[0]), float(cmax[1] - cmin[1])]
    if a.get("collision") == "walls":  # hollow box collider (floor + 4 walls of the bbox) for open containers whose meshes make bad colliders
        t = float(a.get("wall", 0.012))
        X, Y, Z = (hi - lo)
        mx, my = (lo[0] + hi[0]) / 2 - cx, (lo[1] + hi[1]) / 2 - cy
        for nm, size, c in (("floor", (X, Y, t), (mx, my, t / 2)), ("wall_px", (t, Y, Z), (mx + X / 2 - t / 2, my, Z / 2)),
                            ("wall_nx", (t, Y, Z), (mx - X / 2 + t / 2, my, Z / 2)), ("wall_py", (X - 2 * t, t, Z), (mx, my + Y / 2 - t / 2, Z / 2)),
                            ("wall_ny", (X - 2 * t, t, Z), (mx, my - Y / 2 + t / 2, Z / 2))):
            cube = UsdGeom.Cube.Define(stage, f"/World/{name}/collider_{nm}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in c]))
            cube.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
            cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    body.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], pos[2] + (hi[2] - lo[2]) / 2 if role == "object" else pos[2]))
    body.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
    bp = body.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(bp)
    mass_api = UsdPhysics.MassAPI.Apply(bp)
    mass_api.CreateMassAttr(float(a["mass"]))
    if "com_frac" in a:  # centre of mass at this fraction of the height above the bottom (a weighted base keeps tall containers from tipping)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, float(a["com_frac"]) * (hi[2] - lo[2])))
    return [hi[i] - lo[i] for i in range(3)], core_size


def make_scene(manifest: dict, okey: str, ckey: str):
    from pxr import Sdf, Usd, UsdGeom

    out = SIMEVALS / f"molmo_{okey}_{ckey}.usd"
    layer = Sdf.Layer.CreateNew(str(out) + ".tmp.usd")
    layer.TransferContent(Sdf.Layer.FindOrOpen(str(SIMEVALS / "scene1.usd")))
    stage = Usd.Stage.Open(layer)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    tr = cache.ComputeWorldBound(stage.GetPrimAtPath("/World/table")).ComputeAlignedRange()
    table_top = tr.GetMax()[2]
    poses = {}
    for n in ("rubiks_cube", "_24_bowl"):  # reuse scene 1's layout
        poses[n] = stage.GetPrimAtPath(f"/World/{n}").GetAttribute("xformOp:translate").Get()
        stage.RemovePrim(f"/World/{n}")
    z = table_top + manifest.get("table_clearance", 0.003)
    obj, cont = manifest["objects"][okey], manifest["containers"][ckey]
    osize, ocore = add_body(stage, f"obj_{okey}", obj, (poses["rubiks_cube"][0], poses["rubiks_cube"][1], z), "object")
    csize, ccore = add_body(stage, f"cont_{ckey}", cont, (poses["_24_bowl"][0], poses["_24_bowl"][1], z), "container")
    layer.Save()
    os.replace(str(out) + ".tmp.usd", out)
    # reset offsets (x, y) shared by object and container: default +-0.15 / +-0.2, shrunk so each footprint stays on the table
    ranges = []
    for ax, lim in ((0, 0.15), (1, 0.2)):
        lo_r, hi_r = -lim, lim
        for pos, size in ((poses["rubiks_cube"], osize), (poses["_24_bowl"], csize)):
            half = 0.5 * max(size[:2]) + 0.03
            lo_r, hi_r = max(lo_r, tr.GetMin()[ax] + half - pos[ax]), min(hi_r, tr.GetMax()[ax] - half - pos[ax])
        ranges.append([round(lo_r, 3), round(hi_r, 3)])
    fmt = lambda s: tuple(round(v, 3) for v in s)
    print(f"{out.name}: table_top={table_top:.3f} object size={fmt(osize)} container size={fmt(csize)}")
    spec = dict(
        id=f"Molmo-{okey}-{ckey}", scene=f"molmo_{okey}_{ckey}",
        object=[f"obj_{okey}"], container=[f"cont_{ckey}"],
        # object origin = its bbox centre, container origin = its bottom centre: inside = near the container axis, above its floor, below the rim + a fraction of the object's thinnest dimension (0.6 for shallow containers, 0.25 for deep ones)
        thresholds=dict(xy_threshold=round(0.48 * min(ccore), 3), z_min=0.0, max_speed=0.1),
        # centre must be below rim + a fraction of the object's thinnest dimension (an object resting on a rim has its centre >= rim + half of it)
        success_z_max=round(csize[2] + (0.6 if csize[2] < 0.12 else 0.25) * min(osize), 3),
        min_dist=round(0.5 * (max(osize[:2]) + max(ccore)) + 0.05, 3),
        x_range=ranges[0], y_range=ranges[1],
        instruction=f"put the {obj['category']} in the {cont['category']}",
        object_size=fmt(osize), container_size=fmt(csize),
    )
    (SIMEVALS / f"molmo_{okey}_{ckey}.json").write_text(json.dumps(spec, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", help="obj:cont (default: all pairs of the manifest)")
    a = ap.parse_args()
    from isaaclab.app import AppLauncher

    AppLauncher(headless=True)
    manifest = json.loads((MOLMO / "manifest.json").read_text())
    pairs = [p.split(":") for p in a.pairs] if a.pairs else manifest["pairs"]
    for o, c in pairs:
        make_scene(manifest, o, c)
    sys.stdout.flush()
    os._exit(0)  # the Isaac app hangs on a normal exit
