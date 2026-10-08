"""Print a summary of USD assets / scenes: prim tree (truncated), physics schemas, collision approximation, mass, bbox.

pxr is only importable after the (headless) Isaac app starts: `isaacpy scripts/inspect_usd.py <file.usd[a]> [...] [--depth 3]`.
Used to vet MolmoSpaces assets (assets/molmo) and sim-evals scenes (assets/simevals) before building envs.
"""
import argparse



def inspect(path: str, depth: int):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(path)
    up, mpu = UsdGeom.GetStageUpAxis(stage), UsdGeom.GetStageMetersPerUnit(stage)
    root = stage.GetDefaultPrim()
    print(f"== {path}\n   upAxis={up} metersPerUnit={mpu} defaultPrim={root.GetPath() if root else None}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    if root:
        r = cache.ComputeWorldBound(root).ComputeAlignedRange()
        lo, hi = r.GetMin(), r.GetMax()
        print(f"   bbox min={tuple(round(x, 4) for x in lo)} max={tuple(round(x, 4) for x in hi)} size={tuple(round(h - l, 4) for l, h in zip(lo, hi))}")
    n_mesh = n_col = 0
    for prim in stage.Traverse():
        d = len(prim.GetPath().pathString.split("/")) - 1
        schemas = [s for s in prim.GetAppliedSchemas() if "Physics" in s or "Collision" in s]
        if prim.IsA(UsdGeom.Mesh):
            n_mesh += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            n_col += 1
        extra = ""
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            extra += f" approx={UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()}"
        if prim.HasAPI(UsdPhysics.MassAPI):
            extra += f" mass={UsdPhysics.MassAPI(prim).GetMassAttr().Get()}"
        if d <= depth or schemas:
            print(f"   {'  ' * d}{prim.GetPath().name} <{prim.GetTypeName()}> {schemas}{extra}")
    print(f"   meshes={n_mesh} collision_prims={n_col}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--depth", type=int, default=3)
    a = ap.parse_args()
    from isaaclab.app import AppLauncher

    AppLauncher(headless=True)
    for p in a.paths:
        inspect(p, a.depth)
    import os, sys
    sys.stdout.flush()
    os._exit(0)  # the Isaac app hangs on a normal exit
