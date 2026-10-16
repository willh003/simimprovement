import os
import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
import numpy as np

from typing import List
from pathlib import Path
from pxr import Sdf, Usd, UsdPhysics

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.utils import configclass, noise
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg

from polaris.rl.tasks import MULTITASK_ALL_TASKS, MULTITASK_OBJAVERSE_TASKS, MULTITASK_TASKS

from polaris.rl.molmo_specs import MOLMO_SPECS

from .simeval_robot import ASSET_PATH, NVIDIA_DROID

DATA_PATH = ASSET_PATH

# Scenes other than 2: keywords (lowercase substrings of the USD rigid-body names) locating the object and the
# container, plus object_in_container thresholds. Thresholds are guesses; verify with the inspector script.
SCENE_TASKS = {
    "1": dict(object=("rubiks_cube",), container=("bowl",), thresholds=dict(xy_threshold=0.05), success_z_max=0.08),  # put the cube in the bowl
    # the "bin" is the rigid body small_KLT_visual_collision; its origin sits above the floor, so z_min < 0
    "3": dict(object=("banana",), container=("klt",), thresholds=dict(xy_threshold=0.08, z_min=-0.08), success_z_max=0.1),  # put banana in the bin
}
for _spec in MOLMO_SPECS.values():  # generated MolmoSpaces scenes, keyed by scene name (docs/claude/current/envs_molmo.md)
    SCENE_TASKS[_spec["scene"]] = dict(object=tuple(_spec["object"]), container=tuple(_spec["container"]),
                                       thresholds=_spec["thresholds"], success_z_max=_spec["success_z_max"])

MULTI_SCENE = "_multi"  # dynamic_scene name of the combined multitask scene (assets/simevals/scene_multi.usd)
# MultiTask: per task, rigid-body name keywords + object_in_container thresholds (same values as the single-task envs) and the
# minimum xy distance between object and container centres at reset (object / container footprints + margin)
MULTITASK_SPECS = {
    "CubeBowl": dict(object=("rubiks_cube",), container=("bowl",), thresholds=SCENE_TASKS["1"]["thresholds"],
                     success_z_max=SCENE_TASKS["1"]["success_z_max"], min_dist=0.16),
    "CanMug": dict(object=("potted_meat",), container=("mug",), thresholds={}, success_z_max=0.06, min_dist=0.16),  # defaults of set_scene("2")
    "BananaBin": dict(object=("banana",), container=("klt",), thresholds=SCENE_TASKS["3"]["thresholds"],
                      success_z_max=SCENE_TASKS["3"]["success_z_max"], min_dist=0.30),
}
# prims copied from sim-evals scenes 2 and 3 into the combined scene (scene 1 = table, light, cube, bowl is the base)
MULTI_EXTRA_PRIMS = {"2": ["_25_mug", "_10_potted_meat_can"], "3": ["_11_banana", "small_KLT_visual_collision"]}


def _build_scene_file(scene_name: str, base: Path, copies) -> Path:
    """Build `scene<scene_name>.usd` in DATA_PATH if missing: a copy of `base` plus the prim specs `copies` = [(source usd, prim name)],
    copied prim-spec by prim-spec (`Sdf.CopySpec`) so physics / collision overrides are kept. Written next to the sources because the
    specs hold relative asset paths; built under a temp name and renamed, so concurrent runs never see a partial file."""
    path = DATA_PATH / f"scene{scene_name}.usd"
    if path.exists():
        return path
    tmp = DATA_PATH / f"scene{scene_name}.{os.getpid()}.tmp.usd"
    layer = Sdf.Layer.CreateNew(str(tmp))
    layer.TransferContent(Sdf.Layer.FindOrOpen(str(base)))
    for src_usd, name in copies:
        if not Sdf.CopySpec(Sdf.Layer.FindOrOpen(str(src_usd)), f"/World/{name}", layer, f"/World/{name}"):
            raise RuntimeError(f"could not copy /World/{name} from {Path(src_usd).name}")
    layer.Save()
    os.replace(tmp, path)
    return path


def ensure_multitask_scene() -> Path:
    """Build assets/simevals/scene_multi.usd if missing: scene 1 (table, light, cube, bowl) plus the mug and can of scene 2
    and the banana and bin of scene 3."""
    copies = [(DATA_PATH / f"scene{scene}.usd", name) for scene, names in MULTI_EXTRA_PRIMS.items() for name in names]
    return _build_scene_file(MULTI_SCENE, DATA_PATH / "scene1.usd", copies)


MULTI_MOLMO_SCENE = "_multi_molmo"  # dynamic_scene name of the sim-evals + objaverse Molmo combined scene (assets/simevals/scene_multi_molmo.usd)
MULTI_ALL_SCENE = "_multi_all"  # sim-evals + every Molmo env (thor and objaverse), assets/simevals/scene_multi_all.usd
MULTI_MOLMO_SCENES = {MULTI_MOLMO_SCENE: MULTITASK_OBJAVERSE_TASKS, MULTI_ALL_SCENE: [t for t in MULTITASK_ALL_TASKS if t in MOLMO_SPECS]}


def ensure_multitask_molmo_scene(scene_name: str = MULTI_MOLMO_SCENE) -> Path:
    """Build assets/simevals/scene<scene_name>.usd if missing: `scene_multi.usd` (table, light, the 3 sim-evals tasks' 6 objects) plus
    the object / container rigid bodies (`/World/obj_*`, `/World/cont_*`) of the Molmo envs `MULTI_MOLMO_SCENES[scene_name]`, copied
    from their `molmo_*.usd` scenes (containers shared by several envs, e.g. cont_pot2, are copied once)."""
    path = DATA_PATH / f"scene{scene_name}.usd"
    if path.exists():
        return path
    copies = {}  # prim name -> source usd (dict: shared bodies are copied once)
    for env_id in MULTI_MOLMO_SCENES[scene_name]:
        spec = MOLMO_SPECS[env_id]
        for name in spec["object"] + spec["container"]:
            copies.setdefault(name, DATA_PATH / f"{spec['scene']}.usd")
    return _build_scene_file(scene_name, ensure_multitask_scene(), [(usd, name) for name, usd in copies.items()])


def _camera(prim_path: str, focal_length: float, pos, rot) -> CameraCfg:
    """720p pinhole RGB camera (DROID intrinsics); `rot` is an opengl-convention quaternion (w, x, y, z)."""
    return CameraCfg(
        prim_path=prim_path,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="opengl"),
    )


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    sphere_light = AssetBaseCfg(
        prim_path="/World/spehre",
        spawn=sim_utils.SphereLightCfg(intensity=5000),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7)),
    )

    robot = NVIDIA_DROID

    external_cam = _camera("{ENV_REGEX_NS}/external_cam", 2.1, pos=(0.05, 0.57, 0.66), rot=(-0.393, -0.195, 0.399, 0.805))

    external_cam_2 = _camera("{ENV_REGEX_NS}/external_cam_2", 2.1, pos=(0.05, -0.57, 0.66), rot=(0.805, 0.399, -0.195, -0.393))

    wrist_cam = _camera("{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam", 2.8, pos=(0.011, -0.031, -0.074), rot=(-0.420, 0.570, 0.576, -0.409))

    def add_walls(self, x=(-0.8, 0.9), y=(-0.62, 0.62), z=(-0.5, 1.4), thickness=0.05, color=(0.12, 0.12, 0.12),
                  light_intensity=60000.0, emissive=(0.0, 0.0, 0.0), left_wall=True):
        """Enclose each env (robot, table and both external cameras) in a visual-only box room (4 walls, open top) so a
        camera cannot see the neighbouring envs. Bounds are env-local metres (robot base at the origin, table x 0.2-0.9,
        y -0.48-0.52, external cams at y=+-0.57); the walls' inner faces lie on the bounds. The front wall touches the
        table's front edge, the side walls sit just behind the external cams (~0.1 m from the table), the back wall is behind the robot. `left_wall=False` drops the +y wall
        (right behind external_cam; fuzzy in that cam, but the v2 setup with it performed best, see docs/claude/experiments/walled_envs.md)."""
        (x0, x1), (y0, y1), (z0, z1) = x, y, z
        cx, cy, cz, lx, ly, lz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2, x1 - x0, y1 - y0, z1 - z0
        t = thickness
        walls = {  # name: (centre, size)
            "wall_front": ((x1 + t / 2, cy, cz), (t, ly + 2 * t, lz)),
            "wall_back": ((x0 - t / 2, cy, cz), (t, ly + 2 * t, lz)),
            **({"wall_left": ((cx, y1 + t / 2, cz), (lx, t, lz))} if left_wall else {}),
            "wall_right": ((cx, y0 - t / 2, cz), (lx, t, lz)),
        }
        # the walls block the shared dome light, so each room gets its own light above the table (else the table renders black)
        self.room_light = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/room_light",
            spawn=sim_utils.SphereLightCfg(intensity=light_intensity, radius=0.3),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 1.2)),
        )
        for name, (pos, size) in walls.items():
            setattr(self, name, AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                spawn=sim_utils.CuboidCfg(
                    size=size, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, emissive_color=emissive, roughness=0.9)
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            ))

    def dynamic_scene(self, scene_name: str):
        if scene_name == MULTI_SCENE:
            ensure_multitask_scene()
        elif scene_name in MULTI_MOLMO_SCENES:
            ensure_multitask_molmo_scene(scene_name)
        environment_path = DATA_PATH / (f"{scene_name}.usd" if scene_name.startswith("molmo_") else f"scene{scene_name}.usd")
        scene = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/scene",
                spawn = sim_utils.UsdFileCfg(
                    usd_path=str(environment_path),
                    ),
                )
        self.scene = scene

        stage = Usd.Stage.Open(
            str(environment_path)
        )
        scene_prim = stage.GetPrimAtPath("/World")
        children = scene_prim.GetChildren()

        for child in children:
            # if rigid body
            if not UsdPhysics.RigidBodyAPI(child):
                continue

            name = child.GetName()
            print(f"Found rigid body: {name}")
            pos = child.GetAttribute("xformOp:translate").Get()
            rot = child.GetAttribute("xformOp:orient").Get()
            rot = (rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2])
            asset = RigidObjectCfg(
                        prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                        spawn=None,
                        init_state=RigidObjectCfg.InitialStateCfg(
                            pos=pos,
                            rot=rot,
                        ),
                    )
            setattr(self, name, asset)


class BinaryJointPositionZeroToOneAction(BinaryJointPositionAction):
    # override
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # compute the binary mask
        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 0
        else:
            # true: close, false: open
            binary_mask = actions > 0.5
        # compute the command
        self._processed_actions = torch.where(
            binary_mask, self._close_command, self._open_command
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`BinaryJointPositionAction` for more details.
    """

    class_type = BinaryJointPositionZeroToOneAction

@configclass
class ActionCfg:
    body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_default_offset=False,
    )

    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr = {"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )

def arm_joint_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    # get joint inidices
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[:, joint_indices]
    return joint_pos


def gripper_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = ["finger_joint"]
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[:, joint_indices]

    # rescale
    joint_pos = joint_pos / (np.pi / 4)

    return joint_pos


def _image_obs(sensor: str) -> ObsTerm:
    return ObsTerm(func=mdp.observations.image, params={"sensor_cfg": SceneEntityCfg(sensor), "data_type": "rgb", "normalize": False})


@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""

        arm_joint_pos = ObsTerm(func=arm_joint_pos)
        gripper_pos = ObsTerm(
            func=gripper_pos, noise=noise.GaussianNoiseCfg(std=0.05), clip=(0, 1)
        )
        external_cam = _image_obs("external_cam")
        external_cam_2 = _image_obs("external_cam_2")
        wrist_cam = _image_obs("wrist_cam")

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

def randomize_pair_xy(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    container_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (-0.15, 0.15),
    y_range: tuple[float, float] = (-0.2, 0.2),
    yaw_range: tuple[float, float] = (-3.14159, 3.14159),
    min_dist: float = 0.15,
):
    """Reset object and container to their default pose plus a uniform random xy offset (and a random object yaw).

    Offsets are resampled until the two are at least `min_dist` apart (xy), so they never overlap.
    """
    n = len(env_ids)
    dev = env.device

    def uni(rng, shape):
        return torch.rand(shape, device=dev) * (rng[1] - rng[0]) + rng[0]

    obj, cont = env.scene[object_cfg.name], env.scene[container_cfg.name]
    obj_state = obj.data.default_root_state[env_ids].clone()
    cont_state = cont.data.default_root_state[env_ids].clone()
    d_obj = torch.stack([uni(x_range, (n,)), uni(y_range, (n,))], dim=-1)
    d_cont = torch.stack([uni(x_range, (n,)), uni(y_range, (n,))], dim=-1)
    for _ in range(100):
        dist = torch.linalg.norm((obj_state[:, :2] + d_obj) - (cont_state[:, :2] + d_cont), dim=-1)
        bad = dist < min_dist
        if not bad.any():
            break
        k = int(bad.sum())  # resample only the overlapping envs
        d_obj[bad] = torch.stack([uni(x_range, (k,)), uni(y_range, (k,))], dim=-1)
        d_cont[bad] = torch.stack([uni(x_range, (k,)), uni(y_range, (k,))], dim=-1)
    else:
        raise RuntimeError("could not sample non-overlapping object/container offsets")
    origins = env.scene.env_origins[env_ids]
    obj_state[:, :2] += d_obj
    cont_state[:, :2] += d_cont
    yaw = uni(yaw_range, (n,))
    obj_state[:, 3:7] = _yaw_times(yaw, obj_state[:, 3:7])  # yaw about world z
    for asset, state in ((obj, obj_state), (cont, cont_state)):
        asset.write_root_pose_to_sim(torch.cat([state[:, :3] + origins, state[:, 3:7]], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:13]), env_ids=env_ids)


def task_ids(env, num_tasks: int = len(MULTITASK_TASKS)) -> torch.Tensor:
    """(num_envs,) task index of each env of the MultiTask env: env i runs task i % num_tasks (an even split when divisible)."""
    ids = getattr(env, "_multitask_ids", None)
    if ids is None:
        ids = env._multitask_ids = torch.arange(env.num_envs, device=env.device) % num_tasks
    return ids


def _yaw_times(yaw: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Quaternions (w, x, y, z): rotation by `yaw` about world z applied after `q`."""
    w1, x1, y1, z1 = torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)
    w2, x2, y2, z2 = q.unbind(-1)
    return torch.stack(
        [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dim=-1)


def randomize_multitask(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    specs: list[dict],
    x_range: tuple[float, float] = (0.30, 0.62),
    y_range: tuple[float, float] = (-0.30, 0.30),
    park: tuple[float, float, float] = (-2.0, -1.0, -5.0),
):
    """MultiTask reset: each env's task object and container get uniform random xy anywhere in the workspace box (env-local
    metres) and random yaw, at least `spec["min_dist"]` apart. The other tasks' objects are parked far below and behind the robot, out of
    every camera's view (they fall away from there; the next reset parks them again)."""
    dev = env.device
    uni = lambda rng, shape: torch.rand(shape, device=dev) * (rng[1] - rng[0]) + rng[0]
    tid = task_ids(env, len(specs))[env_ids]
    names = list(dict.fromkeys(n for s in specs for n in (s["obj"], s["cont"])))  # containers can be shared by several tasks

    def draw(name, s, idx):
        """xy for `name` in envs `idx`: a spec with x_off / y_off (Molmo tasks) samples around the body's default xy (offsets that keep it
        on the table, as in `randomize_pair_xy`), otherwise uniform in the workspace box."""
        k = len(idx)
        if "x_off" in s:
            return assets[name].data.default_root_state[idx, :2] + torch.stack([uni(s["x_off"], (k,)), uni(s["y_off"], (k,))], dim=-1)
        return torch.stack([uni(x_range, (k,)), uni(y_range, (k,))], dim=-1)

    assets = {n: env.scene[n] for n in names}
    pose = {n: assets[n].data.default_root_state[env_ids, :7].clone() for n in names}
    for j, n in enumerate(names):  # park everything, then place each task's pair in its own envs
        pose[n][:, :3] = torch.tensor([park[0], park[1] + 0.4 * j, park[2]], device=dev)
    for k, s in enumerate(specs):
        sel = (tid == k).nonzero(as_tuple=True)[0]
        if len(sel) == 0:
            continue
        ids = env_ids[sel]
        a, b = draw(s["obj"], s, ids), draw(s["cont"], s, ids)
        for _ in range(200):
            bad = torch.linalg.norm(a - b, dim=-1) < s["min_dist"]
            if not bad.any():
                break
            a[bad], b[bad] = draw(s["obj"], s, ids[bad]), draw(s["cont"], s, ids[bad])  # resample only the overlapping envs
        else:
            raise RuntimeError(f"could not sample object/container positions {s['min_dist']} m apart")
        for n, xy in ((s["obj"], a), (s["cont"], b)):
            default = assets[n].data.default_root_state[env_ids[sel], :7]  # keeps each object's resting height / upright orientation
            pose[n][sel, :2], pose[n][sel, 2] = xy, default[:, 2]
            yaw = uni((-3.14159, 3.14159), (len(sel),))
            if n == s["cont"] and not s.get("cont_yaw", True):  # Molmo containers keep their orientation (handles must stay on the table)
                yaw = torch.zeros_like(yaw)
            pose[n][sel, 3:7] = _yaw_times(yaw, default[:, 3:7])
    origins = env.scene.env_origins[env_ids]
    for n in names:
        assets[n].write_root_pose_to_sim(torch.cat([pose[n][:, :3] + origins, pose[n][:, 3:7]], dim=-1), env_ids=env_ids)
        assets[n].write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=dev), env_ids=env_ids)


@configclass
class CommandsCfg:
    """Command terms for the MDP."""


def object_in_container(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    container_cfg: SceneEntityCfg,
    xy_threshold: float = 0.02,
    z_min: float = 0.0,
    z_max: float = 0.06,
    max_speed: float = 0.05,
) -> torch.Tensor:
    """Sparse success reward: 1.0 if the object rests inside the container, else 0.0.

    The object counts as inside when its origin is within `xy_threshold` (m) of the
    container origin horizontally, between `z_min` and `z_max` (m) above the container
    origin, and (nearly) at rest (linear speed below `max_speed` m/s) so that
    mid-air passes over the container don't count.
    """
    obj = env.scene[object_cfg.name]
    cont = env.scene[container_cfg.name]
    rel = obj.data.root_pos_w - cont.data.root_pos_w
    xy_ok = torch.linalg.norm(rel[:, :2], dim=-1) < xy_threshold
    z_ok = (rel[:, 2] > z_min) & (rel[:, 2] < z_max)
    still = torch.linalg.norm(obj.data.root_lin_vel_w, dim=-1) < max_speed
    return (xy_ok & z_ok & still).float()


def multitask_in_container(env: ManagerBasedRLEnv, specs: list[dict]) -> torch.Tensor:
    """`object_in_container` of each env's own task: `specs[k]` = dict(obj, cont, z_max, <thresholds>) for task k (see `task_ids`)."""
    tid = task_ids(env, len(specs))
    out = torch.zeros(env.num_envs, device=env.device)
    for k, s in enumerate(specs):
        kwargs = {key: v for key, v in s.items() if key not in ("obj", "cont", "min_dist", "x_off", "y_off", "cont_yaw")}
        r = object_in_container(env, SceneEntityCfg(s["obj"]), SceneEntityCfg(s["cont"]), **kwargs)
        out = torch.where(tid == k, r, out)
    return out


def object_near_container(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    container_cfg: SceneEntityCfg,
    xy_threshold: float = 0.02,
    z_min: float = 0.0,
    z_max: float = 0.15,
    success_z_max: float = 0.06,
    max_speed: float = 0.05,
) -> torch.Tensor:
    """Shaping reward: 1.0 if the object is over/in the container within the looser `z_max`
    but NOT yet in the success region (`success_z_max`), else 0.0.

    Exclusive with `object_in_container(z_max=success_z_max)`, so the shaped weight is added
    only on top of nothing and the success reward stays exactly 1.
    """
    kwargs = dict(
        object_cfg=object_cfg, container_cfg=container_cfg,
        xy_threshold=xy_threshold, z_min=z_min, max_speed=max_speed,
    )
    near = object_in_container(env, z_max=z_max, **kwargs)
    success = object_in_container(env, z_max=success_z_max, **kwargs)
    return near * (1.0 - success)


def gripper_table_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """1.0 if any gripper link feels more than `threshold` N from the (filtered) table, else 0.0.

    The sensor must set `filter_prim_paths_expr` to the table collider so that contact
    with the can/mug is not counted.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.force_matrix_w  # (num_envs, num_bodies, num_filters, 3)
    return (torch.linalg.norm(forces, dim=-1) > threshold).any(dim=-1).any(dim=-1).float()


@configclass
class RewardsCfg:
    """Reward terms for the MDP. Scene-specific terms are added by `EnvCfg.set_scene`."""

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class CurriculumCfg:
    """Curriculum configuration."""


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=1, env_spacing=7.0)

    observations = ObservationCfg()
    actions = ActionCfg()
    rewards = RewardsCfg()

    terminations = TerminationsCfg()
    commands = CommandsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        self.episode_length_s = 30

        self.viewer.eye = (4.5, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.decimation = 8
        self.sim.dt = 1 / (15 * 8)
        self.sim.render_interval = self.decimation

        self.sim.physx.enable_ccd = True
        self.sim.physx.gpu_temp_buffer_capacity = 2**30
        self.sim.physx.gpu_heap_capacity = 2**30
        self.sim.physx.gpu_collision_stack_size = 2**30
        self.rerender_on_reset = True

    
    def _rigid_name(self, *keywords: str) -> str:
        """The single scene rigid body whose name contains one of `keywords` (error if none/ambiguous)."""
        names = [k for k, v in vars(self.scene).items() if isinstance(v, RigidObjectCfg)]
        if not any(kw.startswith(("obj_", "cont_")) for kw in keywords):  # generated Molmo bodies (e.g. cont_mug2) never match sim-evals keywords
            names = [n for n in names if not n.startswith(("obj_", "cont_"))]
        hits = [n for n in names if any(kw in n.lower() for kw in keywords)]
        if len(hits) != 1:
            raise ValueError(
                f"expected exactly one rigid body matching {keywords}, got {hits} (all: {names}); "
                "run scripts/inspect_simevals_scene.py and add the right name to SCENE_TASKS"
            )
        return hits[0]

    def set_scene(self, scene_name: str, table_contact_penalty: bool = False):
        self.scene.dynamic_scene(scene_name)
        self.rewards = RewardsCfg()
        step_dt = self.sim.dt * self.decimation  # the reward manager scales terms by step_dt; cancel it so rewards are exactly 0/1
        if str(scene_name) == "2":
            # "put the can in the mug": sparse reward of 1 while the can rests in the mug
            self.rewards.can_in_mug = RewTerm(
                func=object_in_container,
                weight=1.0 / step_dt,
                params={ 
                    "object_cfg": SceneEntityCfg("_10_potted_meat_can"),
                    "container_cfg": SceneEntityCfg("_25_mug"),
                },
            )
        elif str(scene_name) in SCENE_TASKS:
            # same structure as scene 2; objects are found by name keyword, thresholds are UNTUNED guesses
            task = SCENE_TASKS[str(scene_name)]
            obj, cont = self._rigid_name(*task["object"]), self._rigid_name(*task["container"])
            params = {"object_cfg": SceneEntityCfg(obj), "container_cfg": SceneEntityCfg(cont), **task["thresholds"]}
            self.rewards.obj_in_container = RewTerm(
                func=object_in_container, weight=1.0 / step_dt, params={**params, "z_max": task["success_z_max"]}
            )
        if table_contact_penalty:
            # optional penalty (-1/step) while any gripper link touches the table
            self.scene.gripper_table_contact = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/(base_link|left_.*|right_.*)",
                filter_prim_paths_expr=["{ENV_REGEX_NS}/scene/table/table_01/top"],
                update_period=0.0,
            )
            self.rewards.gripper_table_contact = RewTerm(
                func=gripper_table_contact,
                weight=-1.0 / (self.sim.dt * self.decimation),
                params={"sensor_cfg": SceneEntityCfg("gripper_table_contact")},
            )




@configclass
class CanMugEnvCfg(EnvCfg):
    """sim-evals scene 2 ("put the can in the mug") with its reward terms."""

    def __post_init__(self):
        super().__post_init__()
        self.set_scene("2")


@configclass
class CubeBowlEnvCfg(EnvCfg):
    """sim-evals scene 1 ("put the cube in the bowl")."""

    def __post_init__(self):
        super().__post_init__()
        self.set_scene("1")


@configclass
class BananaBinEnvCfg(EnvCfg):
    """sim-evals scene 3 ("put banana in the bin")."""

    def __post_init__(self):
        super().__post_init__()
        self.set_scene("3")



@configclass
class MolmoEnvCfg(EnvCfg):
    """Generated MolmoSpaces pick-and-place scene (`scene` = scene name of the sidecar spec). Reset randomizes object /
    container xy like CubeBowlRandom, with the spec's min_dist."""

    scene_name: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.set_scene(self.scene_name)
        task = SCENE_TASKS[self.scene_name]
        self.events.randomize_poses = EventTerm(
            func=randomize_pair_xy,
            mode="reset",
            params={
                "object_cfg": SceneEntityCfg(self._rigid_name(*task["object"])),
                "container_cfg": SceneEntityCfg(self._rigid_name(*task["container"])),
                "min_dist": MOLMO_SPECS_BY_SCENE[self.scene_name]["min_dist"],
                "x_range": tuple(MOLMO_SPECS_BY_SCENE[self.scene_name]["x_range"]),
                "y_range": tuple(MOLMO_SPECS_BY_SCENE[self.scene_name]["y_range"]),
            },
        )


MOLMO_SPECS_BY_SCENE = {s["scene"]: s for s in MOLMO_SPECS.values()}


@configclass
class CubeBowlWalledEnvCfg(CubeBowlEnvCfg):
    """CubeBowl with each env boxed in by walls (no view of the other envs)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.add_walls()


@configclass
class CanMugWalledEnvCfg(CanMugEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.add_walls()


@configclass
class BananaBinWalledEnvCfg(BananaBinEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.add_walls()


@configclass
class CubeBowlRandomEnvCfg(EnvCfg):
    """CubeBowl with broad resets: cube and bowl each get a random xy offset (and cube yaw) from the default layout."""

    def __post_init__(self):
        super().__post_init__()
        self.set_scene("1")
        task = SCENE_TASKS["1"]
        self.events.randomize_poses = EventTerm(
            func=randomize_pair_xy,
            mode="reset",
            params={
                "object_cfg": SceneEntityCfg(self._rigid_name(*task["object"])),
                "container_cfg": SceneEntityCfg(self._rigid_name(*task["container"])),
            },
        )


@configclass
class MultiTaskEnvCfg(EnvCfg):
    """CubeBowl + CanMug + BananaBin in one scene (the CubeBowl table / light, all six objects present). Env i runs task
    i % 3; at every reset its object and container go to random places in the workspace and the other tasks' objects are
    parked out of view (`randomize_multitask`). One success term `obj_in_container` dispatches on the env's task."""

    multitask_scene: str = MULTI_SCENE
    multitask_tasks: tuple = tuple(MULTITASK_TASKS)

    def __post_init__(self):
        super().__post_init__()
        self.scene.dynamic_scene(self.multitask_scene)
        self.rewards = RewardsCfg()
        specs = []
        for name in self.multitask_tasks:
            if name in MOLMO_SPECS:  # generated Molmo env: bodies named in its sidecar, thresholds / success height / reset offsets from it
                m = MOLMO_SPECS[name]
                specs.append(dict(obj=m["object"][0], cont=m["container"][0], **m["thresholds"], z_max=m["success_z_max"],
                                  min_dist=m["min_dist"], x_off=tuple(m["x_range"]), y_off=tuple(m["y_range"]), cont_yaw=False))
                continue
            t = MULTITASK_SPECS[name]
            specs.append(dict(obj=self._rigid_name(*t["object"]), cont=self._rigid_name(*t["container"]), **t["thresholds"],
                              z_max=t["success_z_max"], min_dist=t["min_dist"]))
        step_dt = self.sim.dt * self.decimation
        self.rewards.obj_in_container = RewTerm(func=multitask_in_container, weight=1.0 / step_dt, params={"specs": specs})
        self.events.randomize_poses = EventTerm(func=randomize_multitask, mode="reset", params={"specs": specs})


@configclass
class MultiTaskWalledEnvCfg(MultiTaskEnvCfg):
    """MultiTask with each env boxed in by walls (walls v2 defaults)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.add_walls()


@configclass
class MultiTaskObjaverseEnvCfg(MultiTaskEnvCfg):
    """MultiTask over CubeBowl + CanMug + BananaBin + the objaverse Molmo envs (`MULTITASK_OBJAVERSE_TASKS`), 11 tasks in one scene
    (`ensure_multitask_molmo_scene`). Same sparse `obj_in_container` success; Molmo tasks reset like their single-task envs."""

    multitask_scene: str = MULTI_MOLMO_SCENE
    multitask_tasks: tuple = tuple(MULTITASK_TASKS) + tuple(MULTITASK_OBJAVERSE_TASKS)


@configclass
class MultiTaskObjaverseWalledEnvCfg(MultiTaskObjaverseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.add_walls()


@configclass
class MultiTaskAllEnvCfg(MultiTaskEnvCfg):
    """MultiTask over everything: CubeBowl + CanMug + BananaBin + all Molmo envs (thor and objaverse, `MULTITASK_ALL_TASKS`)."""

    multitask_scene: str = MULTI_ALL_SCENE
    multitask_tasks: tuple = tuple(MULTITASK_ALL_TASKS)


@configclass
class MultiTaskAllWalledEnvCfg(MultiTaskAllEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.add_walls()
