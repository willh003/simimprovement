import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
import numpy as np

from typing import List
from pathlib import Path
from pxr import Usd, UsdPhysics

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

from .simeval_robot import ASSET_PATH, NVIDIA_DROID

DATA_PATH = ASSET_PATH

# Scenes other than 2: keywords (lowercase substrings of the USD rigid-body names) locating the object and the
# container, plus object_in_container thresholds. Thresholds are guesses; verify with the inspector script.
SCENE_TASKS = {
    "1": dict(object=("rubiks_cube",), container=("bowl",), thresholds=dict(xy_threshold=0.05), success_z_max=0.08),  # put the cube in the bowl
    # the "bin" is the rigid body small_KLT_visual_collision; its origin sits above the floor, so z_min < 0
    "3": dict(object=("banana",), container=("klt",), thresholds=dict(xy_threshold=0.08, z_min=-0.08), success_z_max=0.1),  # put banana in the bin
}


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    sphere_light = AssetBaseCfg(
        prim_path="/World/spehre",
        spawn=sim_utils.SphereLightCfg(intensity=5000),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7)),
    )

    robot = NVIDIA_DROID

    external_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.57, 0.66), rot=(-0.393, -0.195, 0.399, 0.805), convention="opengl"
        ),
    )

    external_cam_2 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam_2",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, -0.57, 0.66), rot=(0.805, 0.399, -0.195, -0.393), convention="opengl"
        ),
    )

    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.011, -0.031, -0.074), rot=(-0.420, 0.570, 0.576, -0.409), convention="opengl"
        ),
    )

    def dynamic_scene(self, scene_name: str):
        environment_path = DATA_PATH / f"scene{scene_name}.usd"
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
    joint_pos = robot.data.joint_pos[0, joint_indices]
    return joint_pos


def gripper_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = ["finger_joint"]
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[0, joint_indices]

    # rescale
    joint_pos = joint_pos / (np.pi / 4)

    return joint_pos


@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""

        arm_joint_pos = ObsTerm(func=arm_joint_pos)
        gripper_pos = ObsTerm(
            func=gripper_pos, noise=noise.GaussianNoiseCfg(std=0.05), clip=(0, 1)
        )
        external_cam = ObsTerm(
                func=mdp.observations.image,
                params={
                    "sensor_cfg": SceneEntityCfg("external_cam"),
                    "data_type": "rgb",
                    "normalize": False,
                    }
                )
        external_cam_2 = ObsTerm(
                func=mdp.observations.image,
                params={
                    "sensor_cfg": SceneEntityCfg("external_cam_2"),
                    "data_type": "rgb",
                    "normalize": False,
                    }
                )
        wrist_cam = ObsTerm(
                func=mdp.observations.image,
                params={
                    "sensor_cfg": SceneEntityCfg("wrist_cam"),
                    "data_type": "rgb",
                    "normalize": False,
                    }
                )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

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
            # "put the can in the mug": sparse reward of 1 while the can rests in the mug,
            # plus a smaller shaping reward while it is above the mug but too high to count
            self.rewards.can_in_mug = RewTerm(
                func=object_in_container,
                weight=1.0 / step_dt,
                params={ 
                    "object_cfg": SceneEntityCfg("_10_potted_meat_can"),
                    "container_cfg": SceneEntityCfg("_25_mug"),
                },
            )
            self.rewards.can_near_mug = RewTerm(
                func=object_near_container,
                weight=0.25 / step_dt,
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
            self.rewards.obj_near_container = RewTerm(
                func=object_near_container, weight=0.25 / step_dt,
                params={**params, "success_z_max": task["success_z_max"], "z_max": task["success_z_max"] + 0.09},
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
