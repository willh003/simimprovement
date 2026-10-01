import cv2
import numpy as np

from .abstract_client import InferenceClient


class Client(InferenceClient):
    """Server-free client that holds the current arm pose and keeps the gripper open.

    Useful for checking the env loop, rendering and video writing without a policy server.
    """

    def __init__(self) -> None:
        pass

    def reset(self):
        pass

    def infer(self, obs: dict, instruction: str) -> dict:
        policy_obs = obs["policy"]
        joint_position = policy_obs["arm_joint_pos"].detach().cpu().numpy()
        action = np.concatenate([joint_position, np.zeros((1,))])

        imgs = [
            cv2.resize(policy_obs[name][0].detach().cpu().numpy(), (224, 224))
            for name in ("external_cam", "wrist_cam")
        ]
        return {"action": action, "viz": np.concatenate(imgs, axis=1)}
