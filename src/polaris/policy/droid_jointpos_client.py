import numpy as np
from openpi_client import websocket_client_policy, image_tools
from polaris.policy.abstract_client import InferenceClient, PolicyArgs


# Joint Position Client for DROID
@InferenceClient.register(client_name="DroidJointPos")
class DroidJointPosClient(InferenceClient):
    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        if args.open_loop_horizon is None:
            raise ValueError("open_loop_horizon must be set for DroidJointPosClient")

        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port
        )
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.open_loop_horizon = args.open_loop_horizon

    @property
    def rerender(self) -> bool:
        return (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        )

    def visualize(self, request: dict):
        """
        Return the camera views how the model sees it
        """
        curr_obs = self._extract_observation(request)
        base_img = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        combined = np.concatenate([base_img, wrist_img], axis=1)
        return combined

    def build_request(self, obs: dict, instruction: str) -> tuple[dict, np.ndarray]:
        """Build the openpi request dict and the model-view visualization."""
        curr_obs = self._extract_observation(obs)
        exterior_image = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_image = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        request_data = {
            "observation/exterior_image_1_left": exterior_image,
            "observation/wrist_image_left": wrist_image,
            "observation/joint_position": curr_obs["joint_position"],
            "observation/gripper_position": curr_obs["gripper_position"],
            "prompt": instruction,
        }
        return request_data, np.concatenate([exterior_image, wrist_image], axis=1)

    def query_chunk(self, request: dict, noise: np.ndarray | None = None) -> np.ndarray:
        """Query the server for an action chunk; optional initial flow-matching noise."""
        if noise is not None:
            request = {**request, "noise": noise}
        return self.client.infer(request)["actions"]

    @staticmethod
    def postprocess_action(action: np.ndarray) -> np.ndarray:
        """Binarize the gripper dimension (kept outside the learned part)."""
        grip = 1.0 if action[-1].item() > 0.5 else 0.0
        return np.concatenate([action[:-1], np.full((1,), grip)])

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Infer the next action from the policy in a server-client setup
        """
        both = None
        ret = {}
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            self.actions_from_chunk_completed = 0
            request_data, both = self.build_request(obs, instruction)
            self.pred_action_chunk = self.query_chunk(request_data)

        if return_viz and both is None:
            curr_obs = self._extract_observation(obs)
            both = np.concatenate(
                [
                    image_tools.resize_with_pad(curr_obs["right_image"], 224, 224),
                    image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                ],
                axis=1,
            )

        if self.pred_action_chunk is None:
            raise ValueError("No action chunk predicted")

        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        action = self.postprocess_action(action)

        return action, both

    def _extract_observation(self, obs_dict):
        # Assign images
        right_image = obs_dict["splat"]["external_cam"]
        wrist_image = obs_dict["splat"]["wrist_cam"]

        # Capture proprioceptive state
        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()[0]
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()[0]

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }


@InferenceClient.register(client_name="DroidJointPosSimEvals")
class SimEvalsJointPosClient(DroidJointPosClient):
    """Same as DroidJointPosClient, but reads the sim-evals env's plain-render cameras (obs['policy'])."""

    def _extract_observation(self, obs_dict):
        policy = obs_dict["policy"]
        return {
            "right_image": policy["external_cam"][0].detach().cpu().numpy(),
            "wrist_image": policy["wrist_cam"][0].detach().cpu().numpy(),
            # sim-evals proprio obs have no env dim (unlike the splat env): (7,) and (1,)
            "joint_position": policy["arm_joint_pos"].detach().cpu().numpy(),
            "gripper_position": policy["gripper_pos"].detach().cpu().numpy(),
        }
