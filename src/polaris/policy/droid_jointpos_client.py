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

    def query_chunks(self, requests: list[dict], noise: np.ndarray | None = None, batch_size: int | None = None) -> np.ndarray:
        """Batched `query_chunk` via the server's `infer_batch`: N requests (+ optional noise (N, H, D)) -> (N, H, A).

        Sent in sub-batches of `batch_size` (default: all at once); the last one is padded so the server always sees the
        same batch size (JAX recompiles for every new size).
        """
        if noise is not None:
            requests = [{**r, "noise": n} for r, n in zip(requests, noise)]
        bs = batch_size or len(requests)
        chunks = []
        for s in range(0, len(requests), bs):
            sub = requests[s : s + bs]
            pad = bs - len(sub)
            out = self.client.infer_batch(sub + [sub[-1]] * pad)
            chunks += [r["actions"] for r in out[: len(sub)]]
        return np.stack(chunks)

    @staticmethod
    def postprocess_actions(actions: np.ndarray) -> np.ndarray:
        """Batched `postprocess_action`: (N, A) with the gripper (last dim) binarized."""
        actions = np.array(actions, dtype=np.float32)
        actions[:, -1] = (actions[:, -1] > 0.5).astype(np.float32)
        return actions

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

    def _extract_observation(self, obs_dict, env_ids=None):
        """Model inputs of env 0; with `env_ids`, of those envs (leading dim N)."""
        policy = obs_dict["policy"]
        idx = 0 if env_ids is None else env_ids
        get = lambda name: policy[name][idx].detach().cpu().numpy()
        return {
            "right_image": get("external_cam"),
            "wrist_image": get("wrist_cam"),
            "joint_position": get("arm_joint_pos"),
            "gripper_position": get("gripper_pos"),
        }

    def build_requests(self, obs: dict, instruction: str, env_ids=None) -> tuple[list[dict], np.ndarray]:
        """Batched `build_request` for envs `env_ids` (default: all): list of requests and (N, 224, 448, 3) model views."""
        if env_ids is None:
            env_ids = slice(None)
        o = self._extract_observation(obs, env_ids)
        ext = image_tools.resize_with_pad(o["right_image"], 224, 224)
        wrist = image_tools.resize_with_pad(o["wrist_image"], 224, 224)
        requests = [
            {
                "observation/exterior_image_1_left": ext[i],
                "observation/wrist_image_left": wrist[i],
                "observation/joint_position": o["joint_position"][i],
                "observation/gripper_position": o["gripper_position"][i],
                "prompt": instruction,
            }
            for i in range(len(ext))
        ]
        return requests, np.concatenate([ext, wrist], axis=2)
