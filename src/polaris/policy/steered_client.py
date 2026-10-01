import numpy as np

from polaris.latent_rl.steered_policy import SteeredPolicy
from polaris.policy.abstract_client import InferenceClient, PolicyArgs
from polaris.policy.droid_jointpos_client import DroidJointPosClient


class ServerChunkPolicy:
    """Adapts DroidJointPosClient to the latent_rl ChunkPolicy protocol."""

    def __init__(self, client: DroidJointPosClient, noise_shape: tuple[int, int]):
        self.client, self.noise_shape = client, noise_shape

    def query(self, obs: dict, noise: np.ndarray | None = None) -> np.ndarray:
        return self.client.query_chunk(obs, noise)


@InferenceClient.register(client_name="Steered")
class SteeredClient(DroidJointPosClient):
    """DroidJointPos client whose chunk is generated from actor-chosen flow noise."""

    def __init__(self, args: PolicyArgs) -> None:
        super().__init__(args)
        if args.steering_ckpt is None:
            raise ValueError("policy.steering_ckpt must be set for the Steered client")
        from polaris.latent_rl.checkpoint import load_checkpoint

        _, cfg = load_checkpoint(args.steering_ckpt)
        self.steered = SteeredPolicy.from_checkpoint(
            args.steering_ckpt,
            ServerChunkPolicy(self, cfg.noise_shape),
            deterministic=args.deterministic,
        )

    def query_chunk(self, request: dict, noise: np.ndarray | None = None) -> np.ndarray:
        # Inside infer() (noise is None) route through the steering policy; the
        # ServerChunkPolicy passes explicit noise, which hits the base implementation.
        if noise is None:
            return self.steered.step(request).chunk
        return super().query_chunk(request, noise)
