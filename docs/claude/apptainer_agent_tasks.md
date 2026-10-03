# Tasks for the apptainer (Isaac) agent

Context: DINO/ResNet encoder finetuning was added to `scripts/train_steering.py` and `src/polaris/latent_rl/encoders.py`
(see `docs/claude/steering_training.md`). A non-apptainer agent on an A40 already verified the following, so don't redo them:
- `droid_resnet50` (`pretrained=False`, cuda): numpy `encode_packed` matches torch `forward` exactly; backward gives non-zero
  param grads; `encoder(imgs, proprio)` works (the earlier `TypeError` is fixed).
- `tests/test_residual_finetune.py` passes (3/3).
- `tests/test_ppo.py`: 4/5 pass. `test_ppo_improves_bandit` fails (before=-3.25, after=-1.53, needs > -0.5). That run used a
  stubbed pytest. `dsrl_ppo.py` is not in the git diff, so it is probably unrelated to the encoder change, but it is unconfirmed.

NOT yet verified (this shell had no Isaac, no `transformers`, no pytest). First check you are inside the apptainer
(`ls /isaac-sim/python.sh`). Remember Isaac hangs at exit; check periodically and terminate with ctrl+].

## Part 1: verification

1. **Unit tests** with real pytest: `tests/test_residual_finetune.py`, `tests/test_ppo.py`, `tests/test_latent_rl.py`
   (the fixture-based tests there, including `test_vision_encoder_shape_and_ckpt_kwargs`, never ran).
2. **Triage `test_ppo_improves_bandit`:** run it on the current tree and on `git stash` (HEAD). If it fails on both, it is
   pre-existing: report it, don't fix it unless trivial.
3. **DINO encoder check:** build `make_encoder("droid_vision", finetune=True, device="cuda")` (needs HF model
   `facebook/dinov2-small` cached or reachable). Check numpy `encode_packed` vs torch `forward` match, and backward gives param grads.
   Also check `pretrained=False` (builds from `AutoConfig.from_pretrained`).
4. **Isaac smoke tests** (each `--iterations 2`):
   ```
   isaacpy scripts/train_steering.py --environment CubeBowl --num-envs 8 --dummy-policy --wandb-mode disabled --iterations 2
   ... --residual-scale 0
   ... --no-finetune-encoder
   ... --encoder droid_vision
   ```
   Look for: `res_abs_mean` only when the residual is on; `enc_grad_norm` only when finetuning; encoder weights change when
   finetuning and stay fixed when frozen (compare state_dict before/after or the saved checkpoint).
5. **Checkpoint reload:** load a saved checkpoint via `SteeredPolicy.from_checkpoint` (for both encoders; finetuned ones
   save/load `encoder.pt` and build with `pretrained=False`).
6. **Memory:** record peak GPU memory in the finetune update at `encoder_microbatch=32` while Isaac is rendering; try 16 if it OOMs.

Report pass/fail per item with exact errors. Fix real bugs in the encoder/script code; tell the user what you changed.

## Part 2: implement critic warmup (iterations with no policy updates)

Where: `DSRLPPO.update` in `src/polaris/latent_rl/algos/dsrl_ppo.py`; the training loop calls it at `scripts/train_steering.py`
(`train_metrics = ppo.update(buf)`).

Design (agreed with the user as a recommendation; the encoder-freezing choice is the one open question, so default to freezing):
1. Add `--critic-warmup-iters N` (default 0) to the script (and a matching field wherever the PPO config lives) and an
   `update(buf, train_policy: bool = True)` argument.
2. When `train_policy=False`: `loss = vf_coef * v_loss` only. Skip `pg_loss` and the entropy term and the actor logp. Still
   run `compute_returns` in the loop so value targets exist.
3. The actor must not move. Adam skips params whose `.grad` is `None`; verify `zero_grad()` sets to None in the torch version
   inside the apptainer. If not, either use `zero_grad(set_to_none=True)` or exclude actor params from `opt.step`. Clip grad norm
   on the critic params only. Verify empirically: actor `state_dict` is bit-identical after a warmup update, including Adam
   state not drifting the actor.
4. Encoder: freeze during warmup by detaching `feat` and skipping `_backward_encoder` (so no `enc_grad_norm`, encoder weights
   unchanged). Add `--warmup-train-encoder` to opt in to training it.
5. Rollouts keep using the unchanged actor (starts near N(0, I)), so data is on-policy for the critic.
6. Skip the `target_kl` early stop during warmup. Skip policy metrics (`pg_loss`, `approx_kl`, `clipfrac`); log `warmup=1`.
   `explained_var` is the metric to watch. Optional: `--warmup-min-explained-var` to end warmup early once it is exceeded.
7. Warmup must be tracked by the iteration counter so resuming from a checkpoint doesn't redo it (check how resume works).
8. Add a unit test in `tests/test_ppo.py` (or a new file): after a warmup update the actor params are identical, critic params
   changed, and with a frozen encoder the encoder params are identical.
9. Smoke test in Isaac with `--critic-warmup-iters 2 --iterations 4`, and confirm policy metrics appear only from iteration 3.
10. Document it in `docs/claude/steering_training.md` (and `steering_metrics.md` for the new `warmup` metric).
