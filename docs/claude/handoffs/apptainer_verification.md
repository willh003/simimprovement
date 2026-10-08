# Handoff: verification that needs the apptainer (Isaac)

Context: DINO/ResNet encoder finetuning was added to `scripts/train_steering.py` and `src/polaris/latent_rl/encoders.py`
(see `../current/steering_training.md`). A non-apptainer agent on an A40 already verified the following, so don't redo them:
- `droid_resnet50` (`pretrained=False`, cuda): numpy `encode_packed` matches torch `forward` exactly; backward gives non-zero
  param grads; `encoder(imgs, proprio)` works (the earlier `TypeError` is fixed).
- `tests/test_residual_finetune.py` passes (3/3).
- `tests/test_ppo.py`: 4/5 pass. `test_ppo_improves_bandit` fails (before=-3.25, after=-1.53, needs > -0.5). That run used a
  stubbed pytest. `dsrl_ppo.py` is not in the git diff, so it is probably unrelated to the encoder change, but it is unconfirmed.

NOT yet verified (this shell had no Isaac, no `transformers`, no pytest). First check you are inside the apptainer
(`ls /isaac-sim/python.sh`). Remember Isaac hangs at exit; check periodically and terminate with ctrl+].

## Checks

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

The critic-warmup feature that used to be Part 2 of this handoff is implemented (`../current/steering_training.md`, `tests/test_ppo.py::test_critic_warmup_*`); its Isaac smoke test
(`--critic-warmup-iters 2 --iterations 4`, policy metrics only from iteration 3) is the one remaining item from it.
