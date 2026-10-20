E123-v2 B, seed 0, episode-14 resumable transfer
=============================================

This deliberately versioned 9.7 MiB archive transfers the stopped B checkpoint
and the exact result artifacts required by its inherited contract. It contains
no A checkpoint and no AntMaze/ECCC data. Code base: db089357d13e1aaf8efe9fd8eccee5f5e29c4893.

Only load trusted PyTorch checkpoints: this archive includes resumable Python
objects, optimizer, replay and RNG state. Verify SHA256SUMS before extraction.

From the repository root:
  (cd artifacts/checkpoint_transfers/e123-b-episode14 && sha256sum -c SHA256SUMS)
  tar --keep-old-files -xzf artifacts/checkpoint_transfers/e123-b-episode14/e123-b-resume-20261020.tar.gz

Extraction preserves existing paths: conflicts are errors, not overwrites.
Extract only into the isolated E123 worktree before starting training. Do not
extract this old endpoint into a directory containing newer training outputs.
The archive paths are relative to the repository root under results/.

With a configured CUDA-capable environment (or PYTHON_BIN pointing to one):
  bash experiments/conditioned_vcg/E123_absolute_progress_8x8_187k_v2/run.sh train-arm --arm B_joint_bounded --model-seed 0 --through-episode 120 --device cuda

The launcher resumes episode 15, validates every six episodes, and saves B
independently. Do not migrate again. This is the normal launcher without an
external watchdog. The compressed archive is intentionally committed for this
one-off transfer; ongoing training checkpoints should not be added to Git.
