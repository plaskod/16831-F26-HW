# HW1 local setup (Apple Silicon)

Installed and verified on macOS arm64 with Python 3.10.18. The repository is cloned directly into this HW1 directory at commit `71d39698615e7f2a9b283ad8d50f42462bdee6b9`.

## Use the environment

From this repository root:

```bash
source .venv/bin/activate
python scripts/check_setup.py --render
cd hw1
```

Run assignment commands from `hw1`, adding `--no_gpu` for this Mac. For example:

```bash
python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Ant.pkl \
  --env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
  --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
  --learning_rate 4e-3 --n_layers 5 --video_log_freq -1 --no_gpu
```

The assignment TODOs are implemented. Run `python -m unittest discover -s tests -v` from the repository root to check the implementation. Short Ant BC and Hopper DAgger runs, including checkpoints and DAgger videos, have passed. The separate setup check exercises the supplied expert policies. The trainer selects CPU on this Mac; it does not use Apple's MPS backend.

For TensorBoard, from `hw1`:

```bash
python -m tensorboard.main --logdir data
```

For notebooks:

```bash
python -m jupyterlab
```

Select **Robot Learning HW1 (Python 3.10)**. The kernel is registered inside `.venv`. In Cursor, select this repository's `.venv/bin/python` interpreter. The supplied notebook is designed for Colab: skip its Colab installation, cloning, Google Drive, and virtual-display cells when using the local environment. Local macOS rendering uses GLFW and does not need Xvfb.

## What was installed and verified

- Gym 0.25.1 and all five original environments: `Ant-v2`, `HalfCheetah-v2`, `Hopper-v2`, `Walker2d-v2`, `Humanoid-v2`.
- PyTorch 1.12.1, NumPy 1.25.2, native MuJoCo 2.2.1, and mujoco-py 2.1.2.14.
- A separate MuJoCo 2.1.1 arm64 runtime for the legacy `-v2` environments, stored in `.local/mujoco/mujoco210`.
- TensorBoard 2.10.0, tensorboardX 2.5.1, Matplotlib, JupyterLab, IPython, MoviePy 1.0.3, and bundled FFmpeg.
- Reset, expert inference, 10 simulation steps, dataset observation/action dimensions, offscreen rendering, and TensorBoard video/scalar event readback passed for every robot. CPU autograd and an optimizer step passed.
- The homework CLI `--help`, JupyterLab and TensorBoard version checks, and `uv pip check --python .venv/bin/python --no-cache` passed.

The render-check transcript is `.local/setup-check.log`. Graphics checks need access to the macOS desktop services; a restrictive sandbox can block them. Use `python scripts/check_setup.py` for a simulation-only check.

## Compatibility details

The original requirements list both `mujoco-py` and `free-mujoco-py`. This installation uses only `mujoco-py` to avoid overlapping bindings, and supplies its native runtime separately. MuJoCo 2.1.1 provides Apple Silicon support for the legacy binding; see the [upstream Apple Silicon installation discussion](https://github.com/openai/mujoco-py/issues/682).

`scripts/configure_mujoco_mac.py` adjusts only the virtualenv's mujoco-py build flags to use Apple clang without OpenMP (the binding imports `prange` but does not use it). It copies/signs the runtime libraries locally and sets their install names. A virtualenv `.pth` file configures runtime discovery and a single GLFW library for Python/Jupyter. No system library installation or global shell configuration is needed.

Old Gym/Protobuf packages emit deprecation warnings. Modern `pip check` also reports that torch 1.12.1 is unsupported: its distributed WHEEL metadata incorrectly says x86_64, although the installed binaries are verified arm64 and pass inference/autograd checks. `uv pip check` passes. Retaining this version preserves the homework's dependency pin.

## Recreate the environment

The complete resolved Python package list is `requirements-macos.lock.txt`. Run these commands from the repository root (requires `uv` and Apple command-line developer tools):

```bash
uv venv --python 3.10.18 --seed .venv
uv pip install --python .venv/bin/python -r requirements-macos.lock.txt
mkdir -p .local/mujoco
curl -fL https://github.com/google-deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-macos-universal2.dmg \
  -o .local/mujoco/mujoco-2.1.1.dmg
hdiutil attach -nobrowse -readonly -mountpoint /private/tmp/rob831-mujoco211 .local/mujoco/mujoco-2.1.1.dmg
.venv/bin/python scripts/configure_mujoco_mac.py
hdiutil detach /private/tmp/rob831-mujoco211
.venv/bin/python -m ipykernel install --sys-prefix --name rob831 --display-name 'Robot Learning HW1 (Python 3.10)'
.venv/bin/python scripts/check_setup.py --render
```

If you move the repository, recreate `.venv` at its new location and repeat the configuration. Paths in virtual environments and the native libraries are absolute.

## Compile the homework PDF

Tectonic 0.17.0 (native Apple Silicon) is installed at `~/.local/bin/tectonic`.
Open a new Terminal, or run `source ~/.zprofile` in an existing Terminal.
From the repository root:

```bash
cd hw1
tectonic hw1_submission.tex
```

This writes `hw1/hw1_submission.pdf` (relative to the repository root).
The first build downloads required LaTeX packages into the user cache; later builds reuse them.
No Python environment activation is needed. The figures are loaded from `../results/`, so keep the repository layout when building.

The official release archive checksum was verified against GitHub release metadata:
`a3f1cac7c5678f01661a92212f58480ae3b0634115d880dbc59e2953ded45667`.
PDF inspection tools are isolated in `.local/pdf-tools`; the robotics environment was not changed.
