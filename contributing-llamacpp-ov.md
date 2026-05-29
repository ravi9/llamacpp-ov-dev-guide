# Contributing to the LlamaCPP OpenVINO Backend

This guide walks you through the end-to-end workflow for contributing to the OpenVINO backend for llama.cpp — from forking the repo to raising a PR into the development branch.

**Target branch for all PRs:** [`ravi9/llama.cpp` — `dev_backend_openvino`](https://github.com/ravi9/llama.cpp/tree/dev_backend_openvino)

---

## Steps Overview

1. [Fork and Clone](#step-1--fork-and-clone)
2. [Create a Feature Branch from `dev_backend_openvino`](#step-2--create-a-feature-branch-from-dev_backend_openvino)
3. [Build with the OpenVINO Backend](#step-3--build-with-the-openvino-backend)
4. [Make Your Changes](#step-4--make-your-changes)
5. [Test Your Changes](#step-5--test-your-changes)
6. [Commit and Push](#step-6--commit-and-push)
7. [Raise a Pull Request](#step-7--raise-a-pull-request)

---

### Step 1 — Fork and Clone

Fork [`ravi9/llama.cpp`](https://github.com/ravi9/llama.cpp) on GitHub, then clone your fork locally:

```bash
git clone https://github.com/<your-username>/llama.cpp.git
cd llama.cpp
```

Add the upstream development repo as a remote and fetch all branches:

```bash
git remote add upstream https://github.com/ravi9/llama.cpp.git
git fetch upstream
```

> **Tip:** Always run `git fetch upstream` before creating a new branch to make sure you have the latest state of `dev_backend_openvino`.

---

### Step 2 — Create a Feature Branch from `dev_backend_openvino`

Always branch off from `dev_backend_openvino`, **not** `master`:

```bash
git checkout -b my-feature-branch upstream/dev_backend_openvino
```

> ⚠️ Branching from `master` instead of `dev_backend_openvino` will result in missing OpenVINO-specific changes and a harder merge later.

---

### Step 3 — Build with the OpenVINO Backend

```bash
# Linux
source /opt/intel/openvino/setupvars.sh
cmake -B build/ReleaseOV -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON
cmake --build build/ReleaseOV --parallel
```

For full build instructions including Windows, see the [Build Instructions section](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md#build-instructions) in the official docs.

---

### Step 4 — Make Your Changes

Key source locations:
- **Main backend implementation:** `ggml/src/ggml-openvino/ggml-openvino.cpp`
- **Backend source directory:** `ggml/src/ggml-openvino/`
- **Backend docs:** `docs/backend/OPENVINO.md`

For adding support for new GGML ops, follow the [Enable New Ops Guide](./enable-llamacpp-ov-new-ops.md).

---

### Step 5 — Test Your Changes

Before testing, make sure you have a sample model downloaded. See [Download Sample Model](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md#3-download-sample-model) in the official docs.

**Minimum required — CPU inference:**

```bash
export GGML_OPENVINO_DEVICE=CPU
./build/ReleaseOV/bin/llama-simple -m ~/models/Llama-3.2-1B-Instruct-Q4_0.gguf -n 50 "The story of AI is "
```

**Minimum required — `llama-bench` for performance regression check** (`-fa 1` is required):

```bash
./build/ReleaseOV/bin/llama-bench -m ~/models/Llama-3.2-1B-Instruct-Q4_0.gguf -fa 1
```

**If available — test on GPU and NPU:**

```bash
# GPU
GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_STATEFUL_EXECUTION=1 \
  ./build/ReleaseOV/bin/llama-bench -m ~/models/Llama-3.2-1B-Instruct-Q4_0.gguf -fa 1

# NPU (keep context small to avoid failures)
GGML_OPENVINO_DEVICE=NPU \
  ./build/ReleaseOV/bin/llama-cli -m ~/models/Llama-3.2-1B-Instruct-Q4_0.gguf -c 512
```

---

### Step 6 — Commit and Push

Use the `ggml-openvino:` prefix in your commit message for consistency with the project's convention:

```bash
git add .
git commit -m "ggml-openvino: <brief description of your change>"
git push origin my-feature-branch
```

Example commit messages:
```
ggml-openvino: add support for GGML_OP_SILU on CPU/GPU
ggml-openvino: fix stateful execution KV cache reset on context clear
ggml-openvino: update CI to OpenVINO 2026.2
```

---

### Step 7 — Raise a Pull Request

1. Go to your fork on GitHub: `https://github.com/<your-username>/llama.cpp`
2. Click **"Compare & pull request"** for your branch.
3. Set the base repository and branch:
   - **Base repository:** `ravi9/llama.cpp`
   - **Base branch:** `dev_backend_openvino`

   > ⚠️ GitHub may default the base repository to `ggml-org/llama.cpp`. Make sure to change it to `ravi9/llama.cpp` with base branch `dev_backend_openvino` before submitting.

4. Fill in the PR description with:
   - **What** the change does
   - **Why** it is needed (link to the related issue if applicable)
   - **How** it was tested (device, model, quantization format, OS)
   - Any known limitations or follow-up work

5. Verify the PR checklist before submitting:
   - [ ] Build passes without errors or new warnings
   - [ ] Tested on CPU (minimum); GPU/NPU if relevant to the change
   - [ ] No unrelated changes included
   - [ ] Commit message follows the `ggml-openvino:` convention
   - [ ] Docs updated if the change affects user-facing behavior

6. Submit the PR for review.

> **Note:** PRs should target `dev_backend_openvino`, **not** `master`. Changes will be upstreamed to `ggml-org/llama.cpp` after validation by the team.