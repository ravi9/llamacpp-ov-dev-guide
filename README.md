# LlamaCPP OpenVINO Backend — Developer Guide

This repository contains development guides, references, and resources for contributing to the [OpenVINO backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md) for [llama.cpp](https://github.com/ggml-org/llama.cpp).

---

## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Key Links](#key-links)
- [Development Branch](#development-branch)
- [Issue Tracking](#issue-tracking)
- [Guides](#guides)
- [Quick Reference: Current OPENVINO.md](#quick-reference-current-openvinomd)

---

## Project Overview

The **OpenVINO backend for llama.cpp** integrates [Intel OpenVINO](https://docs.openvino.ai/) into llama.cpp's inference pipeline, enabling optimized AI inference on Intel CPUs, GPUs, and NPUs.

The backend:
- Walks the GGML compute graph and identifies inputs, outputs, weights, and KV cache tensors.
- Translates GGML operations into an `ov::Model` using OpenVINO's frontend API.
- Compiles and caches the model for the target device.
- Binds GGML tensor memory to OpenVINO inference tensors and runs inference.

> **Status:** The backend is actively under development. Performance and memory optimizations, accuracy validation, broader quantization coverage, and broader operator and model support are work in progress.

---

## Key Links

| Resource | Link |
|---|---|
| 📄 Official Docs in ggml-org/llama.cpp | [docs/backend/OPENVINO.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md) |
| 🔀 Development Branch | [ravi9/llama.cpp — dev_backend_openvino](https://github.com/ravi9/llama.cpp/tree/dev_backend_openvino) |
| 📋 Project Issue Tracker | [ravi9 — Project #4](https://github.com/users/ravi9/projects/4/views/3) |
| 🐛 Development Team Issues | [ravi9/llama.cpp Issues](https://github.com/ravi9/llama.cpp/issues) |
| 🤝 Contributing Guide | [contributing-llamacpp-ov.md](./contributing-llamacpp-ov.md)  |
| 🔧 Enable New Ops Guide | [enable-llamacpp-ov-new-ops.md](./enable-llamacpp-ov-new-ops.md) |

---
## Development Branch

 - Active OpenVINO backend development happens on: [`ravi9/llama.cpp` — `dev_backend_openvino`](https://github.com/ravi9/llama.cpp/tree/dev_backend_openvino)
 and it tracks upstream `ggml-org/llama.cpp` and adds OpenVINO-specific development work.

 - To contribute to LlamaCPP OpenVINO Backend, see: **[contributing-llamacpp-ov.md](./contributing-llamacpp-ov.md)**

 - To get started with the development branch:

    ```bash
    git clone https://github.com/ravi9/llama.cpp.git
    cd llama.cpp
    git checkout dev_backend_openvino
    ```

    Build with the OpenVINO backend:

    ```bash
    # Linux
    source /opt/intel/openvino/setupvars.sh
    cmake -B build/ReleaseOV -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON
    cmake --build build/ReleaseOV --parallel
    ```

 - See the [official docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md) for full build instructions including Windows.

---

## Guides

### 🤝 [Contributing to the LlamaCPP OpenVINO Backend](./contributing-llamacpp-ov.md)

 End-to-end workflow for contributing to the OpenVINO backend: forking, branching off `dev_backend_openvino`, building, testing, and raising a PR.

### 🔧 [Enable New GGML Ops for OpenVINO Backend](./enable-llamacpp-ov-new-ops.md)

Step-by-step guide for adding support for new GGML operations in the OpenVINO backend. Covers:
- Identifying unsupported ops in the GGML compute graph
- Implementing the op translation layer using OpenVINO's frontend API
- Testing and validation

--- 

## Issue Tracking

Development team issues and tasks are tracked in two places:

- **GitHub Issues in dev repo:** [ravi9/llama.cpp/issues](https://github.com/ravi9/llama.cpp/issues)
- **GitHub Issues in ggml-org:** [ggml-org/llama.cpp/issues](https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue%20state%3Aopen%20OpenVINO)

When filing a new issue, please include:
- Hardware (CPU/GPU/NPU model)
- OS and driver versions
- OpenVINO runtime version
- Model name and quantization format
- Steps to reproduce and observed vs. expected behavior

---

## Contributing

- See [contributing-llamacpp-ov.md](./contributing-llamacpp-ov.md) for the full contribution workflow.

- For upstreaming to `ggml-org/llama.cpp`, follow the standard [llama.cpp contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md).