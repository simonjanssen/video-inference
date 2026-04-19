---
name: cuda-setup
description: 'Set up CUDA runtime libraries for ONNX Runtime GPU inference. Use when encountering missing .so errors (libcurand, libcufft, libcudnn, etc.), setting up a new machine for GPU inference, or verifying CUDA dependencies.'
---

# CUDA Setup for ONNX Runtime

Set up and verify the CUDA runtime libraries required by the `ort` crate's CUDA execution provider.

## When to Use

- Setting up a new machine for GPU inference
- Errors like `Failed to load library libonnxruntime_providers_cuda.so` or `cannot open shared object file`
- Verifying CUDA dependencies after a system update
- Debugging missing shared library issues (`libcurand`, `libcufft`, `libcudnn`, etc.)

## Background

The NVIDIA GPU driver alone is **not enough**. The `ort` crate bundles ONNX Runtime with `libonnxruntime_providers_cuda.so`, which dynamically links against CUDA Toolkit runtime libraries — a separate set of shared libraries not included in the driver package. See [CUDA architecture details](./references/cuda-architecture.md) for the full explanation and library reference.

## Procedure

### 1. Verify NVIDIA Driver

Confirm the driver is installed and a GPU is detected:

```bash
nvidia-smi
```

If this fails, install the NVIDIA driver first — that is outside the scope of this skill.

### 2. Detect CUDA Version

Determine the CUDA version supported by the installed driver:

```bash
nvidia-smi | grep "CUDA Version"
```

This prints something like `CUDA Version: 12.x`. Note the **major version** (e.g., `12`) — package name suffixes must match.

If `nvcc` is available, you can also check:

```bash
nvcc --version
```

### 3. Install CUDA Runtime Libraries (Ubuntu/Debian)

The package version suffixes must match the detected CUDA major version. Below are the packages for **CUDA 12**:

```bash
sudo apt install -y \
  libcurand10 \
  libcufft11 \
  libcusparse12 \
  libcusolver11 \
  libcublas12 \
  libnvrtc12 \
  libnvjitlink12 \
  libcudart12 \
  nvidia-cudnn
```

> **Note:** For other CUDA versions, search for the matching package names with `apt search libcurand` and adjust the version suffixes accordingly.

### 4. Build the Project

```bash
cargo build --release
```

On `linux/x86_64`, the `cuda` features is enabled automatically via `Cargo.toml` target-specific dependencies.

### 5. Verify All Dependencies Resolved

```bash
ldd target/release/examples/libonnxruntime_providers_cuda.so | grep "not found"
```

- **No output** = all libraries are correctly installed.
- **Any `not found` lines** = identify the missing library, find the matching package, and install it (repeat from step 2).

### 6. Test GPU Inference

Run an example to confirm end-to-end GPU inference works:

```bash
cargo run --release --example simple
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `libcurand.so.10: cannot open shared object file` | Missing CUDA runtime package | Install the specific `libcurand10` (or matching version) package |
| `nvidia-smi` not found | No NVIDIA driver installed | Install `nvidia-driver-xxx` first |
| `ldd` shows multiple `not found` | CUDA Toolkit runtime not installed | Run the full `apt install` command in step 2 |
| Build succeeds but inference errors at runtime | Library version mismatch | Check CUDA version compatibility between driver and runtime packages |