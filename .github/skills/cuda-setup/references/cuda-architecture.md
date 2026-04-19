# Why the NVIDIA Driver Alone Is Not Enough

Installing the NVIDIA GPU driver (e.g. `nvidia-driver-590`) gives you `nvidia-smi` and the
ability to run GPU-accelerated applications, but the driver only ships the **low-level kernel
module** and a handful of user-space libraries (OpenGL, Vulkan, `libnvidia-compute`, etc.).

The [ort](https://crates.io/crates/ort) crate bundles a pre-built **ONNX Runtime** binary.
When compiled with the `cuda` feature, this binary includes `libonnxruntime_providers_cuda.so`,
which dynamically links against the **CUDA runtime toolkit libraries** — a separate set of
shared libraries that provide higher-level GPU functionality:

| Library | Purpose |
|---|---|
| `libcudart` | CUDA Runtime API (kernel launches, memory management) |
| `libcublas` | GPU-accelerated linear algebra (GEMM, etc.) |
| `libcufft` | Fast Fourier Transforms on GPU |
| `libcurand` | Random number generation on GPU |
| `libcusparse` | Sparse matrix operations |
| `libcusolver` | Dense/sparse direct solvers |
| `libcudnn` | Deep neural network primitives (convolution, pooling, normalization) |
| `libnvrtc` | Runtime compilation of CUDA C++ to PTX |
| `libnvjitlink` | JIT linking of GPU code |

These libraries are part of the **CUDA Toolkit**, not the driver package. On Ubuntu/Debian
they are available as individual runtime packages, so you don't need to install
the full CUDA Toolkit — just the shared libraries that ONNX Runtime links against.
