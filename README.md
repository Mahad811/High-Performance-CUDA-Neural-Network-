<h1 align="center">High-Performance CUDA Neural Network Library</h1>

<p align="center">
  <img src="https://img.shields.io/badge/C++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white" alt="C++" />
  <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA" />
  <img src="https://img.shields.io/badge/Nvidia%20Nsight-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="Nsight" />
</p>

> A high-performance neural network computation library written entirely from scratch in C++ and CUDA. By utilizing raw CUDA kernels for foundational tensor operations, this library achieves up to a **12x throughput increase** compared to standard CPU-bound matrix mathematics while maintaining 97% accuracy on benchmark vision tasks like MNIST.

## 🔥 Key Technical Highlights

* **Raw CUDA Kernel Engineering:** Authored low-level CUDA kernels without relying on high-level frameworks like PyTorch or TensorFlow.
* **Tiled Matrix Multiplication:** Implemented shared memory tiling mechanisms to maximize SM utilization and bypass slow global memory latency.
* **2D-Convolution Acceleration:** Optimized spatial filter application across 2D image domains using optimized grid and block dimensional launches.
* **Nsight Profiling:** Heavily profiled execution paths using Nvidia Nsight to eliminate warp divergence and coalesce memory access.

---

## 🏗️ Repository Architecture

The iteration of optimizations can be viewed in the `src` directory, scaling from naive implementations to fully optimized kernels:

```text
📦 High-Performance-CUDA-Neural-Network
 ┣ 📂 src
 ┃ ┣ 📜 version1_cpu_baseline.c          # CPU Baseline Reference
 ┃ ┣ 📜 version2_naive_cuda.cu           # Naive CUDA Implementation
 ┃ ┣ 📜 version3_shared_memory.cu        # Shared Memory Tiling Kernels
 ┃ ┗ 📜 version4_optimized_convolution.cu# Final Optimized Convolution & MatMul Kernels
 ┣ 📂 data
 ┣ 📂 docs                               # Implementation and Research Reports
 ┣ 📜 .gitignore
 ┣ 📜 Makefile                           # Standard NVCC Build System
 ┗ 📜 README.md
```

## 🚀 Quick Start

### Prerequisites
* You must have the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) installed.
* An NVIDIA GPU with compute capability `sm_60` or higher (configurable in the `Makefile`).
* `nvcc` compiler added to your system `$PATH`.

### Compilation

Clone the repository and build the latest optimized version (`version4_optimized_convolution.cu`):

```bash
git clone https://github.com/Mahad811/High-Performance-CUDA-Neural-Network-.git
cd High-Performance-CUDA-Neural-Network-
make build
```

*(You can also build older versions to benchmark the performance improvements by running `make v2.exe` or `make v3.exe`)*

### Execution

Run the compiled executable directly from the terminal:

```bash
./v4.exe
```

For profiling memory boundaries and function runtimes (requires `gprof`):

```bash
make profile
```

## 📈 Performance Summary

* **Inference Throughput:** Accelerated inference by dynamically batching standard matrix permutations into CUDA block operations. At peak scale, tests confirmed a factor of **12.4x speedup** versus sequential Host/CPU evaluation.
* **Model Accuracy:** Ensured floating-point math stability across thread divergent boundaries, retaining a precise **97% top-1 accuracy** on validation distributions (MNIST database).

---
*Created by [Muhammad Mahad Khan](https://github.com/Mahad811)*
