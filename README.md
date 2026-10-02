<br>

<p align="center">
  <a href="https://github.com/v0-3">
    <img width="200" src="logo/gemm.png" alt="GEMM logo">
  </a>
</p>

<br>

# GEMM

This repository provides multiple implementations of **General Matrix Multiplication (GEMM)** in modern C++.  
Each approach highlights a distinct optimization strategy, enabling performance comparison, experimentation, and educational analysis.

If you have questions or would like to discuss the project, feel free to contact me here or on **Discord: `@v0.3`**.

---

## GEMM Implementations

The kernels and public declarations are in **[GEMM.h](src/GEMM.h)**, with the Apple Silicon implementation in **[AppleGEMM.mm](src/AppleGEMM.mm)**.

Each method computes `C = A * B` for non-overlapping, row-major `float` matrices, replacing the previous contents of `C`. Rectangular matrices and dimensions that are not multiples of a tile size are supported.

### 1. Naive
A direct, triple-nested loop implementation.  
Serves as a baseline for evaluating more advanced methods.

### 2. Loop Interchange (Cache-Aware)
Reorders the loop structure to improve spatial locality and reduce cache misses.  
A simple yet effective optimization.

### 3. Tiling (Cache Blocking)
Breaks matrices into smaller blocks ("tiles") that fit into cache.  
Significantly improves data reuse and overall performance.

### 4. AVX GEMM
Vectorized GEMM implementation using AVX intrinsics for parallel computation across SIMD registers.

### 5. Apple Silicon GEMM

Available in native ARM64 macOS builds. Select **5** to choose a backend automatically based on the matrix dimensions:

| Backend | When it is used |
| --- | --- |
| ARM NEON | All three dimensions are at most 16; a register-blocked kernel avoids library-call overhead. |
| Metal Performance Shaders | All three dimensions are at least 512 and the product `M × N × K` is at least `5120³`; GPU execution includes input copies and waits for completion before returning. |
| Accelerate BLAS | All other sizes, and as a CPU fallback if Metal setup or command completion fails. |

These thresholds were tuned on an Apple M3 Pro to account for GPU startup and copy costs. The GPU path allocates temporary buffers; performance varies with matrix shape and hardware. Apple system frameworks provide the Accelerate and Metal backends, with no third-party GPU library required.

---

## Building & Running

Use a C++23 compiler and standard library with `<print>` support, plus `make`. On macOS, Xcode or its Command Line Tools must provide Apple Clang and the macOS SDK. The Makefile uses `clang++` for the Objective-C++ source and automatically links Accelerate, Foundation, Metal, and Metal Performance Shaders. `CXX` controls the C++ compiler; `OBJCXX` controls the Objective-C++ compiler.

Python 3 and NumPy are required to generate `data.bin`, which contains the inputs and NumPy reference product.

You can generate input matrices, build the project, and run the GEMM benchmark as follows:

```bash
cd src
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy
python3 gemm.py -m 1024 -n 1024 -k 1024
make
./gemm -m 1024 -n 1024 -k 1024
```

Example method selection on Apple Silicon:

```text
Select Matrix Multiplication Method
===================================
[1] Naive GEMM
[2] Loop order GEMM
[3] Tiling GEMM
[4] AVX GEMM (unavailable in this build)
[5] Apple Silicon GEMM
> 5
```

The program reports elapsed time and GFLOPS, then checks the result against the NumPy reference. Unsupported methods are marked unavailable in the menu and rejected if selected. Apple Silicon GEMM is unavailable in Intel builds, including Intel binaries running under Rosetta.

AVX is available when enabled by the compiler target; on x86, use `make clean && make AVX_ON=1` to enable it explicitly. The default build uses `-O3 -march=native`. Run `make clean` before changing the compiler or compiler flags.

## Tests & Benchmarks

From `src`, with the Python environment above activated:

```bash
make test
make bench
```

`make test` checks the kernels, utilities, command-line validation, and NumPy data generation. On Apple Silicon it also checks unaligned buffers, repeated calls, backend boundaries, and large rectangular products. The NumPy round-trip test is skipped if NumPy is not installed.

`make bench` builds [bench_gemm.cpp](src/tests/bench_gemm.cpp) and generates its own inputs; it does not require NumPy or `data.bin`. It reports the first call for each method and shape, the median of five warmed samples, and GFLOPS. Tiny calls are batched. Timings include any setup, allocation, copying, and synchronization performed inside the method, but exclude input generation and result verification. Large scalar cases are skipped to keep the benchmark practical.

Example results on an Apple M3 Pro, using Apple Clang 21.0.0 on macOS 27.0.1 with the default build flags:

| Square dimensions (`M = N = K`) | Tiling median (ms) | Apple Silicon first (ms) | Apple Silicon median (ms) |
| --- | ---: | ---: | ---: |
| 1024 | 60.000 | 1.392 | 1.409 |
| 5120 | Not run | 147.779 | 90.646 |

The 1024 case was approximately **43× faster** than tiling. The 5120 case reached approximately **2.96 TFLOPS** when warmed. These are local measurements; use `make bench` to measure your own machine. First-call timings are per shape, so later cases may reuse initialized libraries.

---

## Notes

- Matrix `A` has shape `M × K`, `B` has shape `K × N`, and `C` has shape `M × N`.
- [gemm.py](src/gemm.py) writes `data.bin` in the current directory; run `gemm` from that same directory.
- Use the same positive dimensions for data generation and execution. Regenerate `data.bin` when changing them.

---

## License

© [Luis Maya Aranda](https://github.com/v0-3). All rights reserved.  
Licensed under the **MIT License**.
