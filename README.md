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

All GEMM methods can be found in **[GEMM.h](src/GEMM.h)**.  
The repository includes several increasingly optimized approaches:

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

---

## Building & Running

You can generate input matrices, build the project, and run the GEMM benchmark as follows:

```bash
python3 gemm.py -m 1024 -n 1024 -k 1024
make
./gemm -m 1024 -n 1024 -k 1024

Select Matrix Multiplication Method
===================================
[1] Naive GEMM
[2] Loop order GEMM
[3] Tiling GEMM
[4] AVX GEMM
 > 2

Results
===================================
    A Matrix [1024 x 1024]
    B Matrix [1024 x 1024]
    C Matrix [1024 x 1024]

    M         =       1024
    N         =       1024
    K         =       1024
    GFLOPS    =   20.45223
    Time (ms) =        105

[SUCCESS] All done!
```

---

## Notes
	•	This project uses C++23, built with GCC/G++ 14.
	•	gemm.py￼ generates matrix data files consumed by main.cpp￼.
	•	Matrix dimensions must match across GEMM configurations and data generation.


---

## License

© Luis Maya Aranda￼. All rights reserved.
Licensed under the MIT License.
