# Experiment Notes: Sequential vs. OpenMP Matrix Multiplication

## 1. Overview and Problem Definition

This experiment evaluates the computational performance, scaling behavior, and execution efficiency between **Sequential (Single-threaded CPU)** and **OpenMP (Shared-Memory Multi-threaded CPU)** implementations of dense matrix multiplication.

### Problem Formulation
Given two square matrices \( A \) and \( B \) of size \( N \times N \) where \( N = 4000 \):
\[
C = A \times B
\]
Each element \( C[i][j] \) is evaluated as the dot product:
\[
C[i][j] = \sum_{k=0}^{N-1} A[i][k] \cdot B[k][j]
\]

### Computational Complexity
- **Total Floating-Point Operations (FLOPs)**: 
  Each element requires \( N \) multiplications and \( N-1 \) additions.
  For \( N = 4000 \):
  \[
  \text{FLOPs} \approx 2 \times N^3 = 2 \times (4000)^3 = 1.28 \times 10^{11} \text{ operations (128 GFLOPs)}
  \]
- **Algorithmic Time Complexity**: \( \mathcal{O}(N^3) \)
- **Algorithmic Space Complexity**: \( \mathcal{O}(N^2) \) dynamic memory allocation for three double-precision matrices:
  \[
  \text{Memory Per Matrix} = 4000 \times 4000 \times 8 \text{ bytes} \approx 128 \text{ MB}
  \]
  \[
  \text{Total Working Set} = 3 \times 128 \text{ MB} \approx 384 \text{ MB}
  \]

---

## 2. Experimental Environment & System Specifications

| Parameter | Specification |
|---|---|
| **Host Operating System** | Windows 11 |
| **Virtualization / Subsystem** | WSL2 (Ubuntu 24.04 / Resolute) |
| **Compiler** | GCC 15.2.0 (`Ubuntu 15.2.0-16ubuntu1`) |
| **Optimization Flag** | `-O2` |
| **Parallel API** | OpenMP (via GCC `-fopenmp`) |
| **Thread Count (Sequential)** | 1 Thread |
| **Thread Count (OpenMP)** | 8 Threads (`OMP_NUM_THREADS=8`) |
| **System Logical CPUs** | 32 Logical Cores (`nproc` = 32) |
| **Memory Allocation** | Dynamic Heap Allocation (`malloc`) with 1D row-major indexing |

---

## 3. Implementation Details

### 3.1 Sequential Implementation (`sequential/matrix_sequential.c`)
- **Execution Model**: Single thread executing on one logical core.
- **Timing Mechanism**: POSIX `clock()` from `<time.h>`, measuring total CPU cycles divided by `CLOCKS_PER_SEC`.
- **Memory Access**: Row-major order \( C[i \times N + j] += A[i \times N + k] \times B[k \times N + j] \).
- **Initialization**: Matrices \( A \) and \( B \) initialized with \( 1.0 \), \( C \) initialized to \( 0.0 \).

### 3.2 OpenMP Implementation (`openmp/matrix_openmp.c`)
- **Execution Model**: Fork-join shared-memory multi-threading.
- **Parallel Directive**:
  ```c
  #pragma omp parallel for private(j, k)
  for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
          for (k = 0; k < N; k++) {
              C[i * N + j] += A[i * N + k] * B[k * N + j];
          }
      }
  }
  ```
- **Loop Scheduling**: Default static schedule dividing the \( 4000 \) outer loop iterations across \( 8 \) threads (500 iterations per thread).
- **Private Variables**: Variables `j` and `k` are declared private to ensure thread safety and avoid race conditions. Matrix \( C \) is written without synchronization overhead because each thread writes to distinct rows \( i \).
- **Timing Mechanism**: `omp_get_wtime()`, capturing high-resolution wall-clock execution time.

---

## 4. Measured Experimental Data

| Implementation | Matrix Size (\( N \)) | Threads | Execution Time (s) | Speedup (\( S \)) | Parallel Efficiency (\( E \)) | Verification \( C[0][0] \) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sequential** | \( 4000 \times 4000 \) | 1 | **176.238106 s** | **1.00×** | **100.00%** | 4000.00 |
| **OpenMP** | \( 4000 \times 4000 \) | 8 | **38.773030 s** | **4.55×** | **56.82%** | 4000.00 |

---

## 5. Performance and Mathematical Analysis

### 5.1 Speedup Calculation
Speedup \( S \) measures the relative performance improvement of the parallel program compared to the sequential baseline:
\[
S = \frac{T_{\text{sequential}}}{T_{\text{parallel}}} = \frac{176.238106 \text{ s}}{38.773030 \text{ s}} \approx 4.54535 \approx \mathbf{4.55\times}
\]

### 5.2 Efficiency Calculation
Parallel Efficiency \( E \) evaluates how effectively the hardware threads are utilized:
\[
E = \left( \frac{S}{P} \right) \times 100\% = \left( \frac{4.54535}{8} \right) \times 100\% \approx \mathbf{56.82\%}
\]

### 5.3 Analysis of Sub-linear Speedup (\( 4.55\times \) vs. Ideal \( 8.00\times \))
While an 8-thread implementation could theoretically achieve up to \( 8.00\times \) speedup under ideal conditions, the observed efficiency of \( 56.82\% \) is expected due to the following hardware and architectural factors:
1. **Memory Bus Saturation**: Matrix multiplication is memory-intensive. Matrix \( B \) is accessed column-wise across the innermost loop (\( B[k \times N + j] \)), causing cache misses (strided access). When 8 threads concurrently read memory, the shared L3 cache and memory bandwidth become a bottleneck.
2. **Cache Coherency & False Sharing**: Although rows of \( C \) are disjoint, thread access patterns to adjacent memory boundaries can result in cache line eviction cycles.
3. **OpenMP Runtime Overhead**: Thread creation, synchronization, work-sharing distribution, and barrier completion consume clock cycles.
4. **Thermal Throttling & Dynamic CPU Frequency**: Sustained 100% CPU utilization across multiple cores may cause the CPU clock frequency to adjust downward compared to single-core boost frequencies.

---

## 6. Verification

The verification metric used is:
\[
C[0][0] = \sum_{k=0}^{3999} (1.0 \times 1.0) = 4000.00
\]
Both the Sequential and OpenMP implementations outputted:
```
Verification C[0][0] = 4000.00
```
> **Note**: While verifying \( C[0][0] = 4000.00 \) confirms that the first element was computed correctly, comprehensive validation of large datasets requires full matrix norm checks or random sampling.
