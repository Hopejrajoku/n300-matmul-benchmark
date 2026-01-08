# Tenstorrent Wormhole N300: Hardware Bring-up & MatMul Performance

This repository documents the successful initialization and performance validation of the **Tenstorrent Wormhole N300** AI accelerator. By leveraging the `tt-metal` and `ttnn` software stack, we achieved high-speed matrix multiplication (MatMul) across a dual-chip architecture.

---

## System Overview
* **Accelerator:** Tenstorrent Wormhole N300 (Dual-Chip)
* **Architecture:** `wormhole_b0`
* **Compute Engine:** 64 Tensix Cores per device (8x8 Grid)
* **Precision:** BFloat16 (Native)

### Hardware Verification
Before benchmarking, the silicon health and connectivity were verified via the `tt-smi` utility.

>### Hardware & Validation Proof
<p align="center">
  <img src="https://github.com/user-attachments/assets/fa3dc79c-987d-46b5-902f-719bf627bdab" width="32%" />
  <img src="https://github.com/user-attachments/assets/88215fa1-834e-4149-a7eb-a878742e7771" width="32%" />
  <img src="https://github.com/user-attachments/assets/1b20255a-bc28-4d13-b63c-1136c0bf9ba2" width="32%" />
</p>
<p align="center"><em>Fig 1: Chip harvesting (Left), Python initialization (Center), and the final Warm-Start benchmark (Right).</em></p>

## Performance Results: 1024x1024 MatMul
We evaluated the dispatch latency and execution speed of a $1024 \times 1024$ matrix multiplication. The results demonstrate the significant performance gain once kernels are resident in the chip's L1 SRAM.

| Execution Phase | Latency | Description |
| :--- | :--- | :--- |
| **Cold Start (JIT)** | **1082.00 ms** | Includes kernel compilation and initial SRAM allocation. |
| **Warm Start (Cached)** | **8.23 ms** | Raw hardware execution time using pre-compiled kernels. |
| **Accuracy Check** | **1024.0** | Mathematical validation (Expected sum for 1024 elements). |

### Technical Analysis
1. **Just-In-Time (JIT) Compilation:** The 1-second delay in the first run represents the `ttnn` compiler generating optimized RISC-V binaries for the Tensix processors. 
2. **99.2% Latency Reduction:** The drop to 8.23ms confirms that once kernels are cached, execution occurs with near-zero software overhead.
3. **Spatial Parallelism:** By targeting an 8x8 `CoreGrid`, we effectively parallelized the workload across 64 independent compute engines, maximizing the N300's spatial compute architecture.

---

## How to Reproduce
1. **Activate Environment:**
   ```bash
   source /opt/venv/bin/activate
   export ARCH_NAME=wormhole_b0
