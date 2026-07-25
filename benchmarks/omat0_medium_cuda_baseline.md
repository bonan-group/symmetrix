# OMAT-0 medium float32 CUDA baseline

Date: 2026-07-24

## Configuration

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB
- Driver: 610.43.02
- Native build: CUDA 13.3.73, Kokkos CUDA, `BLACKWELL120`, SpheriCart CUDA
- PyTorch: 2.13.0+cu130
- MACE source: official revision `22f0809735bd4dd1deba80cf8e16f89913a35ff4`
- cuEquivariance: 0.10.0 (`cuequivariance`, `cuequivariance-torch`, `cuequivariance-ops-torch-cu13`)
- Checkpoint SHA-256: `d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a`
- Compact Symmetrix JSON SHA-256: `9f69ea29c0f0f25b0d3af7a485529af06cb93ca0dca7617e401fbcfe70c7baa4`
- Model: standard OMAT-0 medium, 89 elements, 128 channels, 6.0 A cutoff, `l_max=3`, `L_max=1`, ZBL enabled

The system is periodic wurtzite AlN (`a=3.112 A`, `c=4.982 A`) repeated 4, 6, and 10 times in each direction. These cells contain 256, 864, and 4,000 atoms and 23,296, 78,624, and 364,000 directed edges.

Each fresh process builds the graph once, performs 20 warmup energy-and-force forwards, and then records 20 CUDA-synchronized forwards. Stress and atomic stresses are disabled. Time therefore excludes checkpoint loading and neighbor-list construction. VRAM is the process-resident value reported by `nvidia-smi` after evaluation, not just live tensor allocation.

The upstream measurements use `benchmarks/mace_torch_cuda_benchmark.py` with `--backend e3nn` or `--backend cueq` and `--repeat 4`, `6`, or `10`. Run each combination in a fresh process. The native measurements use `benchmarks/standard_mace_streamed_benchmark.py`; the table below was collected in fresh processes per mode so process-resident VRAM is directly comparable.

## Results

| Atoms | Backend | Median (ms) | Time/atom (us) | Process VRAM (MiB) |
|---:|---|---:|---:|---:|
| 256 | PyTorch/e3nn | 19.568 | 76.439 | 3,208 |
| 256 | PyTorch/cuEquivariance | 15.151 | 59.182 | 1,146 |
| 256 | Symmetrix legacy | 14.572 | 56.923 | 1,318 |
| 256 | Symmetrix R1 streamed | 14.747 | 57.604 | 1,090 |
| 256 | Symmetrix all streamed | 14.716 | 57.485 | 998 |
| 864 | PyTorch/e3nn | 65.348 | 75.634 | 9,046 |
| 864 | PyTorch/cuEquivariance | 12.065 | 13.964 | 2,088 |
| 864 | Symmetrix legacy | 37.131 | 42.976 | 2,672 |
| 864 | Symmetrix R1 streamed | 36.109 | 41.792 | 1,904 |
| 864 | Symmetrix all streamed | 35.797 | 41.432 | 1,596 |
| 4,000 | PyTorch/e3nn | 304.240 | 76.060 | 29,198 |
| 4,000 | PyTorch/cuEquivariance | 28.862 | 7.215 | 7,070 |
| 4,000 | Symmetrix legacy | 146.085 | 36.521 | 9,690 |
| 4,000 | Symmetrix R1 streamed | 140.105 | 35.026 | 6,134 |
| 4,000 | Symmetrix all streamed | 138.980 | 34.745 | 4,710 |

## Interpretation

cuEquivariance accelerates the upstream checkpoint path by 1.29x, 5.42x, and 10.54x at 256, 864, and 4,000 atoms. Its process VRAM is lower than PyTorch/e3nn by 2,062, 6,958, and 22,128 MiB.

For the production-like medium float32 model, native full streaming is essentially neutral at 256 atoms and becomes faster at larger sizes: 3.6% at 864 atoms and 4.9% at 4,000 atoms relative to native legacy. It reduces native resident VRAM by 320, 1,076, and 4,980 MiB. Native full streaming uses 148, 492, and 2,360 MiB less memory than cuEquivariance, but cuEquivariance is 2.97x and 4.82x faster at 864 and 4,000 atoms.

The 4,000-atom PyTorch/e3nn process completed, but its caching allocator emitted recoverable 2.61 GiB allocation-failure warnings near device capacity. Its measured peak reserved memory was 30,594 MiB.

PyTorch/e3nn and cuEquivariance agree closely: their total-energy differences are at most 1.95 meV and force-L2 differences at most 2.62e-6 eV/A across these cases. Native streamed modes agree closely with native legacy at float32 reduction precision. Cross-runtime total energy is more reduction-order-sensitive: the largest native-versus-PyTorch difference is 0.477 eV total at 4,000 atoms (119 micro-eV/atom), while force-L2 differs by 3.00e-4 eV/A.

## Kokkos-CUDA reverse-Phi1 optimization

Nsight Systems profiling at 864 atoms identified the second streamed reverse-Phi1 contraction as the dominant kernel: 22.273 ms per call and 68.1% of total device-kernel time. Its original launch assigned one team to each receiver atom, leaving only 864 teams to process 78,624 directed edges.

The retained CUDA policy assigns one team to eight consecutive directed edges when the graph contains at most 100,000 edges. It builds a compact edge-to-receiver map for those graphs. Above the measured crossover it does not allocate that map and dispatches the original node-owned kernel body unchanged. OpenMP always retains the original node-owned kernel.

| Atoms | Original all (ms) | Optimized all (ms) | Time/atom (us) | Change | Process VRAM (MiB) |
|---:|---:|---:|---:|---:|---:|
| 256 | 14.716 | not accepted | not accepted | clock-sensitive | 998 |
| 864 | 35.797 | 32.965 | 38.154 | 7.9% faster | 1,598 |
| 4,000 | 138.980 | node-owned fallback | unchanged policy | no new schedule | 4,710 |

The accepted 864-atom result uses 40 warmups and 40 measured calls. A separate stable 20/20 run measured 32.116 ms, consistent with the improvement. The final 256-atom 40/40 process oscillated between 12.410 and 43.839 ms as GPU boost state changed, so its 18.985 ms median is not used. The final 4,000-atom fallback process likewise ran in a shifted clock regime (145.412 ms); because the retained large-graph hot loop and launch policy are source-identical to the baseline, this is recorded as environmental variability rather than an optimization result.

Rejected experiments were one team per edge (31.930 ms at 864 atoms but 142.466 ms at 4,000 atoms), eight edges per team globally (32.116 and 140.822 ms), 64 edges per team at large size (144.255 ms), and a schedule branch inside the hot edge loop (144.933 ms). The final implementation keeps the small-graph gain without applying any of those changed schedules at 4,000 atoms.

## CPU results at 256 atoms

The CPU host is an AMD Ryzen 9 9950X3D2 with 16 physical cores and 32 hardware threads. The Release CPU build enables Kokkos OpenMP+Serial and disables CUDA. OpenMP placement is `close` on `cores`, and BLAS is pinned to one thread. Native serial supports only float64; Kokkos uses float32. Each mode was measured in a fresh process with 20 warmups and 20 measured calls.

| Backend | Threads | Mode | Dtype | Median (ms) | Time/atom (us) | RSS after/peak (MiB) |
|---|---:|---|---|---:|---:|---:|
| Native serial | 1 | legacy | float64 | 713.245 | 2,786.114 | 2,639.0 |
| Native serial | 1 | R1 streamed | float64 | 920.750 | 3,596.680 | 2,183.1 |
| Native serial | 1 | all streamed | float64 | 972.356 | 3,798.266 | 2,109.6 |
| Kokkos OpenMP | 1 | legacy | float32 | 1,217.556 | 4,756.079 | 849.1 |
| Kokkos OpenMP | 1 | R1 streamed | float32 | 1,512.273 | 5,907.318 | 713.2 |
| Kokkos OpenMP | 1 | all streamed | float32 | 1,571.435 | 6,138.418 | 711.3 |
| Kokkos OpenMP | 2 | legacy | float32 | 622.995 | 2,433.576 | 848.1 |
| Kokkos OpenMP | 2 | R1 streamed | float32 | 767.216 | 2,996.939 | 711.3 |
| Kokkos OpenMP | 2 | all streamed | float32 | 797.383 | 3,114.776 | 712.9 |
| Kokkos OpenMP | 8 | legacy | float32 | 187.954 | 734.195 | 850.2 |
| Kokkos OpenMP | 8 | R1 streamed | float32 | 216.446 | 845.491 | 711.3 |
| Kokkos OpenMP | 8 | all streamed | float32 | 217.134 | 848.179 | 711.3 |
| Kokkos OpenMP | 16 | legacy | float32 | 120.231 | 469.654 | 849.7 |
| Kokkos OpenMP | 16 | R1 streamed | float32 | 136.304 | 532.438 | 712.7 |
| Kokkos OpenMP | 16 | all streamed | float32 | 130.536 | 509.904 | 712.5 |

Every CPU process started at approximately 68.3 MiB RSS. At 16 threads, scaling relative to one-thread Kokkos is 10.13x/11.10x/12.04x for legacy/R1/all. Full streaming reduces RSS by approximately 137 MiB versus Kokkos legacy but costs 8.6% median latency at 16 threads. The 16-thread results are noisier than the lower thread counts: sample ranges are 114.672-125.202, 127.727-147.914, and 119.853-150.150 ms for legacy/R1/all.
