# MACE-MH-1 streamed-edge CPU and CUDA benchmark

This benchmark compares the full-edge and block-streamed implementations of
the strict two-layer MACE-MH-1 fast path. Results include native energy and
analytic-force evaluation on a prebuilt graph.

## Configuration

- Base commit: `bab66d5d7a7ffad69a66c956bad2511b00a1b697`
- Worktree state: dirty with the streamed-edge implementation under test
- Model: `mace-mh-1-omat-pbe-universal-v3.json`
- Model SHA-256: `8384b616054cc4391531ca95af7f9b4737ce902628701faaf8e54878b5a79f00`
- System: 864-atom periodic wurtzite AlN, 78,624 directed edges
- CPU: AMD Ryzen 9 9950X3D2 16-Core Processor
- Backends: native serial on CPU `0`; Release Kokkos OpenMP on four pinned
  physical CPUs (`0-3`)
- Threading: one BLAS thread; four Kokkos/OpenMP threads for Kokkos
- Protocol: fresh process per mode, three warmups, ten measured repeats
- Streaming block size: 1024 directed edges

## Results

| Backend | Mode | Median (ms) | Median (ms/atom) | Final RSS (MiB) | Peak RSS (MiB) |
|---|---|---:|---:|---:|---:|
| Serial | `legacy` | 6081.596 | 7.03888 | 2348.645 | 5913.129 |
| Serial | `all` | 5519.110 | 6.38786 | 480.770 | 1241.223 |
| Kokkos OpenMP (4) | `legacy` | 5306.958 | 6.14231 | 20295.641 | 20299.949 |
| Kokkos OpenMP (4) | `all` | 5186.027 | 6.00235 | 2614.703 | 2617.578 |

For serial, `all` is 9.25% faster, final RSS is lower by 1,867.875 MiB
(79.53%), and peak RSS is lower by 4,671.906 MiB (79.01%). For four-thread
Kokkos, `all` is 2.28% faster, final RSS is lower by 17,680.938 MiB (87.12%),
and peak RSS is lower by 17,682.371 MiB (87.11%). Every row reports the same
energy (`-6423.653034540963 eV`) and force norm
(`3.1685322548340 eV/A`) to the precision printed by the benchmark.

The benchmark driver records the base commit, dirty state, build cache, model
hash, individual timing samples, total and per-atom timing, process RSS, and
result checksums. Reproduce a row with:

```bash
SYMMETRIX_BENCHMARK_THREADS=THREADS \
SYMMETRIX_BENCHMARK_BLAS_THREADS=1 \
python benchmarks/mh1_serial_benchmark.py MODEL.json \
    --backend BACKEND --cpus CPU_LIST --atom-counts 864 \
    --streamed-edges MODE --warmups 3 --repeats 10 --evaluator-only
```

## CUDA Float32 Results

The strict `omat_pbe` head was requalified in Float32 on an NVIDIA RTX 5090
with driver 610.43.02, CUDA 13.3, and the same 864-atom/78,624-edge AlN graph.
The Symmetrix rows use base commit
`543def66b9195e45d27e12541ce1d886bbe5529f` with the uncommitted Float32
implementation, a 16,384-edge CUDA block, three warmups, and ten measured
evaluator calls. The upstream rows use the official checkpoint with MACE
0.3.15, PyTorch 2.13.0+cu130, and cuEquivariance 0.10.0.

| Implementation | Median (ms) | Median (ms/atom) | Process VRAM after setup (MiB) | Process VRAM after evaluation (MiB) | Peak allocated/workspace (MiB) |
|---|---:|---:|---:|---:|---:|
| Symmetrix `legacy` | 158.247 | 0.18316 | 506 | 10,584 | 8,955.813 workspace |
| Symmetrix `all` | 175.196 | 0.20277 | 506 | 3,574 | 1,866.000 workspace |
| PyTorch/e3nn | 95.762 | 0.11084 | 682 | 9,900 | 7,974.226 allocated |
| PyTorch/cuEquivariance | 20.793 | 0.02407 | 666 | 3,986 | 2,462.438 allocated |

`all` reduces post-evaluation process VRAM by 7,010 MiB (66.23%) and retained
edge workspace by 79.16% versus `legacy`. The ten `legacy` samples are stable
between 158.138 and 158.829 ms. The streamed samples are affected by device
scheduling variance: the protocol run spans 165.612-230.453 ms, while a
separate one-warmup smoke is 155.241 ms and a ten-warmup diagnostic remains
bimodal. The table retains the declared three-warmup/ten-repeat median rather
than selecting the fastest run.

Full-array CUDA Float32 versus Float64 comparison gives maximum differences of
`2.800e-6 eV/atom` in total energy, `6.162e-6 eV` in per-atom energy,
`2.180e-6 eV/A` in a force component, and `2.486e-7` in stress. At the same
block size, the precision-owned retained workspace is exactly halved from
3,913,285,632 bytes to 1,956,642,816 bytes.

An Nsight Systems trace of one warmup plus one measured `all` call records
1,228 CUDA kernel launches and 1,665 device synchronizations. E3 tensor-product
forward/reverse and E3 linear forward/reverse account for about 75% of GPU
kernel time. Consequently, the remaining cuEquivariance gap requires more
coarse-grained or fused equivariant kernels; further affine-only tuning or a
larger streamed block does not address the dominant cost. Tested 8,192- and
32,768-edge blocks regress to 254.163 and 207.055 ms respectively.

## CUDA equivariant-kernel optimization

The strict Float32 CUDA `all` path was subsequently optimized and committed as
`f9f020969a1cecc4625ccd074b36a20783db2436`. The tracked source was clean at
that commit; generated build trees and local planning files remained untracked.
The build uses CUDA 13.3.73, Kokkos CUDA for `BLACKWELL120`, KokkosKernels with
cuBLAS, and the same RTX 5090/driver 610.43.02. The system, graph, model hash,
three warmups, ten measured calls, explicit device fences, Float32 precision,
and 16,384-edge block are unchanged.

| Implementation | Median (ms) | Median (ms/atom) | Process VRAM before/after (MiB) | Precision workspace (MiB) |
|---|---:|---:|---:|---:|
| Original synchronized scalar `all` | 174.718 | 0.202220 | 506 / 3,574 | 1,866.00 |
| Optimized Symmetrix `all` (`f9f0209`) | 62.133 | 0.071913 | 506 / 3,184 | 1,477.06 |
| PyTorch/cuEquivariance 0.11.0 | 22.823 | 0.026416 | 0 / 3,616 | 2,456.90 peak allocated |

The optimized path is 2.81x faster than the synchronized scalar control, a
64.4% latency reduction. Process VRAM falls by 390 MiB (10.9%), and total
precision-owned workspace falls by 388.94 MiB (20.8%) even after including
the new 59.06 MiB shared linear workspace. It remains 2.72x slower than the
fresh same-session cuEquivariance result, so it clears the plan's 2x
Symmetrix speed gate but not its 1.25x cuEquivariance stretch target.

The retained stages are shared packed-SGEMM storage for E3 linears, 128-thread
sample teams for all official tensor-product forward/reverse channel work,
parallel harmonic reverse reductions, CUDA execution-space-ordered internal
copies, and lifetime reuse of forward edge-message storage by reverse
edge-message adjoints. Float64, Kokkos CPU, and generic nonlinear layouts keep
their existing selection rules. Rejected controls include workspace-free
direct-tiled linears, split tensor instruction/channel kernels, adjacent-power
radial derivatives, CUDA team LayerNorm/SiLU, and a 20,480-edge block.

The optimized ten samples in milliseconds are:
`[62.153722, 62.110902, 62.108758, 62.120200, 64.423274, 65.991663,
63.740525, 62.128926, 62.136410, 62.128215]`. The matched cuEquivariance
samples are:
`[22.775048, 22.822446, 22.990168, 22.784306, 22.799584, 22.825472,
22.824000, 22.791510, 22.945676, 22.828027]`.

Final qualification covers 30 CUDA physics, primitive, lifecycle, resizing,
species, finite-difference, and workspace tests; six direct OpenMP tests; and
CUDA memcheck of the larger official tensor-product layer with zero errors.
The optimized evaluator reports `e3_linear_backend="packed_gemm"`,
`tensor_product_backend="official_kokkos"`, 1,486,880,768 edge-workspace
bytes, 61,931,520 linear-workspace bytes, and 1,548,812,288 total
precision-workspace bytes.
