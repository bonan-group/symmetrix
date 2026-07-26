# MACE-MH-1 streamed-edge CPU benchmark

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
