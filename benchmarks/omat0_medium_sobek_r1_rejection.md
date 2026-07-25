# OMAT-0 Medium Sobek R1 Prototype Rejection

## Scope

- Model: OMAT-0 medium, compact format-v2 standard MACE
- Precision/backend: float32 Kokkos CUDA
- GPU: NVIDIA GeForce RTX 5090
- System: 864 atoms, 78,624 directed edges
- CUDA: 13.3, native `sm_120`
- Candidate: 64-wide penultimate R1 receiver aggregation with a bounded
  64 MiB workspace, followed by the bias-free `1280 x 64` projection

## Correctness

The split MLP reconstruction test passed. On the four-atom CUDA fixture, the
candidate matched the projected-spline control within `2e-5 eV` in energy and
`3.12e-5 eV/Angstrom` maximum force component error. The 864-atom candidate
reported energy `-6406.387533442576 eV` and force L2 norm
`3.2379334188619118 eV/Angstrom`, consistent with float32 regrouping relative
to the accepted control.

## 864-Atom Gate

| Implementation | Median time | Time/atom | Resident VRAM |
|---|---:|---:|---:|
| Accepted native `all` (20 warmups, 10 repeats) | 33.706 ms | 39.01 us | 1,598 MiB |
| PyTorch + cuEquivariance (20 warmups, 10 repeats) | 12.452 ms | 14.41 us | 2,088 MiB |
| Penultimate-R1 prototype (cold one-sample gate) | 9,046.608 ms | 10,470.61 us | 1,796 MiB |

The candidate is about 268 times slower than native `all` and 727 times slower
than cuEquivariance. It also uses 198 MiB more resident VRAM than native `all`.
It therefore fails both the performance and memory retention gates. Ten-repeat
timing was intentionally skipped after this decisive one-sample result.

## Nsight Attribution

Nsight Systems 2026.1.3 captured one 864-atom evaluation. The forward receiver
aggregation kernel consumed `5.494750210 s` across 40 receiver chunks, or
`99.2%` of the `5.54 s` summed CUDA-kernel time. The reverse candidate consumed
about `25.25 ms`; projection, CG, and A1 mixing were individually below 5 ms in
aggregate. The remaining wall time was dominated by allocation, initialization,
and synchronization around the temporary chunk workspaces.

The raw receiver aggregate performs the 64-wide outer product for every raw
angular row and channel. For this MACE shape, scalar Kokkos execution increases
the dominant forward accumulation work by roughly the embedding width. Closing
the 864-atom acceptance threshold would require more than a 350-fold improvement
in the measured aggregation kernel. Launch tuning or source ownership cannot
plausibly recover that gap; a future attempt would require a generated,
tensor-core matrix formulation with a different contraction schedule.

## Decision

The experimental runtime branch is rejected and removed. Existing compact JSON,
`legacy`, `r1`, and `all` behavior remain unchanged. Further optimization should
start from the accepted projected-spline `all` path and measure source ownership,
reverse fusion, or custom CUDA specialization independently.
