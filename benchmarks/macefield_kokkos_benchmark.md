# MACEField streamed-edge benchmark and support matrix

## `streamed_edges="all"` support matrix

| Backend | Model | `all` | Qualified precision | Energy/forces/stress | Polarization | Polarizability/BEC | Edge-memory behavior |
|---|---|---:|---|---:|---:|---:|---|
| native CPU | compact MACE | yes | float32, float64 | yes | n/a | n/a | fully streamed R0/R1 |
| native CPU | compact MACEField | yes | float32, float64 | yes | yes | yes, analytic | first order fully streamed; response calls transiently materialize R0/R1 |
| Kokkos OpenMP | compact MACE | yes | float32, float64 | yes | n/a | n/a | fully streamed R0/R1 |
| Kokkos OpenMP | compact MACEField | yes | float32, float64 | yes | yes | yes, analytic | first order fully streamed; response calls transiently materialize R0/R1 |
| Kokkos CUDA | compact MACE | yes | float32, float64 | yes | n/a | n/a | fully streamed R0/R1 |
| Kokkos CUDA | compact MACEField | yes | float32, float64 | yes | yes | yes, analytic | first order fully streamed; response calls transiently materialize R0/R1 |

The MH-0 and MACEField rows above require the format-v2 compact, fixed-weight
radial representation. Format-v1 pair-spline MACE models reject `r1` and
`all`. Compatible format-v3 MH-1-family models support both streamed modes;
generic nonlinear format-v3 models remain legacy-only. The field-aware native
and Kokkos paths support float32 and float64 through the same analytic response
implementation instantiated at both precisions.

## Configuration

- Model: `MACEField-MH-0-omat-dielectric.model`, head `mp-dielectric`
- Checkpoint SHA-256: `f92e043aaf2cd8879919db8452503553fe7b608cb749d8d169dd96d4aa094aa2`
- Compact Al/N artifact: 256 spline points, 16,220,750 bytes
- Artifact SHA-256: `997512552ec2a19aeab1260fc8c88065c564c1e225e10e055ed0d1002d107da4`
- Structure: periodic wurtzite AlN, 91 directed neighbors per atom
- Electric field: `(0.01, -0.02, 0.03)`
- Precision: float64
- CUDA: 13.3, Kokkos `Cuda`, Blackwell `sm_120`
- CUDA timing: 20 warmups, 10 measured energy/force calls, median reported

The driver is `benchmarks/macefield_kokkos_benchmark.py`. GPU memory is the
memory attributed to the benchmark process by `nvidia-smi`. Host current and
high-water memory come from `/proc/self/status`.

## 864-atom optimization stages

| implementation | mode | ms/call | ms/atom | R0+R1 bytes | GPU before | GPU setup | GPU after | sampled GPU high-water |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| untouched control | legacy | 134.843 | 0.156068 | 2,254,307,328 | 0 MiB | 508 MiB | 4,562 MiB | 4,562 MiB |
| streamed edge kernels only | r1 | 131.492 | 0.152189 | 644,087,808 | 0 MiB | 508 MiB | 3,028 MiB | 3,028 MiB |
| streamed edge kernels only | all | 131.990 | 0.152766 | 0 | 0 MiB | 508 MiB | 2,412 MiB | 2,412 MiB |
| precomposed field transform | legacy | 43.308 | 0.050125 | 2,254,307,328 | 0 MiB | 510 MiB | 4,558 MiB | 4,558 MiB |
| precomposed field + streamed edges | all | 40.100 | 0.046412 | 0 | 0 MiB | 510 MiB | 2,406 MiB | 2,406 MiB |

The retained fully streamed implementation is 3.36x faster than the untouched
control. It reduces process GPU memory after evaluation by 2,156 MiB, from
4,562 MiB to 2,406 MiB (47.3%). Host RSS for the retained 864-atom run was
252.38 MiB before setup, 417.67 MiB after setup, and 525.34 MiB after the
evaluation, with a 529.23 MiB host high-water mark.

The field transform precomposition is responsible for most of the speedup.
Streaming R0/R1 is retained primarily for its 2.10 GiB radial-storage removal;
after precomposition it also improves the 864-atom time from 43.31 to 40.10 ms.

## CUDA scaling

| atoms | directed edges | ms/call | ms/atom | GPU before | GPU setup | GPU after | sampled GPU high-water |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 23,296 | 14.071 | 0.054967 | 0 MiB | 510 MiB | 1,210 MiB | 1,208 MiB |
| 864 | 78,624 | 40.100 | 0.046412 | 0 MiB | 510 MiB | 2,406 MiB | 2,406 MiB |
| 4,000 | 364,000 | 172.322 | 0.043081 | 0 MiB | 510 MiB | 8,636 MiB | 8,636 MiB |

The 256-atom run also timed one response call after the force benchmark:
68.16 ms for the electric-field Hessian and 65.50 ms for the electric-field
force derivative. This explains why its final GPU memory can be 2 MiB above
the high-water sampled during the single energy/force pass.

## Analytic response port

The Kokkos field Hessian and field-force derivative now use the native CPU
forward-over-reverse algorithm instead of a `1e-6` forward difference. The
implementation propagates three exact electric-field directions through H1,
Phi1, A1, M1, H2, and the readout, then differentiates the reverse pass. CUDA
uses the same streamed edge-team layout as the qualified first-order reverse
Phi1/A0 kernels. Constant field/linear-up maps are composed at model load,
and the 3x3 Hessian is accumulated by a reduction rather than global atomics.

Matched 864-atom `all` measurements use the float32 production condition and
separate latency timing from the additional `nvidia-smi` memory-sampling call.
The finite-difference control reconstructs the former baseline plus three
perturbed evaluations, including its device-to-host response copies.

| precision | response | ms/call | ms/atom | GPU before response | GPU after response | sampled response high-water |
|---|---|---:|---:|---:|---:|---:|
| float32 | analytic field-force derivative | 43.832 | 0.050731 | 1,524 MiB | 1,536 MiB | 1,536 MiB |
| float32 | analytic field Hessian, first call | 47.471 | 0.054943 | 1,524 MiB | 1,536 MiB | 1,538 MiB |
| float32 | finite-difference control | 107.758 | 0.124720 | 1,524 MiB | 1,524 MiB | 1,526 MiB |
| float64 | analytic field-force derivative | 167.453 | 0.193811 | 2,408 MiB | 2,420 MiB | 2,420 MiB |
| float64 | analytic field Hessian, first call | 172.664 | 0.199842 | 2,408 MiB | 2,420 MiB | 2,420 MiB |
| float64 | finite-difference control | 213.365 | 0.246950 | 2,420 MiB* | 2,420 MiB* | 2,420 MiB* |

`*` The float64 finite-difference control was measured after the analytic
workspace had already raised the process allocation to 2,420 MiB. Float32's
control was also run in a fresh process and retained 1,524 MiB after the
response, with a 1,526 MiB high-water.

At float32, the steady analytic response is 2.46x faster than the matched
finite-difference control. At float64 it is 1.27x faster. The analytic
workspace adds 12 MiB of retained CUDA process memory at both precisions.

### Response agreement

The benchmark also records max-absolute, RMS, and relative-L2 differences for
the complete 3x3 field Hessian and raw directed-edge field-force derivative.
At float64, the analytic path agrees directly with the former one-sided
`1e-6` finite-difference path:

| float64 quantity | max absolute difference | RMS difference | relative L2 |
|---|---:|---:|---:|
| field Hessian | 7.570e-5 | 2.562e-5 | 2.276e-7 |
| field-force derivative | 7.551e-7 | 3.453e-8 | 6.193e-7 |

The former `1e-6` perturbation is below a useful subtraction scale for
float32: its raw force-derivative relative-L2 difference is 0.970. A centered
step sweep demonstrates convergence toward the analytic result instead:

| float32 centered step | Hessian relative L2 | force-derivative relative L2 |
|---:|---:|---:|
| `1e-3` | 2.134e-4 | 9.281e-3 |
| `3e-3` | 7.040e-5 | 2.932e-3 |
| `1e-2` | 5.240e-5 | 1.023e-3 |
| `3e-2` | 3.050e-4 | 3.472e-4 |

At the balanced `1e-2` step, maximum absolute differences are 1.549e-2 for
the Hessian and 1.434e-3 for the raw force derivative. The remaining error is
finite-difference quantization/truncation, not disagreement between the two
analytic backends: permanent tests compare Kokkos float32 and float64 directly
against the native analytic implementation, while separate native tests check
the analytic equations against centered finite differences.

## OpenMP scaling

The current-source Kokkos OpenMP build used the same 864-atom artifact in
fully streamed float64 mode, with BLAS pinned to one thread, 3 warmups, and
10 measured calls.

| Kokkos threads | ms/call | ms/atom | host before | host setup | host after | host high-water |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5,340.841 | 6.181529 | 69.00 MiB | 118.98 MiB | 1,880.29 MiB | 1,882.99 MiB |
| 2 | 2,712.128 | 3.139037 | 69.22 MiB | 119.09 MiB | 1,880.18 MiB | 1,883.80 MiB |
| 8 | 756.915 | 0.876058 | 69.20 MiB | 119.29 MiB | 1,879.50 MiB | 1,881.61 MiB |
| 16 | 441.384 | 0.510862 | 69.37 MiB | 120.30 MiB | 1,881.52 MiB | 1,883.98 MiB |

The 16-thread result was remeasured from the rebuilt extension after the
native port. Its fresh-process legacy control was 487.570 ms/call
(0.564317 ms/atom), 4,031.23 MiB RSS after evaluation, and 4,033.39 MiB
high-water. Thus OpenMP `all` is 9.5% faster and lowers post-evaluation RSS by
2,149.71 MiB, from 4,031.23 to 1,881.52 MiB (53.3%).

## Native CPU before and after

The native CPU comparison used separate fresh processes, one BLAS thread,
3 warmups, and 10 measured calls on the same 864-atom float64 case.

| mode | ms/call | ms/atom | R0+R1 bytes | host before | host setup | host after | host high-water |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | 2,447.748 | 2.833042 | 2,254,307,328 | 69.18 MiB | 136.43 MiB | 4,473.07 MiB | 4,476.01 MiB |
| all | 3,384.924 | 3.917736 | 0 | 69.22 MiB | 136.43 MiB | 2,323.27 MiB | 2,326.77 MiB |

Native `all` lowers post-evaluation RSS by 2,149.80 MiB, from 4,473.07 to
2,323.27 MiB (48.1%), and lowers process high-water by 2,149.23 MiB. It is
38.3% slower than native legacy on this case. Unlike CUDA, the serial CPU
path benefits from retaining radial splines across the forward and reverse
passes; strict edge streaming must recompute those values during reverse.
Native `all` is therefore currently a memory-capacity mode, not a CPU speed
optimization.

## Numerical checks

At 864 atoms, the retained result has the same printed energy as the control,
`-6419.674540523367 eV`. Maximum-force and electric-field-adjoint differences
are approximately `1e-14`. Focused CUDA and OpenMP tests cover the standalone
field transform and reverse, full energy/forces, polarization,
polarizability, Born effective charges, field Hessian, field-force derivative,
legacy/r1/all equivalence, radial-storage release, and malformed Phi1
hidden-degree rejection. The streamed property comparison is parameterized
over native CPU and Kokkos and passed with both OpenMP and CUDA builds.

## Native float32 template qualification

The native CPU evaluator now instantiates the same standard-MACE and MACEField
algorithm at float32 and float64. These matched 864-atom `all` runs used one
native CPU thread, one BLAS thread, 10 warmups, and 10 measured first-order calls
from base commit `fd5dd4a468e8e53eadca3f157fee1ac2e017b973` plus the Phase 56
working tree. The binding confirmed four-byte learned tensors and workspaces in
the float32 evaluator and eight-byte storage in float64.

| Dtype | First order (ms) | Time/atom (ms) | RSS after (MiB) | Peak RSS incl. response (MiB) | Field Hessian (ms) | Field-force derivative (ms) |
|---|---:|---:|---:|---:|---:|---:|
| float64 | 1,293.913 | 1.497585 | 2,327.6 | 6,611.8 | 11,788.7 | 11,894.4 |
| float32 | 1,112.741 | 1.287895 | 1,244.8 | 3,369.0 | 10,772.8 | 11,501.2 |

Float32 is 1.16x faster for the first-order field-aware call, lowers final RSS
by 1,082.8 MiB (46.5%), and lowers the analytic-response high-water RSS by
3,242.7 MiB (49.0%). The analytic Hessian and field-force derivative are 1.09x
and 1.03x faster, respectively. The response calculations use the same analytic
forward-over-reverse implementation at both precisions. Permanent cross-
precision tests cover energy, forces, polarization, polarizability, Born
effective charges, the field Hessian, and the raw field-force derivative.
