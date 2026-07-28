# `symmetrix`

To build the `symmetrix` Python package:

```
git clone --recursive https://github.com/wcwitt/symmetrix
cd symmetrix/symmetrix
pip install .
```

If CUDA is not detected, the defaults will build a CPU-only version, and the `use_kokkos`
flag to the ASE calculator will switch between non-Kokkos-serial and Kokkkos-OpenMP
CPU implementations.

If CUDA is available at build time, the defaults should produce a Kokkos-CUDA GPU version
of the package. The `use_kokkos` flag to the ASE calculator
will then switch between non-Kokkos CPU and Kokkos-CUDA GPU implementations.

For other build types, `CMake` settings need to be specified explicitly, and
they can be passed as arguments to the `pip install` command, e.g.
```
pip install --verbose . \
    --config-settings=cmake.define.CMAKE_BUILD_TYPE=Release \
    --config-settings=cmake.define.CMAKE_CXX_FLAGS="-march=native -ffast-math" \
    --config-settings=cmake.define.CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    --config-settings=cmake.define.Kokkos_ENABLE_SERIAL=ON  \
    --config-settings=cmake.define.Kokkos_ENABLE_CUDA=ON  \
    --config-settings=cmake.define.Kokkos_ARCH_NATIVE=ON  \
    --config-settings=cmake.define.Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON  \
    --config-settings=cmake.define.SYMMETRIX_KOKKOS=ON  \
    --config-settings=cmake.define.SYMMETRIX_SPHERICART_CUDA=ON
```

The manually dispatched `CUDA qualification` GitHub Actions workflow builds
this configuration and runs either the focused production-path checks or the
complete MACE-MH-1 suite. It requires a Linux x64 self-hosted runner with the
`cuda` label, an NVIDIA GPU, and `nvcc` on `PATH`; ordinary pull requests do not
wait for that optional hardware runner.

### Generating Symmetrix `.json` model files

Once the Python package is installed, use
```
symmetrix_extract_mace --model my-mace.model
```
from the command line to extract a `.json` file from a Torch-based model.
The default compact output is `my-mace-universal.json`. It retains every
element in the checkpoint and can be reused across compositions without
conversion. At runtime, Symmetrix materializes radial splines only for the
elements present in the current structure.

To make a smaller compact artifact, select a subset explicitly:
```
symmetrix_extract_mace --model my-mace.model --atomic-numbers 1 8
```
This produces `my-mace-1-8.json`, which is suitable only for H/O structures.

For multi-head models, choose the head explicitly:
```
symmetrix_extract_mace --model my-mace.model --head mp-dielectric
```

Legacy-interaction compact files use Symmetrix format version 2. Nonlinear
MACE-MH-1-family models use the separate format-version-3 `MACE_Nonlinear`
schema. The spline resolution used for the transient active-composition cache
defaults to 256 nodes and can be set with `--num-spline-points`. To generate
the previous pair-table format for a legacy interaction model, provide an
explicit element list and request it directly:
```
symmetrix_extract_mace --model my-mace.model \
    --atomic-numbers 1 8 \
    --radial-format pair-splines
```

#### Backward compatibility

Existing unversioned JSON artifacts remain supported and are interpreted as
format version 1. Compact version 2 is the new converter default, including
when an explicit species subset is provided. Older Symmetrix installations
cannot read version 2 artifacts.

At runtime, compact version-2 MACE and MACEField models default to fully
streamed `R0` and `R1` execution (`streamed_edges="all"`). Version-1
pair-spline models retain legacy execution and emit a migration warning. Pass
`streamed_edges="legacy"`, `"r1"`, or `"all"` explicitly to override the
automatic selection where the model format supports it.

Standard MACE format-version-1 evaluators and compact format-version-2 MACE and
MACEField evaluators support both `dtype="float64"` and `dtype="float32"` with
`use_kokkos=False` or `use_kokkos=True`. The native CPU classes are exposed as
`MACE` (float64) and `MACEFloat` (float32). Coordinates, distances, and electric
fields enter through the existing float64 interface; learned tensors and
evaluator workspaces use the selected precision, and ASE results are promoted
to NumPy float64 during assembly. Format-version-1 pair splines are available
only for standard MACE. Compatible two-layer MACE-MH-1-family format-version-3
models also support float32 through Kokkos; generic nonlinear format-version-3 models and
the native serial nonlinear evaluator remain float64-only. The native LAMMPS
pair style also remains float64-only; its Kokkos styles retain their existing
precision selection.

To generate version 1 data for an older reader or for code that consumes the
legacy `radial_spline_*` keys, request pair splines explicitly. The equivalent
Python API is:

```python
from symmetrix.extract_mace_data import extract_mace_data

data = extract_mace_data(
    "my-mace.model",
    species=[1, 8],
    radial_format="pair-splines",
)
```

An explicit species subset is strongly recommended for version 1 output
because persisted pair tables scale quadratically with the number of retained
elements. Updated Symmetrix readers support both unversioned version 1 and
compact version 2 files.

### MACE-MH-1 checkpoints

`Symmetrix` accepts MACE-MH-1 `.model` checkpoints through the ASE calculator.
The `RealAgnosticResidualNonLinearInteractionBlock` graph is evaluated natively
by both the serial and Kokkos backends, including analytic forces and stress.
The Kokkos evaluator supports `dtype="float32"` and `dtype="float64"`; the
serial evaluator currently requires float64. The default `use_kokkos=True`
selects Kokkos. Select a model head explicitly when needed:

```
from symmetrix import Symmetrix

calc = Symmetrix("mace-mh-1.model", head="matpes_r2scan", dtype="float32")
```

The serial and Kokkos CPU evaluators have specialized fast paths for the
published two-layer MACE-MH-1 architecture. They require the checkpoint's fixed
512 feature channels, 128 edge channels, `l_max=3`, correlation-three agnostic
products, gated residual irreps, LayerNorm/SiLU edge networks, and official
`uvu [128, 1]` tensor products. Head selection and supported species subsets
remain dynamic. Universal JSON with all 89 model elements is supported without
providing an explicit species list.

For this strict architecture, both CPU evaluators default to
`streamed_edges="all"`. The modes have layer-level semantics: `legacy` retains
full-edge intermediate state for both interactions, `r1` streams only the
second interaction, and `all` streams both interactions. Streaming evaluates
the conditioned edge networks and tensor products in bounded blocks of at most
1024 directed edges on CPU or 16384 directed edges on CUDA, accumulates their
node contributions immediately, and recomputes each block during analytic
reverse propagation. Shared radial and angular geometry remains available for
the final force chain rule. Evaluator
properties `edge_workspace_rows` and, for Kokkos, `edge_workspace_bytes` expose
the retained layer-edge workspace. Changing a Kokkos evaluator's mode first
completes queued device work; transitions to `r1` or `all` then release the
corresponding grow-only full-edge capacities.
On strict Float32 CUDA models, the default specialization also selects shared
packed SGEMM for equivariant linears and 128-thread sample teams for the
official tensor products. CPU, Float64, and related nonlinear layouts retain
their existing dispatch. The selected paths and complete precision-owned
scratch are available through `e3_linear_backend`,
`selected_e3_linear_backend(samples)`, `tensor_product_backend`,
`tensor_product_execution_backend`, `linear_workspace_bytes`,
`tensor_workspace_bytes`, and `precision_workspace_bytes`. The configured
linear policy and its sample-count-dependent effective selection are reported
separately.

Related format-version-3 nonlinear models that do not satisfy the complete
fast-path predicate remain in `legacy` mode and reject explicit `r1` or `all`
requests. The streamed MH-1 specialization is qualified in float64 for native
serial, Kokkos Serial, Kokkos OpenMP, and Kokkos CUDA execution. CUDA energy,
analytic forces, and stress have been qualified against native CPU float64 on
an RTX 5090 with CUDA 13.3. The same Kokkos specialization is precision
templated for float32: learned parameters, radial/angular features,
equivariant intermediates, tapes, and adjoints are four-byte values, while
double coordinates and published ASE outputs preserve the existing API. On an
864-atom CUDA qualification graph, Float32 agrees with Float64 within
`2.800e-6 eV/atom` in energy and `2.180e-6 eV/A` in a force component while
halving the retained precision-owned workspace. Timings and absolute memory
are recorded in [the MH-1 benchmark report](../benchmarks/mh1_streamed_edges_864.md).
Related nonlinear layouts continue to use the generic float64 evaluator and
accept only `legacy`.

| Nonlinear format-v3 model | Native serial CPU | Kokkos Serial/OpenMP | Kokkos CUDA |
|---|---|---|---|
| Published MH-1 (512/128/10, `l_max=3`) | float64: `legacy`, `r1`, `all` | float64/float32: `legacy`, `r1`, `all` | float64/float32: `legacy`, `r1`, `all`; Float32 team schedule qualified |
| Generalized MH-1 family | float64: `legacy`, `r1`, `all` | float64/float32: `legacy`, `r1`, `all` | float64/float32: `legacy`, `r1`, `all`; runtime-sized team schedule |
| Related/generic architecture | float64: `legacy` | float64: `legacy` | float64: `legacy` |

The generalized family retains exactly two nonlinear residual interactions,
correlation-three element-agnostic products with skip connections, external
weighted `uvu` convolution tensor products, SiLU/sigmoid gates, and linear then
SiLU nonlinear readouts. Node and edge multiplicities, Bessel count, radial MLP
widths, species count, and readout widths are runtime values. `l_max=2` and
`l_max=3` are supported; other depths, correlations, product modes, tensor
connection modes, and angular cutoffs stay on the generic Float64 path or fail
schema validation.

The Kokkos CPU path is selected by the default `use_kokkos=True`. It shares the
serial model-load compiler for sparse product coefficients, then executes
native Kokkos kernels for sparse products and tensor products, compact
species-conditioned affine networks, packed equivariant linears, and analytic
reverse propagation. `calculator.evaluator.is_mh1_family` reports the relational
model contract, while `uses_mh1_fast_path` additionally requires every compiled
product, conditioned MLP, and tensor primitive. The `mh1_node_channels`,
`mh1_edge_channels`, `mh1_radial_size`, and `mh1_l_max` properties report the
selected dimensions. On Float32 CUDA, tensor-product teams are rounded to a
32-thread warp and capped at 128 threads; the effective channel and harmonic
team sizes are reported by `tensor_product_channel_team_size` and
`tensor_product_harmonic_team_size`. Configure CPU
parallelism before Python initializes Kokkos, for example:

```
OMP_NUM_THREADS=4 KOKKOS_NUM_THREADS=4 python my_calculation.py
```

Keep BLAS single-threaded when Kokkos OpenMP provides the outer parallelism to
avoid oversubscription.

On a 12th Gen Intel Core i9-12900HK, a Release build with CUDA disabled, pinned
physical cores, and single-threaded OpenBLAS produced the following warmed
evaluator medians in milliseconds, including energies and analytic forces:

| Atoms | Serial CPU | Kokkos 1 thread | Kokkos 2 threads | Kokkos 3 threads | Kokkos 4 threads |
|---:|---:|---:|---:|---:|---:|
| 2 | 16.1 | 87.8 | 49.0 | 36.3 | 30.7 |
| 16 | 86.2 | 315.1 | 178.5 | 137.3 | 111.5 |
| 54 | 281.9 | 870.3 | 505.9 | 379.2 | 301.5 |
| 128 | 718.1 | 1980.8 | 1146.4 | 866.2 | 676.3 |

The native serial backend is faster through 54 atoms; four-thread Kokkos is
about 5.8 percent faster at 128 atoms. Kokkos one-to-four-thread speedups range
from 2.83x to 2.93x. After capacities were warmed through the 54-atom graph, 40
alternating 2/54-atom evaluations held resident memory constant at 880.4 MiB on
that host. For example, run the following with a suitable physical-core list to
reproduce the timing, thread-affinity metadata, and lifecycle RSS result:

```
SYMMETRIX_BENCHMARK_THREADS=4 python benchmarks/mh1_serial_benchmark.py \
    mh1.json --backend kokkos --cpus 0,2,4,6 \
    --lifecycle-sizes 1,3 --lifecycle-cycles 20 \
    --max-lifecycle-growth-mib 16
```

The non-Kokkos specialization remains available with:

```python
calc = Symmetrix(
    "mace-mh-1.model",
    head="matpes_r2scan",
    dtype="float64",
    use_kokkos=False,
)
```

Related nonlinear models that do not match the complete MH-1 architecture use
the generic native evaluator for the selected backend. An explicit
`use_kokkos=True` request is never redirected to the serial evaluator. The
specialized Kokkos path is available with Kokkos Serial, OpenMP, and CUDA in
both float32 and float64. Float32 construction fails closed unless the model
matches the complete generalized MACE-MH-1-family fast-path predicate.

Native nonlinear evaluator objects retain mutable forward tapes and grow-only
workspaces. Calls on one evaluator instance must be serialized; use a separate
calculator/evaluator instance for each concurrently executing native thread.

Loading a raw `.model` checkpoint requires `mace-torch` for checkpoint
extraction. It is not used to evaluate energies or derivatives. To run without
`mace-torch`, extract the selected head once and use the resulting JSON:

```
symmetrix_extract_mace --model mace-mh-1.model \
    --head matpes_r2scan \
    --output mace-mh-1-matpes-r2scan.json
```

Format-version-3 `MACE_Nonlinear` JSON can be used through the ASE calculator
or the native `MACENonlinear`, `MACENonlinearKokkos`, and family-qualified
`MACENonlinearKokkosFloat` library classes. The LAMMPS pair styles do not
support this model family in the current release and fail at `pair_coeff` with
an explicit unsupported-model error.

Native extraction supports the standard per-layer `LinearReadoutBlock` and
SiLU `NonLinearReadoutBlock` layout used by MACE-MH-1. Checkpoints using joint
embeddings, embedding readouts, `use_last_readout_only`, biased nonlinear
readouts, or different interaction/readout gate activations are rejected with
an explicit compatibility error rather than evaluated with changed semantics.

### ASE Calculator

One can import the ASE calculator with
```
from symmetrix import Symmetrix
```
MACEField `.json` models can be evaluated through the native CPU backend or
with `use_kokkos=True` when Symmetrix is built with Kokkos support. In the ASE calculator this path
supports field-aware energies, forces, polarization, Born effective charges,
and polarizability for graph-level electric fields with either supported dtype.
See [the source code](source/symmetrix/symmetrix_calc.py) and [this test](test/test_symmetrix_calc.py)
for additional details.

### ASE Calculator with MACEField models

MACEField models must be converted to Symmetrix JSON before they are passed to
`Symmetrix`. Passing an original PyTorch MACEField `.model` checkpoint directly
to the ASE calculator raises an error instead of silently delegating back to
PyTorch.

For the MACEField dielectric models, retain the dielectric head. One universal
JSON can then be used for AlN, MgO, or any other composition whose elements are
supported by the checkpoint:
```
symmetrix_extract_mace --model MACEField-MH-0-omat-dielectric.model \
    --head mp-dielectric \
    --output macefield-dielectric-universal.json
```

The output JSON is the file used by the ASE calculator. It can run in float32 or
float64 through either the native serial evaluator or the field-aware Kokkos
evaluator when Symmetrix is built with Kokkos support and `use_kokkos=True`.

The original PyTorch checkpoint does not need to be trained or saved in double
precision. `symmetrix_extract_mace` loads the checkpoint and extracts the
Symmetrix JSON data in double precision, so a float32-trained MACEField model
can be converted once and then loaded into either runtime precision. Float32
weights are range-checked while the model is loaded.

```python
import numpy as np
from ase.build import bulk
from symmetrix import Symmetrix

atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
atoms.info["electric_field"] = np.array([0.01, -0.02, 0.03])

atoms.calc = Symmetrix(
    "macefield-dielectric-universal.json",
    use_kokkos=False,
    dtype="float64",
)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
polarization = atoms.calc.get_property("polarization", atoms)
becs = atoms.calc.get_property("becs", atoms)
polarizability = atoms.calc.get_property("polarizability", atoms)
```

The calculator updates its active radial cache when the composition changes,
so the same calculator instance can be assigned to a different supported
structure. Evaluator instances are stateful and should not be used by
concurrent host calls.

The response properties are computed selectively. A plain
`atoms.get_potential_energy()` or `atoms.get_forces()` call does not compute or
cache BECs or polarizability, which avoids second-derivative overhead during
molecular dynamics. Request response properties explicitly with
`atoms.calc.get_property(...)`, or with ASE's multi-property API:
```python
results = atoms.get_properties(["energy", "forces", "polarization"])
```

Set the electric field the same way as in the upstream MACEField ASE
calculator. Use a three-component vector in V/A:
```python
calc = Symmetrix(
    "macefield.json",
    electric_field=np.array([0.01, 0.0, 0.0]),
    use_kokkos=False,
    dtype="float64",
)
```

The calculator-level field is a global override for every calculation. It can
also be changed after construction, which is useful for finite-field dynamics:
```python
atoms.calc = calc
calc.electric_field = [0.0, 0.0, Ez_t]
```

If no calculator-level override is set, Symmetrix reads the field from the ASE
structure, following the upstream MACEField priority order:
```python
atoms.info["electric_field"] = [0.0, 0.0, 0.02]
```
or, for datasets that store the reference key:
```python
atoms.info["REF_electric_field"] = [0.0, 0.0, 0.02]
```

If none of these are set, Symmetrix uses a zero electric field.

The MACEField response properties follow the upstream ASE calculator shapes:

- `polarization`: shape `(3,)`
- `becs`: shape `(natoms, 9)`, with the polarization and Cartesian components
  flattened for each atom
- `polarizability`: shape `(9,)`

The native implementation computes polarization from the electric-field adjoint,
polarizability from the analytic graph-field Hessian, and BECs from the analytic
field derivative of forces. Finite differences are used in tests as an oracle,
not in the production ASE path.

Symmetrix also exposes upstream-compatible `node_energy`. ASE `energies` include
the atomic reference terms, while `node_energy` subtracts those atomic reference
energies to match the upstream MACEField calculator.

### Combining MACEField with another ASE potential

`FieldContributionCalculator` exposes the exact field-dependent part of a
MACEField model. For every additive property `Q`, it evaluates the same model at
the requested and zero electric fields and returns `Q(E) - Q(0)`. The correction
therefore vanishes exactly at zero field while retaining polarization, BECs, and
polarizability from the field-aware model.

```python
from symmetrix import FieldContributionCalculator, Symmetrix

field_model = Symmetrix(
    "macefield-dielectric.json",
    use_kokkos=True,
    dtype="float64",
)
atoms.calc = FieldContributionCalculator(field_model)

field_energy = atoms.get_potential_energy()
field_forces = atoms.get_forces()
field_stress = atoms.get_stress()
```

`FieldAwareCalculator` adds that correction to any field-independent ASE
calculator. The baseline supplies the zero-field energy landscape, forces,
phonons, and elastic response, while MACEField supplies the finite-field
coupling and electrical response.

```python
from mace.calculators import MACECalculator
from symmetrix import FieldAwareCalculator, Symmetrix

base = MACECalculator(model_paths=["more-accurate-mechanical.model"])
field_model = Symmetrix(
    "macefield-dielectric.json",
    use_kokkos=True,
    dtype="float64",
)
atoms.calc = FieldAwareCalculator(
    base,
    field_model,
    electric_field=[0.01, -0.02, 0.03],
)

total_energy = atoms.get_potential_energy()
total_forces = atoms.get_forces()
total_stress = atoms.get_stress()
born_effective_charges = atoms.calc.get_property("becs", atoms)

# The override remains mutable for finite-field simulations.
atoms.calc.electric_field = [0.02, -0.02, 0.03]
```

At nonzero field, an exact correction needs two MACEField evaluations for each
new geometry, plus one baseline evaluation. The zero-field MACEField result is
cached across field-only changes, and zero-field additive requests skip the
MACEField evaluation entirely. The baseline must not contain its own electric
field coupling, otherwise that coupling would be counted twice.
