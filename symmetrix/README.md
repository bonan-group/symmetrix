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

### Generating Symmetrix `.json` model files

Once the Python package is installed, use
```
symmetrix_extract_mace --model my-mace.model --atomic-numbers 1 8
```
from the command line to extract a `.json` file from a Torch-based model.
The result will be `my-mace-1-8.json`, and this model is only suitable
for simulations involving H and O.

For multi-head models, choose the head explicitly:
```
symmetrix_extract_mace --model my-mace.model --atomic-numbers 8 14 22 56 --head mp-dielectric
```

### ASE Calculator

One can import the ASE calculator with
```
from symmetrix import Symmetrix
```
MACEField `.json` models can be evaluated with `use_kokkos=True` when
Symmetrix is built with Kokkos support. In the ASE calculator this path
supports field-aware energies, forces, polarization, Born effective charges,
and polarizability for graph-level electric fields with `dtype="float64"`.
See [the source code](source/symmetrix/symmetrix_calc.py) and [this test](test/test_symmetrix_calc.py)
for additional details.

### ASE Calculator with MACEField models

MACEField models must be converted to Symmetrix JSON before they are passed to
`Symmetrix`. Passing an original PyTorch MACEField `.model` checkpoint directly
to the ASE calculator raises an error instead of silently delegating back to
PyTorch.

For the MACEField dielectric models, include every atomic number that can appear
in the ASE structures and keep the dielectric head. For example, an AlN-only
JSON can be extracted with:
```
symmetrix_extract_mace --model MACEField-MH-0-omat-dielectric.model \
    --atomic-numbers 7 13 \
    --head mp-dielectric \
    --output macefield-dielectric-7-13.json
```

The output JSON is the file used by the ASE calculator. MACEField JSON requires
`dtype="float64"`. It can run through either the native serial evaluator or the
field-aware Kokkos evaluator when Symmetrix is built with Kokkos support and
`use_kokkos=True`.

The original PyTorch checkpoint does not need to be trained or saved in double
precision. `symmetrix_extract_mace` loads the checkpoint and extracts the
Symmetrix JSON data in double precision, so a float32-trained MACEField model
can still be used by converting it first and running the resulting JSON with
`dtype="float64"`.

```python
import numpy as np
from ase.build import bulk
from symmetrix import Symmetrix

atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
atoms.info["electric_field"] = np.array([0.01, -0.02, 0.03])

atoms.calc = Symmetrix(
    "macefield-dielectric-7-13.json",
    use_kokkos=False,
    dtype="float64",
)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
polarization = atoms.calc.get_property("polarization", atoms)
becs = atoms.calc.get_property("becs", atoms)
polarizability = atoms.calc.get_property("polarizability", atoms)
```

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
