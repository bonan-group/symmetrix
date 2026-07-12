# Field-Aware MACE Kokkos ASE Support

## Objective

Enable field-aware MACE JSON models to run through the existing Kokkos MACE backend and through the ASE `Symmetrix` calculator with `use_kokkos=True`, matching the existing native serial MACEField behavior for energy, forces, polarization, Born effective charges, and polarizability.

The acceptance surface is the ASE calculator: a field-aware model passed to `Symmetrix(..., use_kokkos=True, dtype="float64")` should keep `calc.use_kokkos` true, instantiate `MACEKokkos`, expose the same field response properties as the native serial path, and return results that agree with native serial and existing MACEField/PyTorch reference tests within established tolerances.

## Current Code Context

The ASE calculator currently detects field-aware JSON models and forces them back to the native serial evaluator when `use_kokkos=True` at `symmetrix/source/symmetrix/symmetrix_calc.py:50`. The same calculator already has the response-property assembly that we want to reuse: it computes polarization from `electric_field_adj` at `symmetrix/source/symmetrix/symmetrix_calc.py:187`, calls `compute_electric_field_hessian` for polarizability at `symmetrix/source/symmetrix/symmetrix_calc.py:220`, and calls `compute_electric_field_force_derivative` for BECs at `symmetrix/source/symmetrix/symmetrix_calc.py:207`.

The native serial evaluator already exposes the complete MACEField response API in `libsymmetrix/source/mace.hpp:127`, including `electric_field_adj`, `electric_field_hessian`, `electric_field_force_derivative`, `compute_electric_field_hessian`, and `compute_electric_field_force_derivative`. The Kokkos evaluator currently has field-aware energy/force entry points and first field adjoint state at `libsymmetrix/source/mace_kokkos.hpp:42` and `libsymmetrix/source/mace_kokkos.hpp:133`, but it does not yet expose Hessian or field-force derivative state.

The native response derivative implementation starts at `libsymmetrix/source/mace.cpp:121`. It runs the normal field-aware forward graph through `compute_field_H1`, seeds three field-coordinate directional derivatives, propagates those derivatives through H1 linear-up, Phi1/A1/M1/H2/readout, and back-propagates force derivative contributions into `electric_field_force_derivative`. The native force-derivative method delegates to the Hessian method at `libsymmetrix/source/mace.cpp:819`, so Kokkos should also treat these as one derivative implementation with two public entry points.

The existing tests already cover native ASE responses at `symmetrix/test/test_symmetrix_calc.py:192`, native response method dispatch at `symmetrix/test/test_symmetrix_calc.py:308`, and native finite-difference response derivatives at `symmetrix/test/test_macefield_native.py:266`. A current regression test explicitly asserts the old serial fallback at `symmetrix/test/test_symmetrix_calc.py:451`; this should be converted into a Kokkos-enabled test.

## Implementation Plan

- [ ] Establish red ASE tests for the desired `use_kokkos=True` behavior. Convert the existing serial-fallback test at `symmetrix/test/test_symmetrix_calc.py:451` so that a field-aware JSON model constructed with `use_kokkos=True` keeps `calc.use_kokkos` true, exposes field response properties, and computes finite energy plus polarization. Add a monkeypatched unit test that proves `Symmetrix` selects `symmetrix.MACEKokkos` rather than `symmetrix.MACE` for a field-aware JSON when Kokkos is requested. This anchors the user-facing behavior before changing the calculator gate.

- [ ] Add ASE-level Kokkos/native equivalence tests for field-aware energy, forces, and responses. Extend the existing native response comparison flow at `symmetrix/test/test_symmetrix_calc.py:192` so the same field-aware JSON can be evaluated with `use_kokkos=False` and `use_kokkos=True`, then compare energy, forces, polarization, BECs, and polarizability. Keep PyTorch comparison as an optional reference when MACEField dependencies are available, but make native serial versus Kokkos the core backend parity assertion.

- [ ] Add lower-level Kokkos evaluator red tests for response derivative API parity. Extend `symmetrix/test/test_macefield_native.py:221` with Kokkos tests that call `compute_electric_field_hessian` and `compute_electric_field_force_derivative` on `MACEKokkos`, compare `electric_field_hessian` and `electric_field_force_derivative` against native serial outputs, and preserve the existing finite-difference checks as the native source of truth. These tests should initially fail because Kokkos lacks the methods and state.

- [ ] Promote field-aware support to a first-class optional capability of `MACEKokkos`. Add Kokkos evaluator members matching the native API at `libsymmetrix/source/mace.hpp:135`, specifically field Hessian state, field-force derivative state, and public derivative methods. The intent is API parity: the ASE calculator should be able to call the same evaluator attributes and methods regardless of whether the evaluator is native serial or Kokkos.

- [ ] Port the native response derivative pipeline into Kokkos in small, testable sections. Start from the existing native flow at `libsymmetrix/source/mace.cpp:121` and map its intermediate arrays to Kokkos views. Preserve the current field-aware forward order already represented in Kokkos at `libsymmetrix/source/mace_kokkos.hpp:128`: H1 product, field H1, H1 linear-up, then the existing Phi1/A1/M1/H2/readout path. Treat `compute_electric_field_force_derivative` as a public alias that computes both derivative outputs, matching native behavior at `libsymmetrix/source/mace.cpp:819`.

- [ ] Make reductions and pair derivative accumulation explicit and deterministic enough for tests. The native implementation accumulates field Hessian terms at `libsymmetrix/source/mace.cpp:673` and pair-level force derivative terms at `libsymmetrix/source/mace.cpp:455`, `libsymmetrix/source/mace.cpp:532`, `libsymmetrix/source/mace.cpp:752`, and `libsymmetrix/source/mace.cpp:789`. The Kokkos port should choose reductions, per-seed temporary views, or scratch/team-local buffers that avoid unsafely racing on device backends while keeping CPU/GPU behavior within tolerance.

- [ ] Expose the new Kokkos response API through pybind. Update the Kokkos binding at `symmetrix/source/cpp/mace_kokkos.cpp:37` so `MACEKokkos` and `MACEKokkosFloat` expose readonly `electric_field_hessian`, readonly `electric_field_force_derivative`, `compute_electric_field_hessian`, and `compute_electric_field_force_derivative` with the same Python call shape as native `MACE`. Even if field-aware models remain float64-only initially, keeping the template binding consistent avoids surprising Python API gaps.

- [ ] Relax ASE calculator fallback after Kokkos API parity exists. Replace the forced serial fallback at `symmetrix/source/symmetrix/symmetrix_calc.py:50` with capability-based validation. For field-aware JSON plus `use_kokkos=True`, require `MACEKokkos` to be present, require `dtype="float64"` unless float32 field-aware parity is explicitly validated, instantiate Kokkos, and let `_has_native_field_coupling` at `symmetrix/source/symmetrix/symmetrix_calc.py:134` drive the existing field-aware path.

- [ ] Reuse ASE response assembly unchanged except for backend capability checks. Keep `_compute_macefield` at `symmetrix/source/symmetrix/symmetrix_calc.py:173` and `_calculate_macefield_responses` at `symmetrix/source/symmetrix/symmetrix_calc.py:187` as the central path for both native and Kokkos evaluators. Only add guardrails if a requested evaluator is missing a derivative method, so failures explain whether Kokkos was not built, field coupling was absent, or response derivatives are unsupported.

- [ ] Preserve graph-level field semantics and cache invalidation. Keep the current graph-level field requirement for response properties at `symmetrix/source/symmetrix/symmetrix_calc.py:140` and the `atoms.info` cache invalidation behavior at `symmetrix/source/symmetrix/symmetrix_calc.py:121`. Add a Kokkos variant of the cached-field-change test at `symmetrix/test/test_symmetrix_calc.py:472` so changing `atoms.info["electric_field"]` invalidates energy, forces, and response results with `use_kokkos=True`.

- [ ] Update documentation and LAMMPS scoping. Update `pair_symmetrix/README.md` and any Python-facing documentation to state that ASE supports field-aware Kokkos energy, forces, polarization, BECs, and polarizability once the evaluator API is present. Keep LAMMPS scoped to field-aware energy and forces unless a separate LAMMPS compute/fix interface is designed for tensor outputs.

- [ ] Run focused verification in dependency-appropriate layers. First run the lower-level MACEField Kokkos tests in `symmetrix/test/test_macefield_native.py`, then the ASE calculator tests in `symmetrix/test/test_symmetrix_calc.py`, then the existing LAMMPS MACEField tests if a LAMMPS Python module is available. Also run `git diff --check` and rebuild/install the Python package before tests when C++ bindings change.

## Verification Criteria

- [ ] `Symmetrix(field_aware_json, use_kokkos=True, dtype="float64")` leaves `calc.use_kokkos` true and selects `MACEKokkos` when Kokkos bindings are available.

- [ ] The ASE calculator reports `polarization`, `becs`, and `polarizability` in `implemented_properties` for field-aware Kokkos models.

- [ ] ASE `use_kokkos=True` field-aware energy and forces agree with ASE `use_kokkos=False` native serial results within the existing field-aware tolerances.

- [ ] ASE `use_kokkos=True` polarization, BECs, and polarizability agree with ASE `use_kokkos=False` native serial results within tolerances comparable to the existing PyTorch/native tests at `symmetrix/test/test_symmetrix_calc.py:230`.

- [ ] `MACEKokkos.compute_electric_field_hessian` produces a 3 by 3 field Hessian matching native serial for the same model, geometry, and electric field.

- [ ] `MACEKokkos.compute_electric_field_force_derivative` produces a per-field-component, per-pair, per-Cartesian derivative tensor matching native serial for the same model, geometry, and electric field.

- [ ] Plain non-field MACE models still run through the existing Kokkos path without requiring an electric field and without exposing field response properties.

- [ ] Atom-wise electric fields remain accepted for energy/forces if currently supported, while response properties still require a graph-level field and raise `PropertyNotImplementedError` otherwise.

## Potential Risks and Mitigations

**Risk: Kokkos derivative implementation diverges numerically from native serial because reductions happen in a different order.**  
Mitigation: Compare against native serial with realistic tolerances, prefer deterministic per-seed reductions for small 3-component field outputs, and isolate pair derivative accumulation so expected floating-point differences are bounded.

**Risk: Porting the full native Hessian routine in one pass produces a large, fragile Kokkos function.**  
Mitigation: Port in sections that mirror the native pipeline and add lower-level parity tests around the field layer, H1 linear-up derivative propagation, Hessian output, and pair-force derivative output before relying on ASE-level tests alone.

**Risk: `MACEKokkosFloat` exposes a response API that has not been validated for field-aware models.**  
Mitigation: Keep field-aware ASE construction restricted to `dtype="float64"` initially, while still arranging the template binding so the class surface is consistent. Add float32 support only after explicit numeric validation.

**Risk: ASE tests become dependent on optional local MACEField/PyTorch assets.**  
Mitigation: Keep tests that require `/home/bonan/appdir/mace-field/MACEField-MH-0-omat-dielectric.model` skippable, but make native serial versus Kokkos JSON comparison the main regression path once the JSON fixture can be generated.

**Risk: LAMMPS users expect BECs and polarizability through the pair style.**  
Mitigation: Document that LAMMPS pair style support remains energy/force focused and that tensor responses need a separate output interface. Do not overload pair energy/force APIs with global response tensors.

## Alternative Approaches

**ASE-only finite-difference responses:** Leave Kokkos evaluator without analytic Hessian and force-field derivative methods, then compute BECs and polarizability in ASE by finite-differencing Kokkos energy/forces over electric field components. This is much easier to implement but would multiply calculation cost, introduce step-size sensitivity, and fail to match the native analytic API.

**Separate `MACEFieldKokkos` class:** Add a new evaluator class dedicated to field-aware models. This would isolate field-specific code, but it would duplicate MACEKokkos model loading and make ASE/LAMMPS backend selection more complicated.

**Energy/force-only Kokkos field support:** Stop after enabling ASE `use_kokkos=True` for field-aware energy, forces, and polarization. This is a useful partial milestone, but it does not satisfy the request to support Born effective charges and polarizability through the existing ASE response path.

## Assumptions

- Field-aware Kokkos support should initially target `dtype="float64"` only.
- The ASE calculator is the primary user-facing harness for BECs and polarizability.
- LAMMPS tensor response output is out of scope for this plan unless explicitly requested later.
- Existing native serial MACEField behavior is the source of truth for Kokkos parity.
