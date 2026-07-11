# MACEField Second Derivatives

## Objective

Implement native serial Symmetrix support for MACEField response properties through the ASE calculator: polarization, Born effective charges, and polarizability for converted Symmetrix JSON models with field coupling. The implementation should match the upstream ASE MACEField calculator semantics, keep original PyTorch MACEField checkpoints out of the Symmetrix calculator path, and preserve normal MACE behavior.

This plan assumes the existing native JSON field path remains the base: libsymmetrix/source/mace.cpp:68 already evaluates field-aware energies and forces, symmetrix/source/symmetrix/symmetrix_calc.py:147 routes native field-coupled JSON through that path, and symmetrix/test/test_symmetrix_calc.py:162 keeps the ASE energy and force oracle for converted JSON models.

## Current Status

As of the current worktree state, the ASE-facing baseline is implemented for converted MACEField JSON. Direct PyTorch MACEField checkpoints are rejected, graph-level field semantics are enforced for response properties, native polarization is exposed from the serial evaluator field adjoint, and BECs plus polarizability are available as central finite differences of native raw polarization. This finite-difference implementation is intentionally a baseline and oracle harness; it is not the final analytic fixed-graph second-derivative implementation described in Tasks 4 through 7.

The remaining core work is to replace the finite-difference BEC and polarizability path with native directional second derivatives through the serial C++ kernels. Kokkos-requested field JSON is currently routed back to the serial native field path so field coupling is not silently ignored.

## Implementation Plan

- [x] 1. Freeze the upstream response-property contract before changing native code.
  Capture focused tests that call the upstream MACEField ASE calculator for polarization, becs, and polarizability using the example model and the same AlN fixture already used by symmetrix/test/test_macefield_native.py:122. The oracle must document that upstream MACEField computes response derivatives from inter_e at /home/bonan/appdir/mace-field/mace/modules/extensions.py:511, defines polarization as the negative field derivative at /home/bonan/appdir/mace-field/mace/modules/utils.py:551, computes BECs as the position derivative of raw polarization at /home/bonan/appdir/mace-field/mace/modules/utils.py:574, and computes polarizability as the field derivative of raw polarization at /home/bonan/appdir/mace-field/mace/modules/utils.py:599. The tests should also lock the ASE-facing shapes from /home/bonan/appdir/mace-field/mace/calculators/mace.py:414 and the final flattening behavior from /home/bonan/appdir/mace-field/mace/calculators/mace.py:613.
  Status: Done for the current ASE baseline through upstream oracle tests in symmetrix/test/test_symmetrix_calc.py.

- [x] 2. Separate graph-level electric-field semantics from per-atom implementation convenience.
  The current ASE adapter tiles a length-3 field to per-atom shape in symmetrix/source/symmetrix/symmetrix_calc.py:130, while the native field kernel accepts either a graph field or per-node field in libsymmetrix/source/mace.cpp:157. Response derivatives should use a graph-level field so electric_field_adj directly represents dE/dfield for the graph instead of a per-node tensor that later needs summation. Update the native ASE path to preserve the user-facing length-3 field when possible, still accept explicit per-atom fields for the energy-force path, and define response properties only for graph-level fields unless a clear per-atom response convention is added.
  Status: Done for ASE response properties. Per-atom fields remain accepted for energy and forces, while response requests require graph-level shape 3.

- [x] 3. Expose native first-derivative polarization as the first incremental deliverable.
  The native reverse field block already accumulates electric_field_adj in libsymmetrix/source/mace.hpp:135 and libsymmetrix/source/mace.cpp:253, with the field adjoint contributions computed in libsymmetrix/source/mace.cpp:303. Add native evaluator storage for raw polarization and volume-scaled polarization, then expose the ASE property only when has_field_coupling is true. The sign must follow upstream get_polarization at /home/bonan/appdir/mace-field/mace/modules/utils.py:570, and the volume scaling must follow /home/bonan/appdir/mace-field/mace/modules/extensions.py:553. Verify this step against both the upstream ASE oracle and a central finite-difference energy derivative before using it as the base for second derivatives.
  Status: Done for the ASE calculator. Polarization is computed from the native serial field adjoint and volume-scaled.

- [ ] 4. Add a serial directional-derivative framework for forward-over-reverse response derivatives.
  Introduce tangent buffers beside the existing primal and adjoint buffers in libsymmetrix/source/mace.hpp:115, covering the tensors that already participate in the field-aware forward and reverse pass. The driver should seed one direction at a time in either Cartesian positions or graph electric field, run the normal primal calculation, run the normal reverse calculation, and propagate tangent-adjoints through the reverse pass to obtain directional changes in electric_field_adj. This keeps the response implementation local to the serial evaluator and avoids adding a general automatic differentiation dependency.
  Status: Pending. This is the main next phase needed to replace the finite-difference baseline.

- [ ] 5. Implement and test the field-coupling block's second-derivative rules first.
  The field block is isolated enough to be a clean proving ground: compute_field_H1 saves H1_pre_field at libsymmetrix/source/mace.cpp:169, applies the scalar-vector and vector-scalar field paths at libsymmetrix/source/mace.cpp:183, and reverse_field_H1 propagates adjoints through the same paths at libsymmetrix/source/mace.cpp:238. Add tangent propagation for compute_field_H1 and tangent-adjoint propagation for reverse_field_H1, then compare H1, H1_adj, electric_field_adj, and their directional derivatives against PyTorch autograd using the existing standalone field-transform fixtures in symmetrix/test/test_macefield_native.py:78 and symmetrix/test/test_macefield_native.py:97.
  Status: Pending. Existing first-derivative field-block tests remain in place, but second directional derivatives are not implemented yet.

- [ ] 6. Extend geometric and interaction kernels with the second-order ingredients needed for BECs.
  Position-seeded response derivatives require second derivatives wherever the current force path differentiates geometry-dependent quantities. Add spline second derivatives for radial functions beside compute_R0 and compute_R1, use sphericart Hessian support instead of finite-differencing spherical harmonics because libsymmetrix/external/sphericart/sphericart/include/sphericart.hpp:147 provides compute_with_hessians, and propagate directional derivatives through A0, A0 scaling, M0, H1 product, Phi1, A1, A1 scaling, M1, H2, and readouts in the same order as the existing field-aware reverse pass at libsymmetrix/source/mace.cpp:107. This is the largest task and should be split internally by layer with local finite-difference checks for each kernel.
  Status: Pending. The current BEC implementation uses finite differences of native raw polarization and does not yet provide fixed-graph analytic derivatives.

- [ ] 7. Assemble native BEC and polarizability drivers from the directional framework.
  For BECs, seed each atomic Cartesian displacement direction and record the directional derivative of raw polarization, returning the upstream shape before ASE flattening. For polarizability, seed each graph electric-field direction and record the directional derivative of raw polarization, then apply the same volume and eps0 normalization as /home/bonan/appdir/mace-field/mace/modules/extensions.py:535. Store the raw intermediate tensors long enough to debug sign and scaling, but expose only the upstream ASE properties by default.
  Status: Partially covered by the finite-difference ASE baseline. The native analytic drivers remain pending and should be implemented after Tasks 4 through 6.

- [x] 8. Bind the new native response API through pybind and the ASE calculator.
  Add pybind exposure next to the existing field methods in symmetrix/source/cpp/mace.cpp:57 and symmetrix/source/cpp/mace.cpp:178. Update Symmetrix.implemented_properties in symmetrix/source/symmetrix/symmetrix_calc.py:37 for native field-coupled JSON models, populate results for polarization, becs, and polarizability in calculate, and extend check_state invalidation at symmetrix/source/symmetrix/symmetrix_calc.py:101 so field changes invalidate cached response results. Plain MACE JSON and PyTorch-converted normal MACE checkpoints must keep their current property set and behavior.
  Status: Done at the ASE layer using the existing pybind-exposed native field adjoint. No new C++ response-property pybind API has been added yet.

- [x] 9. Add end-to-end native ASE tests and numerical sanity checks.
  Extend symmetrix/test/test_symmetrix_calc.py:150 and symmetrix/test/test_macefield_native.py:122 with focused tests that compare native JSON polarization, BECs, and polarizability against the upstream ASE MACEField oracle on CPU. Add central finite-difference checks that relate polarization to field energy derivatives, BECs to field derivatives of forces, and polarizability to field derivatives of polarization. Keep tolerances separate for direct oracle comparisons and finite-difference checks because finite-difference step size will amplify spline and neighbor-list sensitivity.
  Status: Done for upstream ASE oracle comparison and response API guards. Additional derivative-identity tests should be added when the analytic directional framework lands.

- [x] 10. Document scope limits and keep unsupported paths explicit.
  Document that this phase targets the serial native ASE path only, matching the user's current scope. Kokkos and LAMMPS should raise clear not-implemented errors for response properties until their own derivative paths exist, while energy and forces continue to work where currently supported. The extractor support in symmetrix/source/symmetrix/extract_mace_data.py:70 should remain unchanged except for metadata needed by response tests, and direct PyTorch MACEField checkpoints should continue to raise the conversion error covered by symmetrix/test/test_symmetrix_calc.py:150.
  Status: Done with one implementation adjustment: Kokkos-requested MACEField JSON is forced onto the serial native field path instead of raising or silently ignoring field coupling. Direct PyTorch MACEField checkpoints still raise the explicit conversion error.

## Verification Criteria

- Native Symmetrix JSON MACEField polarization matches upstream ASE MACEField for the AlN fixture at zero and finite graph electric fields within a tolerance justified by the existing energy-force agreement in symmetrix/test/test_macefield_native.py:168.
- Native BECs match upstream ASE MACEField in the ASE-flattened shape used by /home/bonan/appdir/mace-field/mace/calculators/mace.py:613 and also satisfy the finite-difference relation to force changes under small graph-field perturbations.
- Native polarizability matches upstream ASE MACEField after volume and eps0 scaling from /home/bonan/appdir/mace-field/mace/modules/extensions.py:541 and satisfies the finite-difference relation to polarization changes under small graph-field perturbations.
- The standalone native field-transform tests continue to match PyTorch autograd for first and second directional derivatives of the field block.
- Existing normal MACE calculator tests still pass, including on-the-fly normal PyTorch conversion, because the new response API is gated on has_field_coupling.
- Direct MACEField PyTorch checkpoint use through Symmetrix continues to raise the explicit JSON-conversion error, so native implementation gaps cannot be hidden by PyTorch fallback.
- The focused CPU command for MACEField tests passes after rebuilding the extension, including test_extract_mace_data.py, test_macefield_field_transform.py, test_macefield_native.py, and the MACEField subset of test_symmetrix_calc.py.

Current verification status: the focused CPU suite passes with 16 tests selected for MACEField/native/fallback behavior. The analytic second-derivative criteria for the field block and fixed-graph BECs remain pending because the current BEC and polarizability path is a finite-difference baseline.

## Potential Risks and Mitigations

1. **Second-order native derivatives are much larger than the current hand-written reverse pass.**
   - Impact: The implementation can become brittle if tangent and tangent-adjoint buffers are added without tight layer-level tests.
   - Likelihood: High.
   - Mitigation: Implement the field block first, then one layer at a time, with PyTorch or finite-difference checks at each layer before integrating into the full response driver.
   - Contingency: Ship polarization first and keep BECs and polarizability behind explicit not-implemented errors until the layer-level second-order checks are in place.

2. **Field shape ambiguity can produce incorrect polarization by summing or not summing per-atom adjoints.**
   - Impact: Energy and forces may still look correct while response properties have the wrong magnitude.
   - Likelihood: Medium.
   - Mitigation: Use graph-level electric fields for response properties, test both length-3 and per-atom energy-force inputs, and reject per-atom response requests until a formal convention is defined.
   - Contingency: Expose only graph-field response properties in ASE and document per-atom fields as energy-force-only for now.

3. **Volume, eps0, and unit conversions may differ between native Symmetrix and upstream MACEField.**
   - Impact: Tensor shapes can match while physical values differ by constant factors.
   - Likelihood: Medium.
   - Mitigation: Freeze upstream ASE outputs and finite-difference identities in tests before implementation, and keep raw polarization, volume-scaled polarization, and polarizability normalization separate internally.
   - Contingency: Add diagnostic properties only in tests so scaling mismatches can be localized without changing the public ASE API.

4. **Neighbor-list discontinuities can pollute finite-difference response checks.**
   - Impact: Numerical sanity tests can fail even when analytic derivatives are correct.
   - Likelihood: Medium.
   - Mitigation: Use small perturbations on stable structures with pair distances away from the cutoff, and keep finite-difference tests looser than direct ASE oracle tests.
   - Contingency: Add a lower-level fixed-neighbor native test for derivative identities when ASE-level finite differences are noisy.

5. **Kokkos and serial property sets can diverge in confusing ways.**
   - Impact: Users may request polarization from a Kokkos-backed calculator and get an obscure missing-method failure.
   - Likelihood: Medium.
   - Mitigation: Gate response properties on the serial evaluator until Kokkos support is implemented, and raise a direct not-implemented error for Kokkos response requests.
   - Contingency: Leave Kokkos energy and force behavior unchanged and add a separate future plan for Kokkos response support.

## Alternative Approaches

1. **Finite-difference production response properties.**
   - Description: Compute polarization from the native field adjoint, then compute BECs from finite differences of forces with respect to field and polarizability from finite differences of polarization with respect to field.
   - Pros: Much smaller implementation and useful as an oracle harness.
   - Cons: More model evaluations, step-size sensitivity, and not true native second-derivative support.
   - Recommendation: Use this as a validation scaffold and possible explicit debug fallback, not as the default final implementation.

2. **Embed an automatic differentiation library in the C++ serial evaluator.**
   - Description: Replace or wrap scalar calculations with an AD scalar type capable of nested derivatives.
   - Pros: Reduces manual second-derivative algebra and could help future Hessian-like features.
   - Cons: Large dependency and rewrite risk for BLAS-heavy kernels, pybind surfaces, and performance.
   - Recommendation: Do not use for this phase unless manual directional derivatives prove unmaintainable.

3. **Delegate response properties to upstream PyTorch MACEField for ASE only.**
   - Description: Keep native energy and forces but call the upstream PyTorch calculator for polarization, BECs, and polarizability.
   - Pros: Fastest route to matching upstream outputs.
   - Cons: Reintroduces the fallback behavior the user asked to remove and masks native gaps.
   - Recommendation: Do not use.

4. **Implement only raw polarization now and defer BECs and polarizability.**
   - Description: Expose the existing native electric_field_adj-derived polarization first, then plan second derivatives separately.
   - Pros: Low risk and immediately useful.
   - Cons: Does not satisfy full MACEField response support.
   - Recommendation: Accept only as an incremental milestone inside this plan, not as the endpoint.

## Assumptions

- The target for this phase is the ASE interface and serial native evaluator; LAMMPS and Kokkos response properties are out of scope.
- The converted Symmetrix JSON field model remains the only supported MACEField input to Symmetrix.
- The example model at /home/bonan/appdir/mace-field/MACEField-MH-0-omat-dielectric.model remains available for CPU oracle tests.
- The initial supported field convention for response properties is a graph-level electric field with shape 3.
- Central finite differences are acceptable for tests and debugging, but the desired production path is native directional differentiation.

## Dependencies

- CPU torch and mace-field remain installed in the virtual environment so upstream ASE MACEField can be used as the oracle.
- The native extension can be rebuilt locally after modifying libsymmetrix/source/mace.cpp and symmetrix/source/cpp/mace.cpp.
- sphericart Hessian support is available through the vendored external library referenced at libsymmetrix/external/sphericart/sphericart/include/sphericart.hpp:147.
- Existing native MACEField JSON extraction remains available through symmetrix/source/symmetrix/extract_mace_data.py:19.

## Notes

- The current native field-aware energy and force work is a solid base, but response properties should not rely on the previous PyTorch fallback.
- The hardest part is not the field coupling alone; it is propagating directional derivatives through the full geometry-dependent reverse force path.
- The force-field identity for BECs is a particularly valuable independent check because it validates the mixed derivative without relying solely on upstream autograd values.
