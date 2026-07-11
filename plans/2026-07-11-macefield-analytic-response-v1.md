# MACEField Analytic Response Derivatives

## Objective

Implement native serial Symmetrix support for MACEField Born effective charges and polarizability through analytical, hand-written directional second derivatives of the electric-field enthalpy. The ASE interface should keep the same public behavior as upstream MACEField: converted Symmetrix JSON models with field coupling expose polarization, becs, and polarizability; direct PyTorch MACEField checkpoints remain rejected; normal MACE behavior remains unchanged.

This plan refines the pending analytic-response part of plans/2026-07-11-macefield-second-derivatives-v1.md. It uses the current finite-difference ASE path in symmetrix/source/symmetrix/symmetrix_calc.py:187 as the numerical baseline, and the upstream PyTorch implementation in /home/bonan/appdir/mace-field/mace/modules/extensions.py:511 and /home/bonan/appdir/mace-field/mace/modules/utils.py:551 as the oracle.

## Current Status

Analytic graph-field polarizability and Born effective charges are now implemented for the native serial MACEField JSON path. The implementation adds directional gradient products for MultivariatePolynomial and MultilayerPerceptron, a native field-seeded electric_field_adj Hessian driver, a native field derivative of pair forces, pybind exposure, and ASE calculator wiring for polarizability and becs. The result is checked against central finite differences of native electric_field_adj, central finite differences of native forces with respect to field, and the existing upstream PyTorch MACEField response-property oracle.

The BEC implementation uses the mixed-derivative identity Z_i,a,b = d Force_i,b / d F_a. This avoids the position-seeded geometry Hessian route while producing the same upstream tensor convention.

## Mathematical Formula

For graph g, let R be Cartesian positions, F_g be the graph electric field, Omega_g be the cell volume, and E_int,g(R,F) be the MACEField interaction energy. Upstream MACEField differentiates E_int, not E0 + E_int, for forces and field responses. This is visible at /home/bonan/appdir/mace-field/mace/modules/extensions.py:480, /home/bonan/appdir/mace-field/mace/modules/extensions.py:485, and /home/bonan/appdir/mace-field/mace/modules/extensions.py:513.

The raw polarization conjugate is p_g,a = - dE_int,g / dF_g,a. The reported ASE polarization is P_g,a = p_g,a / Omega_g. This sign and scaling follow /home/bonan/appdir/mace-field/mace/modules/utils.py:570 and /home/bonan/appdir/mace-field/mace/modules/extensions.py:553. Native Symmetrix already exposes this through electric_field_adj in symmetrix/source/symmetrix/symmetrix_calc.py:187.

The Born effective charge tensor should be Z_i,a,b = d p_g,a / d R_i,b = - d2 E_int,g / dF_g,a dR_i,b. Equivalently, because Force_i,b = - dE_int,g / dR_i,b, Z_i,a,b = d Force_i,b / dF_g,a. The upstream tensor shape is atoms by polarization component by displacement component, then ASE flattens the last two axes; this is defined by /home/bonan/appdir/mace-field/mace/modules/utils.py:574 and currently mirrored by symmetrix/source/symmetrix/symmetrix_calc.py:220.

The polarizability tensor should be alpha_g,a,b = d p_g,a / dF_g,b / (Omega_g * epsilon0) = - d2 E_int,g / dF_g,a dF_g,b / (Omega_g * epsilon0). Upstream computes the derivative of raw polarization first, then divides by volume and epsilon0 at /home/bonan/appdir/mace-field/mace/modules/extensions.py:535. The epsilon0 constant is 8.8541878128e-12 / 1.602176634e-19 / 1e10 in e/V/Angstrom.

The native implementation should compute these quantities as forward-over-reverse directional derivatives. For a seed s, first run the normal primal forward pass and normal reverse pass to obtain electric_field_adj = dE_int / dF. Then propagate dot variables through the primal pass and dot-adjoint variables through the reverse pass. The result is dot p = - dot electric_field_adj. Field seeds produce alpha columns after division by Omega_g * epsilon0, and the same field seeds produce BEC rows through dot Force.

The isolated field-coupling block has a useful closed form. In native scalar-vector layout, S_u is the scalar H1 pre-field channel, V_u,a is the vector H1 pre-field channel, and F_a is the electric field. With field-feats weights A_uw and B_uw and field-linear weights Ls_uw and Lv_uw, the block computes delta_vector_w,a += A_uw S_u F_a, delta_scalar_w += B_uw (-sum_a V_u,a F_a), linear_scalar_w += Ls_uw delta_scalar_u, linear_vector_w,a += Lv_uw delta_vector_u,a, then S'_w = S_w - linear_scalar_w and V'_w,a = V_w,a + linear_vector_w,a. This is implemented at libsymmetrix/source/mace.cpp:183 through libsymmetrix/source/mace.cpp:233, and its reverse is implemented at libsymmetrix/source/mace.cpp:238 through libsymmetrix/source/mace.cpp:321. The tangent and tangent-adjoint rules should be obtained by differentiating these assignments directly.

For position-seeded BECs, the geometry part would require true second derivatives. The common force factor R(r) Y(x) contributes d_b[(x_a/r) R'(r) Y + R(r) Y_a] = (delta_ab/r - x_a x_b/r^3) R'(r) Y + (x_a/r)(x_b/r) R''(r) Y + (x_a/r) R'(r) Y_b + (x_b/r) R'(r) Y_a + R(r) Y_ab. The implemented force-field route avoids those geometry Hessians for BECs, but the formula remains useful if future position-seeded Hessians are needed.

## Implementation Plan

- [ ] 1. Add formula-locking tests that compare native finite-difference identities against the upstream PyTorch MACEField oracle before changing analytic kernels.
  The tests should assert the sign, volume scaling, epsilon0 scaling, and tensor orientation above. They should include P = -electric_field_adj / Omega, Z_i,a,b = dForce_i,b / dF_a, and alpha_a,b = d raw_p_a / dF_b / (Omega * epsilon0). This makes the finite-difference harness a contract rather than only a temporary implementation.

- [ ] 2. Introduce a serial response-derivative driver that owns one seeded directional pass at a time.
  Add native entry points that seed either one graph-field component or one atomic Cartesian displacement component, run the existing field-aware primal and reverse pass from libsymmetrix/source/mace.cpp:68, and return the directional derivative of electric_field_adj. Keep this fixed-neighbor and graph-field-only for response properties. The initial implementation should favor clarity over batching; batched seeds can come after correctness.

- [ ] 3. Add tangent buffers and zeroing discipline beside the existing primal and adjoint buffers.
  Extend libsymmetrix/source/mace.hpp around the current field-aware buffers for dot values and dot-adjoints of Y, R0, A0, M0, H1 pre-field, H1 post-field, R1, Phi1, A1, M1, H2, readout inputs, electric_field_adj, and node_forces. The response driver should make buffer lifetime explicit so normal energy-force calls do not pay extra allocation or stale-state costs.

- [ ] 4. Implement the field-coupling block tangent and tangent-adjoint rules as the first analytic kernel.
  Differentiate compute_field_H1 and reverse_field_H1 directly from the scalar-vector formulas in the Mathematical Formula section. Add C++ or pybind-visible test hooks mirroring the existing first-derivative field tests in symmetrix/test/test_macefield_native.py:78 and symmetrix/test/test_macefield_native.py:97. Compare dot H1, dot H1_adj, and dot electric_field_adj against PyTorch autograd and central finite differences for both field and H1 seeds.

- [x] 5. Add Hessian-vector support for the nonlinear algebraic kernels before touching geometry.
  MultilayerPerceptron currently exposes evaluate_gradient only at libsymmetrix/source/multilayer_perceptron.hpp:16, and MultivariatePolynomial exposes evaluate_gradient only at libsymmetrix/source/multivariate_polynomial.hpp:16. Add Hessian-vector or tangent-gradient APIs for the readout MLP and polynomial blocks used by M1, so tangent-adjoint reverse propagation can pass through readouts, M1, and H2 without forming dense full Hessians unless the tensor is already tiny.
  Status: Done for the serial native MultivariatePolynomial and MultilayerPerceptron utilities through evaluate_gradient_directional, with focused finite-difference tests.

- [ ] 6. Add radial, cutoff, and spherical-harmonic second-derivative primitives for fixed-neighbor geometry.
  Extend CubicSpline and CubicSplineSet with value, first derivative, and second derivative evaluation, then store R0'', R1'', and cutoff-scale second derivatives where reverse_A0_scaled, reverse_A1_scaled, reverse_A0, and reverse_Phi1 need them. Replace compute_Y for response mode with a path that also stores Y Hessians from sphericart. Validate these primitives with local finite differences of the existing first-derivative quantities before integrating them into BECs.

- [ ] 7. Propagate forward tangents through the full field-aware MACE stack.
  Implement dot versions of compute_Y, compute_R0, compute_A0, compute_A0_scaled, compute_M0, compute_H1_product, compute_field_H1, compute_H1_linear_up, compute_R1, compute_Phi1, compute_A1, compute_A1_scaled, compute_M1, compute_H2, and compute_readouts. Field seeds should only enter through dot electric field; position seeds should enter through edge-vector and distance tangents while the neighbor topology stays fixed.

- [ ] 8. Propagate tangent-adjoints through the reverse pass and read out dot electric_field_adj.
  Differentiate the reverse sequence at libsymmetrix/source/mace.cpp:107 in the same order: readout seeding, reverse_H2, reverse_M1, reverse_A1_scaled, reverse_A1, reverse_Phi1, reverse_H1_linear_up, reverse_field_H1, reverse_H1_product, reverse_M0, reverse_A0_scaled, and reverse_A0. Each layer should get a focused test that compares its local dot-adjoint output to finite differences of the existing reverse kernel before being trusted in the full driver.
  Status: Done for graph-field seeds through the native response driver. The driver carries tangent-adjoints through the force path and is validated against native finite differences of electric_field_adj and forces.

- [x] 9. Assemble analytic polarizability and BEC tensors behind the native JSON ASE path.
  For polarizability, run three graph-field seeds and set alpha[:,b] from -dot electric_field_adj divided by volume and epsilon0. For BECs, run three graph-field seeds and set Z[i,a,b] from dot Force[i,b] with no volume division. Preserve ASE flattening and the current property gates in symmetrix/source/symmetrix/symmetrix_calc.py, and keep finite differences as tests rather than production fallback.
  Status: Done. Polarizability uses the native field-seeded electric_field_hessian path in ASE, and BECs use native field derivatives of forces aggregated to atoms.

- [ ] 10. Replace the production finite-difference response path only after analytic outputs match both oracle families.
  The final switch should happen after analytic polarization, BECs, and polarizability match upstream PyTorch MACEField on the AlN fixture and match the native finite-difference harness within agreed tolerances. Keep direct .model MACEField rejection, graph-level response-field requirements, and plain MACE property behavior unchanged.

## Verification Criteria

- Native analytic polarization remains bitwise or near-bitwise consistent with the current first-derivative path from electric_field_adj.
- Native analytic BECs match upstream ASE MACEField tensor orientation and values on the example MACEField model within the tolerance already justified by the finite-difference baseline.
- Native analytic BECs satisfy Z_i,a,b = dForce_i,b / dF_a under central finite differences of native forces for stable fixed-neighbor structures.
- Native analytic polarizability matches upstream ASE MACEField after volume and epsilon0 scaling, and satisfies alpha_a,b = d raw_p_a / dF_b / (Omega * epsilon0) under central finite differences of native raw polarization.
- The field-coupling block tests compare first and second directional derivatives against PyTorch autograd for both field-seeded and H1-seeded perturbations.
- Geometry primitive tests compare R'', cutoff-scale'', and Y Hessian contractions against finite differences of R', scale', and Y gradients.
- Existing normal MACE tests and MACEField energy-force tests continue to pass, including the direct PyTorch .model rejection behavior.
- The focused CPU MACEField suite passes after rebuilding the native extension.

## Potential Risks and Mitigations

1. The hand-written tangent-adjoint pass can diverge from the existing reverse pass.
   - Impact: BECs and polarizability may be subtly wrong even while energy and forces remain correct.
   - Likelihood: High.
   - Mitigation: Add layer-level finite-difference checks for every tangent-adjoint kernel, starting with the isolated field block before integrating the full network.
   - Contingency: Keep finite-difference response properties as a private debug oracle and avoid switching production behavior until the analytic path is covered.

2. BEC geometry derivatives are substantially harder than polarizability derivatives.
   - Impact: Polarizability may be ready earlier than BECs, but the API currently exposes both response properties.
   - Likelihood: High.
   - Mitigation: Implement the shared forward-over-reverse framework and field-seeded polarizability first, then add position-seeded BECs after spline and spherical-harmonic Hessian tests pass.
   - Contingency: If needed, gate analytic BECs separately while leaving finite-difference BECs only in explicit debug tests, not hidden production fallback.

3. Dense Hessian construction in MLP or polynomial blocks can be slow or memory-heavy.
   - Impact: Response calls may be too expensive for realistic structures.
   - Likelihood: Medium.
   - Mitigation: Implement Hessian-vector or tangent-gradient products, since the response driver needs seeded directional derivatives rather than full Hessians.
   - Contingency: Batch seed directions only after correctness, and profile the slowest kernels before adding broader optimizations.

4. Neighbor-list changes can make finite-difference comparisons noisy.
   - Impact: Correct analytic fixed-neighbor derivatives can appear to disagree with ASE-level finite differences.
   - Likelihood: Medium.
   - Mitigation: Use structures with pair distances away from cutoffs, keep native fixed-neighbor tests for local identities, and separate tight PyTorch oracle comparisons from looser finite-difference sanity checks.
   - Contingency: Diagnose failures with lower-level fixed-neighbor tests before relaxing tolerances.

5. Shape and scaling mistakes can hide behind flattened ASE arrays.
   - Impact: The values may be transposed or scaled by volume while still having the expected shape.
   - Likelihood: Medium.
   - Mitigation: Keep raw polarization, volume-scaled polarization, BECs, and normalized polarizability as separately tested intermediate values. Test asymmetric fields and displaced structures so transposes are visible.
   - Contingency: Add temporary debug-only unflattened outputs in tests if a mismatch is hard to localize.

## Alternative Approaches

1. Keep finite differences as production behavior.
   - Description: Continue using central differences of native raw polarization for BECs and polarizability.
   - Pros: Already implemented and already compared to upstream PyTorch.
   - Cons: Expensive, step-size sensitive, and not the analytical support requested.
   - Recommendation: Keep only as a validation and debug baseline.

2. Implement polarizability analytically first and leave BECs finite-differenced.
   - Description: Field-seeded derivatives avoid geometry Hessians, so this is a smaller first analytical milestone.
   - Pros: Reduces risk and proves the forward-over-reverse framework.
   - Cons: Mixed analytic and finite-difference behavior can be confusing if exposed silently.
   - Recommendation: Accept as an internal milestone, but expose final production behavior only when policy is explicit.

3. Add a C++ automatic differentiation scalar type.
   - Description: Rework kernels around nested AD values for second derivatives.
   - Pros: Less manual algebra in principle.
   - Cons: High rewrite risk for the existing hand-optimized serial kernels and likely poor performance without substantial redesign.
   - Recommendation: Do not choose for this phase.

4. Delegate response properties back to upstream PyTorch MACEField.
   - Description: Call the PyTorch calculator for BECs and polarizability while using native Symmetrix for energy and forces.
   - Pros: Fastest way to match upstream numerically.
   - Cons: Reintroduces the fallback behavior that was deliberately removed and masks native gaps.
   - Recommendation: Do not use.

## Assumptions

- The target remains the ASE interface and native serial evaluator; LAMMPS and Kokkos response derivatives are out of scope.
- Response properties remain defined only for graph-level electric fields with shape 3.
- The converted Symmetrix JSON field model is the only supported MACEField input to Symmetrix.
- The upstream MACEField PyTorch implementation remains available in the local virtual environment for CPU oracle tests.
- Fixed-neighbor analytical derivatives are acceptable; finite-difference tests must avoid cutoff discontinuities.
