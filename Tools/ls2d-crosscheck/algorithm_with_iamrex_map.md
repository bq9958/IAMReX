# Algorithm correspondence table (2D, single grid)

Sources: **S99** = Sussman et al., JCP 148 (1999); **A98** = Almgren et al., JCP 142 (1998).
Third column = where the same step lives in IAMReX (AMReX-based reference implementation).
Equation numbers refer to the printed papers; only the *form* of each step is restated here.

Staggering (same as S99 §3 and IAMReX): `u, v, φ, ρ, μ` cell-centred; `u^ADV` face-centred (MAC); `p` node-centred.

## A. State and material properties
| Step | Form | Source | ls2d function | IAMReX |
|---|---|---|---|---|
| Smoothed Heaviside | H_ε(φ): 0 / ½[1+φ/ε+sin(πφ/ε)/π] / 1, ε = α·Δx | S99 (50) | `heaviside(phi, eps)` | `NS_LS.cpp::phi_to_heavi` (α = `ns.epsilon`, default 2) |
| Density, viscosity | ρ = ρ₂ + (ρ₁−ρ₂)H_ε, same for μ | S99 (5)-(6) | `material_from_phi` | `NS_LS.cpp::heavi_to_rhoormu` |
| Mid-time φ | φ^{n+½} = ½(φⁿ+φ^{n+1}) | S99 (13) | `advance()` step 3 | `NavierStokesBase::get_phi_half_time` |
| ρ^{n+½}, μ^{n+½} | from φ^{n+½} | S99 (14)-(15) | same | `NavierStokes.cpp` ~L2260-2275 (`rho_half`, `viscn_cc`, `viscnp1_cc`) |

## B. One time step (S99 §3.1)
| Step | Form | Source | ls2d function | IAMReX |
|---|---|---|---|---|
| 1a. Predictor to faces | Taylor extrapolation of U, φ to faces at t^{n+½}, PLM limited normal slopes, transverse terms upwinded | S99 (20)-(24); A98 §3.2 | `godunov::predict_faces` | AMReX-Hydro `Godunov::ExtrapVelToFaces`, `ComputeEdgeState` (PLM) |
| 1b. Upwind face velocity | choose L/R state by sign rule | S99 (25) | `godunov::upwind_face` | AMReX-Hydro (same) |
| 1c. MAC projection | solve D^MAC(1/ρⁿ G^MAC p^MAC) = D^MAC U^{n+½}; U^ADV = U^{n+½} − (1/ρⁿ)G^MAC p^MAC | S99 (27)-(28) | `mac::project` | `NavierStokesBase::mac_project` → `MacProjector` |
| 2. Level-set advection | φ^{n+1} = φⁿ − Δt [U^ADV·∇φ]^{n+½} (non-conservative form) | S99 (12), (29)-(30) | `advect_scalar_nonconservative` | `scalar_advection` + `scalar_update(phicomp)` with `advectionType=NonConservative` |
| 3. Velocity advection term | [(U·∇)U]^{n+½} from face states and U^ADV | S99 (31)-(32) | `advect_velocity_convective` | `velocity_advection` (`do_mom_diff=0`) |
| 4. Semi-implicit viscous solve | (U*−Uⁿ)/Δt = −[(U·∇)U] − Gp^{n−½}/ρ^{n+½} + (L*+Lⁿ)/(2ρ^{n+½}) + F | S99 (16); L in §3.3 | `viscous::crank_nicolson_solve` | `velocity_update` → `Diffusion::diffuse_velocity` (MLABecLaplacian) |
| 5. Approximate nodal projection | V = U*/Δt + Gp^{n−½}/ρ^{n+½}; L_ρ p^{n+½} = D V; U^{n+1} = Δt(V − Gp^{n+½}/ρ^{n+½}) | S99 (17), §3.4; A98 §3.3 | `projection::nodal_project` | `level_projector` → `Projection::level_project` (MLNodeLaplacian) |
| 6. Redistance | see C | S99 §3.5 | `levelset::redistance` | `NavierStokesBase::reinit` |
| 7. Δt | min of CFL, gravity, viscous (and capillary in v2) constraints | S99 §3.1.1 | `compute_dt` | `NavierStokesBase::estTimeStep` (+ `ns.fixed_dt`) |

Order note: IAMReX runs step 6 (reinit) *immediately after* step 2 and before the viscous/projection steps, so ρ^{n+1}, ρ^{n+½} are built from the *reinitialised* φ^{n+1}. S99 lists redistance as the last step. ls2d keeps the IAMReX order (a switch `reinit_after_projection` reproduces the S99 order for comparison).

## C. Redistance (S99 §3.5)
| Step | Form | Source | ls2d function | IAMReX |
|---|---|---|---|---|
| Init | d⁰ = φ^{n+1} | S99 step 1 | `redistance()` | `reinit`: copy `phi_ctime → phi_original` |
| Sign function | S = 2(H_ε(φ⁰) − ½), frozen | S99 (57) | `sign_smoothed(phi0)` | `phi_to_sgn0` (only for upwind choice; see deviations D2) |
| Pseudo-time step | Δτ = Δx/2; iterate until τ = ε → N = ε/Δτ = 2α | S99 after (55) | `params.n_reinit` | `dtlevel = 0.5*dxmin`, `ns.number_of_reinit` (default 4) |
| One-sided ENO2 derivatives | D^L = D⁻d + (Δx/2) m(D⁺D⁻d_i, D⁺D⁻d_{i−1}); D^R = D⁺d − (Δx/2) m(D⁺D⁻d_i, D⁺D⁻d_{i+1}); m = minmod | S99 (58)-(59), (63)-(65) | `eno2_onesided` | `rk_*_reinit` (`phi1_face`, `phi2`, `dxm/dxp`) |
| Godunov upwind choice | w^L = S·D^L, w^R = S·D^R; take D^L if w^L>0 & w^L+w^R>0; D^R if w^R<0 & w^L+w^R<0; else 0 | S99 (60)-(62) | `godunov_gradient` | `rk_*_reinit` (`ddx` branch; else-branch averages → D1) |
| Spatial operator | L(d) = S(1 − |∇d|) | S99 (56) | `redistance_rhs` | `G0 = phi3 − 1`, multiplied by sign |
| RK2 (Heun) | d^{(1)} = d^k + Δτ L(d^k); d^{k+1} = d^k + ½Δτ [L(d^k) + L(d^{(1)})] | S99 (54)-(55) (printed form differs, see D3) | `redistance_rk2` | `rk_first_reinit`, `rk_second_reinit` |
| Volume constraint | λ = −∫H'_ε(d⁰)·(d̃^k−d⁰)/(τ^k−τ⁰) dx / ∫H'_ε(d⁰)² dx ; d^k = d̃^k + λ(τ^k−τ⁰)H'_ε(d⁰), per cell, 9-pt stencil | S99 (66)-(71); Sussman–Fatemi 1999 | `volume_fix` | `mass_fix` (see D4, D5) |
| H'_ε | 0 outside |d|≤ε; ½[1/ε + cos(πd/ε)/ε] inside | S99 (68) | `delta_eps` | `mass_fix` (`deltafunc`) |

## D. Diagnostics (S99 §5.1)
| Quantity | Form | Source | ls2d |
|---|---|---|---|
| Interface error between resolutions | E = Σ∫|H(φ_c) − H(φ_f)| dx, midpoint rule on subcells, bilinear interp | S99 (80) | `diag::interface_error` |
| Phase volume | V = Σ∫H(φ) dx | S99 (81) | `diag::volume` |
| Distance-function quality | max / mean of ||∇φ|−1| in band |φ|<ε | (ours) | `diag::grad_norm_error` |

## E. Boundary conditions (IAMReX `ns.lo_bc/hi_bc` codes kept)
0 periodic · 1 inflow · 2 outflow · 3 symmetry · 4 slip wall · 5 no-slip wall.
φ: zero-gradient at all non-periodic walls (IAMReX `fill_allgts` with FOEXTRAP). p (nodal): Neumann at walls, Dirichlet 0 at outflow.
