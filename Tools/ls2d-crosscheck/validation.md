# Validation summary (2D) — final numbers, one table per class

Everything below is produced by the *current* code (Godunov-PLM advection, reinit every step with the Sussman–Fatemi
constraint, MAC + Crank–Nicolson + Q1 nodal projection with geometric multigrid, surface tension v2).
Older intermediate numbers (M2 ENO2 advection, mis-scaled multigrid, …) are kept in the chronological log at the
bottom of this file but are **superseded**; do not quote them.

Error measure for interface cases: err/L = (1/L)∫|H(φ_exact) − H(φ)| dx, L = initial interface length
(Sussman 1999 eq. 80 = Enright 2002 eq. 14). "E:" = Enright 2002 plain level-set column (WENO5/RK3 advection).

## A. Interface transport (prescribed velocity) — vs Enright 2002
Inputs: `tests/inputs.zalesak`, `tests/inputs.vortex` (T = 8). Paper defaults (`reinit.alpha_delta = 2`).

| ID | case | grid | area change (ls2d) | area change (E) | err/L (ls2d) | err/L (E) | verdict |
|---|---|---|---|---|---|---|---|
| V2 | Zalesak disk, 1 rev | 50²  | −7.6 % | −100 % | 1.60 | 4.03 | ✓ better |
| V2 | | 100² | −1.8 % | +5.3 % | **0.40** | 0.61 | ✓ better |
| V2 | | 200² | −0.04 % | +0.54 % | **0.036** | 0.08 | ✓ better, order 3.5 |
| V3 | single vortex, T = 8 | 64²  | +101 % | −100 % | 0.080 | 0.075 | ~ (both unresolved) |
| V3 | | 128² | +29 % | −39.8 % | **0.0256** | 0.031 | ✓ error better, area sign flipped (see note) |
| V3 | | 256² | +3.1 % | −10.3 % | 0.0079 | 0.008 | ✓ |

Note on the vortex area *gain*: the Sussman–Fatemi constraint conserves the ε-smoothed volume ∫H_ε(φ); on
filaments thinner than 2ε this pushes the zero contour outward. It is a property of the method, not a bug
(static-disk unit test: constraint reduces drift 0.53 % → 0.23 %). Narrowing only the band of H'_ε used by the
constraint to 1 dx (`reinit.alpha_delta = 1`) gives 128²: −0.8 % / 0.0297. This became IAMReX PR #78 (`ns.epsilon_massfix`).

## B. Single-phase / hydrostatics — vs analytic solutions
| ID | case | grid | quantity | ls2d | reference | verdict |
|---|---|---|---|---|---|---|
| V4 | Taylor–Green, ν 0.01, t 0.2 | 32² / 64² / 128² | L2(u) | 4.81e-4 / 9.08e-5 / 2.37e-5 | order 2 (A98) | ✓ order 2.40, 1.94 |
| V5 | hydrostatic interface, ρ 1:1000, g 9.81, t 0.5 | 64² | max \|u\| | 1.7e-12 | 0 | ✓ no spurious currents |

## C. Two-phase flow — vs literature
| ID | case | grid | quantity | ls2d | reference | verdict |
|---|---|---|---|---|---|---|
| V6 | Rayleigh–Taylor, Tryggvason setup (At 0.5, Re 1000) | 64×256 | spike / bubble tip at t_T = 2.5 | −1.06 / +0.65, secondary roll-up | ≈ −1.1 / +0.65 (Guermond–Salgado 2009 Fig. 1, read off plot) | ✓ within plot-reading accuracy |
| V8a | static drop, R 0.25, σ 1 | 64² | Δp ; max spurious \|u\| | 4.03 ; 1.5e-3 | σ/R = 4 | ✓ +0.8 % |
| V8 | Hysing 2009 rising bubble case 1 (Re 35, Eo 10) | 40×80 | circ_min (t) / v_max (t) / y_c(3) | 0.9060 (1.87) / 0.2385 (0.94) / 1.0856 | 0.9013 (1.90) / 0.2417 (0.92) / 1.0813 (TP2D 1/h=320) | ✓ ≤ 1.3 % |
| V8 | | 80×160 | same | 0.9014 (1.92) / 0.2410 (0.93) / 1.0820 | same | ✓ ≤ 0.3 % |
| V7 | breaking wave | — | — | cancelled (too expensive); `prob.type = breakingwave` kept as example only | — | — |

## D. Code-to-code, ls2d vs IAMReX (same physics, independent implementations)
| case | setup | IAMReX | ls2d | verdict |
|---|---|---|---|---|
| Taylor–Green | 64², ν 0.01, CFL 0.7, t 0.2 | L2(u) 2.16e-4, 18 steps | 1.52e-4, 17 steps | ✓ same order, ls2d slightly better |
| Rayleigh–Taylor (`Tutorials/RayleighTaylor_LS` params) | 64×256, dt 1e-3 | tips at t=0.985: 1.068 / 2.592 | 1.055 / 2.601 | ✓ ≤ 0.015 apart through t ≈ 1 |
| RSV (`Tutorials/RSV`, T 4, 64²) | band 2 dx (old default) | area +18.3 % (mass.txt, PR #78 diagnostic) | +17.1 % | ✓ same defect reproduced |
| RSV | band 1 dx (`ns.epsilon_massfix = 1`, new tutorial default) | +4.5 % | +4.1 % | ✓ |

## E. Deviation / bug findings that came out of the validation
| item | outcome |
|---|---|
| D4 IAMReX `mass_fix` delta support tested on the wrong field | fixed, PR #77 (merged) |
| D13 IAMReX Taylor–Green exact-pressure comment sign; D7 LevelSet.rst μ time level | fixed, PR #77 |
| D14 RSV tutorial +18 % area gain, band width of the constraint | `ns.epsilon_massfix` + phase-volume columns in `mass.txt`, PR #78 (merged) |
| D5 nine-point weights 16/1 | confirmed = Sussman–Fatemi 1999 eq. (4.8), h²/24(16 g_ij + Σ g_nbr) |
| ls2d internal bugs fixed on the way | nodal RHS sign; periodic double count in nodal assembly; nodal MG restriction /16 → /4 (iters 77 → 5) |

Solver cost after the MG fix (RT 64×256): PCG iterations MAC / viscous / nodal = 9 / 3 / 5.

---
---
# Appendix: literature targets and chronological results log (historical; superseded numbers included)

## Literature targets (2D) — numbers to hit, with sources

All errors use the Heaviside-difference measure of Sussman 1999 eq. (80) = Enright 2002 eq. (14): (1/L)∫|H(φ_exact) − H(φ)| dx (L = interface length; Enright normalises by L, Sussman does not — record both).

## V2 Zalesak's disk (Enright 2002 §3.1, Table 1 "level set method")
Setup: domain [0,100]², slotted circle centre (50,75), R=15, slot width 5, slot length 25; u = (π/314)(50−y), v = (π/314)(x−50); one revolution = 628 time units.
| grid | area after 1 rev | % area loss | L1 error | order |
|---|---|---|---|---|
| exact | 582.2 | – | – | – |
| 50² | 0 | 100% | 4.03 | – |
| 100² | 613.0 | −5.3% | 0.61 | 2.7 |
| 200² | 579.1 | 0.54% | 0.08 | 2.9 |
Pass band for ls2d (plain level set, no particles): 100² area loss within ±8 %, 200² within ±1.5 %; L1 error within 2× of table; observed order ≥ 2.

## V3 Single vortex, time-reversed (Enright 2002 §3.2, Table 3)
Setup: unit box, circle R=0.15 at (0.5,0.75), ψ = (1/π) sin²(πx) sin²(πy), velocity × cos(πt/T), T = 8.
| grid | area at t=T | % area loss | L1 error | order |
|---|---|---|---|---|
| exact | 0.0707 | – | – | – |
| 64² | 0 | 100% | 0.075 | – |
| 128² | 0.0425 | 39.8% | 0.031 | 1.3 |
| 256² | 0.0634 | 10.3% | 0.008 | 2.0 |
Note: IAMReX `Tutorials/RSV` uses T = 4 (`totalTimeRsv`), which is a milder test; ls2d runs both T=4 (for IAMReX cross-check) and T=8 (for the literature table).

## V4 Taylor–Green (analytic)
u = sin(2πx)cos(2πy)e^{−8π²νt}, v = −cos(2πx)sin(2πy)e^{−8π²νt}, p = −¼(cos4πx + cos4πy)e^{−16π²νt}; periodic unit box. Target: L2(u) order ≥ 1.9 on 32/64/128/256.

## V6 Rayleigh–Taylor (Tryggvason 1988 setup as restated in Guermond–Salgado 2009 §5.2)
Domain (−d/2,d/2)×(−2d,2d); heavy fluid above; ρ ratio 3 (At = 0.5 Tryggvason def.); η(x) = −0.1 d cos(2πx/d); density profile ρ/ρ_min = 2 + tanh((y−η)/0.01d) (ls2d uses φ = y−η with ε-smoothed Heaviside instead); Re = ρ_min d^{3/2} g^{1/2}/μ = 1000 (and 5000); no-slip top/bottom, symmetry/periodic sides; time scale √(d/g); Tryggvason time t_T = t√At.
Targets: interface snapshots at t_T = 1, 1.5, 1.75, 2, 2.25, 2.5 (Guermond–Salgado Fig. 1; Tryggvason Fig.); spike-tip and bubble-tip y(t) curves; linear-stage growth rate vs. viscous dispersion relation.
IAMReX `RayleighTaylor_LS` differs: domain [0,1]×[0,4], amplitude 0.1, ρ 3:1, ν=0.003132 → matches Re≈1000 with d=1, g=9.81 only if scaled — check before comparing.

## V8 (v2) Rising bubble (Hysing 2009)
Domain [0,1]×[0,2], bubble R=0.25 at (0.5,0.5), no-slip top/bottom, free-slip sides, T=3.
| case | ρ1 | ρ2 | μ1 | μ2 | g | σ | Re | Eo |
|---|---|---|---|---|---|---|---|---|
| 1 | 1000 | 100 | 10 | 1 | 0.98 | 24.5 | 35 | 10 |
| 2 | 1000 | 1 | 10 | 0.1 | 0.98 | 1.96 | 35 | 125 |
Case-1 reference (TP2D, finest): min circularity 0.9013 at t≈1.90; max rise velocity 0.2417 at t≈0.92; y_c(3) = 1.0813. Full time series in `references/hysing_featflow_data/`.

---
# Results log

## M2 (2026-08-25) — ENO2 upwind + Heun advection, Sussman redistance every step (4 iters, dtau=dx/2, eps=2dx), volume constraint on
err/L = Heaviside-difference error divided by initial interface length (Enright 2002 eq. 14). Enright's plain level-set column in brackets.

| Zalesak, 1 revolution | area change | err/L |
|---|---|---|
| 50²  | −0.09 %  (E: −100 %) | 1.71 (E: 4.03) |
| 100² | +2.6 %   (E: +5.3 %) | 0.75 (E: 0.61) |
| 200² | −0.6 %   (E: +0.54 %)| 0.23 (E: 0.08) |

| Single vortex, T = 8 | area loss | err/L |
|---|---|---|
| 64²  | 100 %  (E: 100 %)  | 0.0752 (E: 0.075) |
| 128² | 80 %   (E: 39.8 %) | 0.062 (E: 0.031) |
| 256² | 15.2 % (E: 10.3 %) | 0.017 (E: 0.008) |

64² identical to Enright; finer grids 2–3× worse because Enright advects with WENO5/RK3, we use ENO2/RK2 (the paper's Godunov-PLM comes in M3).

### Deviation sensitivity (Zalesak 100², vortex 128²) — change relative to the default
| variant | Zalesak area / err | vortex area-loss / err | verdict |
|---|---|---|---|
| D1 upwind=average (IAMReX) | +2.58 % / 0.752 (=) | 80.2 % / 0.0615 (=) | negligible |
| D2 sign=peng (IAMReX) | +2.93 % / 0.767 | 80.1 % / 0.0614 | negligible |
| D4 iamrex_delta_bug=1 | +2.59 % / 0.751 (=) | 80.1 % / 0.0614 | negligible in these tests (still a bug: wrong support test) |
| D8 alpha_sign=1 (paper) | +2.58 % / 0.750 | 79.7 % / 0.0611 | negligible |
| D6 sign_guard=0 | — | 80.2 % / 0.0615 (0 hits anyway) | never triggers here |
| volume_fix=0 | −2.25 % / **1.31** | **47.8 %** / **0.040** ; 256²: 12.7 % / 0.0112 | constraint helps Zalesak, hurts the stretching vortex |
| D5 uniform 9-pt weights | — | 83.4 % / 0.063 | worse than 16/1 weights — need Sussman–Fatemi's actual rule |
| n_reinit=2 | — | 79.5 % / 0.061 | insensitive |
| reinit every 5 steps | −8.6 % / 0.98 | 100 % / 0.075 | much worse: with ENO2 advection the field must be redistanced every step |

Static distorted disk (unit test of the constraint, init 40 iters): 32² area drift 0.23 % with constraint vs 0.53 % without; err 4.7e-4 vs 1.05e-3 → constraint sign/scale verified.

## M3 (2026-08-25) — unsplit Godunov-PLM advection (Sussman §3.2, 4th-order limited slopes, transverse terms), reinit every step, volume constraint on
| Zalesak 1 rev | area change | err/L | ENO2 (M2) |
|---|---|---|---|
| 50²  | +7.6 % loss | 1.60 | 1.71 |
| 100² | −1.8 %  (E: −5.3 %) | **0.40** (E: 0.61) | 0.75 |
| 200² | −0.04 % (E: +0.54 %) | **0.036** (E: 0.08) | 0.23 |

| vortex T=8 | area change | err/L | ENO2 (M2) |
|---|---|---|---|
| 64²  | **+101 %** (E: −100 %) | 0.080 (E: 0.075) | 0.0752 |
| 128² | **+29 %** (E: −39.8 %) | 0.0256 (E: 0.031) | 0.062 |
| 256² | +3.1 % (E: −10.3 %) | 0.0079 (E: 0.008) | 0.017 |

Vortex 128² ablations: no reinit → −4.1 % / 0.0198; reinit, volume_fix=0 → −22.6 % / 0.0266; reinit every 5 steps → +31 % / 0.0257.
Zalesak 100² without reinit: Godunov −0.48 % / 0.31 vs ENO2 +0.45 % / 1.53 — the Godunov step is verified independently of the redistance.

Interpretation: the interface error now matches or beats Enright's level-set column. The sign of the area drift flips with the volume constraint: on filaments thinner than 2·eps the constraint preserves the *smoothed* volume ∫H_eps(phi), and flattening a compressed profile (|grad phi| > 1 → 1) while keeping that smoothed volume pushes the zero contour outward. This is a property of the Sussman–Fatemi constraint at under-resolution, not an implementation error (static-disk unit test above). Kept as the paper's default; `reinit.volume_fix = 0` is the documented alternative.

### Volume-constraint band width (vortex 128², Godunov, reinit every step) — the area drift is controlled by the eps used in H'_eps of eq. (68)
| eps for H'_eps (delta) | eps for H_eps (rho, S) | area change | err/L |
|---|---|---|---|
| 2 dx (paper/IAMReX default) | 2 dx | +29 % | 0.0256 |
| 1 dx (`reinit.alpha_delta=1`) | 2 dx | **−0.8 %** | 0.0297 |
| 0.5 dx | 2 dx | −10.9 % | 0.0319 |
| 1 dx (`ns.epsilon=1`) | 1 dx | −2.4 % | 0.0307 |
| 1.5 dx | 1.5 dx | +17 % | 0.0253 |
| 3 dx | 3 dx | +35 % | 0.0272 |
The gain grows monotonically with the delta width: the constraint conserves the eps-smoothed volume, which differs from the sharp area once the filament is thinner than 2 eps. A delta of width dx (the scale the paper uses for S_dx in eq. 57) makes the sharp area nearly conserved at a small cost in interface error.

### Cross-check against IAMReX (RSV tutorial: T = 4, 64², fixed dt = 1/512, periodic, reinit every step)
| code | area t=0 | area t=4 | change |
|---|---|---|---|
| IAMReX `Tutorials/RSV` (development branch, unmodified) | 0.07059 | 0.08320 | **+17.9 %** |
| ls2d, IAMReX-equivalent switches (upwind=average, sign=peng, delta-bug on) | 0.07063 | 0.08277 | +17.1 % |
| ls2d, paper defaults | 0.07063 | 0.08260 | +16.8 % |
ls2d reproduces IAMReX's RSV behaviour (same 17 % area gain); the gain is the volume-constraint band-width effect above, present in IAMReX as shipped.

## M4 (2026-08-25) — projection method (MAC + Crank–Nicolson + Q1 nodal approximate projection, PCG)
Bugs fixed on the way: (1) sign of the nodal RHS (b = Σ V·∫∇N_a); (2) periodic double-counting of cell −1 in the nodal assembly.

### V4 Taylor–Green, ν = 0.01, t = 0.2, CFL 0.5
| grid | L2(u) error | order | max nodal div |
|---|---|---|---|
| 32²  | 4.81e-4 | – | 7.5e-3 |
| 64²  | 9.08e-5 | 2.40 | 8.9e-4 |
| 128² | 2.37e-5 | 1.94 | 1.7e-4 |
Second order in velocity; the approximate projection leaves an O(h²) nodal divergence, as expected (A98).

### V5 hydrostatic interface, ρ 1:1000, g = 9.81, 64², t = 0.5
max|u| = 1.7e-12, KE = 1.5e-24 — no spurious currents from the density jump (balanced Gp/ρ treatment).

### V6 Rayleigh–Taylor, Tryggvason setup (ρ 3:1, At 0.5, Re 1000, 64×256, ls2d Godunov + CN + nodal projection, reinit every step, alpha_delta=1)
Interface tips (zero contour of φ) vs Tryggvason time t_T = t√At; compare Guermond–Salgado 2009 Fig. 1 (Re = 1000, frames at t_T = 1 … 2.5, half domain):
| t | t_T | spike tip y | bubble tip y |
|---|---|---|---|
| 1.420 | 1.00 | -0.367 | 0.297 |
| 2.152 | 1.52 | -0.623 | 0.433 |
| 2.472 | 1.75 | -0.726 | 0.486 |
| 2.832 | 2.00 | -0.837 | 0.545 |
| 3.177 | 2.25 | -0.943 | 0.598 |
| 3.535 | 2.50 | -1.061 | 0.653 |
At t_T = 2.5 the spike reaches y ≈ −1.05 and the bubble y ≈ +0.64 with the secondary roll-up of the mushroom; Guermond–Salgado's last frame shows the spike near −1.1 and the bubble near +0.65 with the same roll-up. Figure: `out/rt64/interface_evolution.png`. Phase area drift over the run: 2.000 → 2.008 (0.4 %).

### Code-to-code: Taylor–Green 64², ν = 0.01, CFL 0.7, t = 0.2
| code | steps | L2(u) error | Linf(u) |
|---|---|---|---|
| IAMReX (`probtype=11`, Godunov_PLM, MLNodeLaplacian) | 18 | 2.16e-4 | 5.3e-4 |
| ls2d | 17 | 1.52e-4 | 6.6e-4 |

### Code-to-code: Rayleigh–Taylor with the IAMReX `Tutorials/RayleighTaylor_LS` parameters (domain [0,1]×[0,4], ρ 3:1, μ 0.003132, g 9.81, 64×256, fixed dt 1e-3, slip walls)
Interface tips (zero contour of φ) from the IAMReX plotfiles vs ls2d VTK at the same steps (`out/rt_code_to_code.png`):
| step | t | IAMReX spike / bubble | ls2d spike / bubble |
|---|---|---|---|
| 250 | 0.235 | 1.826 / 2.166 | 1.816 / 2.174 |
| 500 | 0.485 | 1.597 / 2.318 | 1.581 / 2.327 |
| 750 | 0.735 | 1.321 / 2.462 | 1.306 / 2.471 |
| 1000 | 0.985 | 1.068 / 2.592 | 1.055 / 2.601 |
Differences ≤ 0.015 (≈ 1 % of the excursion) with ls2d marginally ahead — consistent with the remaining discretisation differences (IAMReX AMReX-Hydro Godunov corner coupling, MLNodeLaplacian, MLMG tolerances). Note: with g = 9.81 the tutorial's t = 2.5 corresponds to Tryggvason time t_T ≈ 5.5, far beyond the literature frames; the interface is fragmented there in both codes.

### Multigrid (2026-08-25 late): nodal restriction fixed to P^T (weights /4, not /16)
RT 64×256, 5 steps: PCG iterations MAC/viscous/nodal = 9/3/5 (was 550/4/336 with Jacobi, 9/3/77 with the mis-scaled restriction); wall time 3.6 s → 0.62 s. Results unchanged to all printed digits (TG 64²: L2(u) 9.08293e-5).

## v2 (2026-08-25) — surface tension (S99 §3.3 eq. 34-39), capillary time step (§3.1.1)
### V8a static drop (R 0.25, σ 1, ρ 1:1, μ 0.1, 64², periodic, t = 0.5)
Pressure jump p_max − p_min = 4.03 (Laplace σ/R = 4, +0.8 %); spurious velocity max|u| = 1.5e-3 (capillary number μ|u|/σ ≈ 1.5e-4), steady; circularity 1.002.
### V8 Hysing et al. 2009 rising bubble, test case 1 (Re 35, Eo 10, ρ 1000/100, μ 10/1, σ 24.5, g 0.98)
| grid | min circularity (t) | max rise velocity (t) | y_c(t=3) |
|---|---|---|---|
| 40×80 | 0.9060 (1.87) | 0.2385 (0.94) | 1.0856 |
| 80×160 | 0.9014 (1.92) | 0.2410 (0.93) | 1.0820 |
| reference (TP2D, 1/h = 320) | 0.9013 (1.90) | 0.2417 (0.92) | 1.0813 |

### IAMReX `mass_fix` band width (feature branch, `ns.epsilon_massfix`), RSV tutorial T=4, 64², from the new `mass.txt` phase-volume columns
| band of H'_eps | sharp phi>0 area t=0 → t=4 | change | ls2d prediction |
|---|---|---|---|
| 2 dx (previous behaviour) | 0.07080 → 0.08374 | +18.3 % | +17.1 % |
| 1 dx (new tutorial default) | 0.07080 → 0.07397 | +4.5 % | +4.1 % |
