# Architecture

## Runtime

```
UI thread (web/src)                     Worker (web/src/sim/worker.ts)
-------------------                     ------------------------------
menu → hash route (#/knp|#/emi|#/particles)
ScaleView.mount(root, SimClient) ──────▶ init: load WASM
  ControlPanel → ToWorker messages ────▶ create / run / setParams / action / reset / setBudget / dispose
  startRenderLoop (rAF): takeFrame ◀──── frame { seq, timeMs, fields: Float32Array[] (transferred), stats, runtime }
                         ack(seq) ─────▶ RunLoop (web/src/sim/loop.ts) posts the next frame
```

- Back-pressure: at most one frame is in flight. The worker's `RunLoop` keeps ticking up to two
  ticks ahead while waiting for an `ack`, then idles, so a hidden tab stops the solver.
- A scale view typically does: `new ControlPanel(root, client, schema)`, `await client.ready()`,
  `client.create(scale, panel.params())`, `startRenderLoop(client, { draw, stats: panel })`.
  Both register with `client.addDisposer`, so terminating the client cleans them up.

- `web/src/sim/protocol.ts`: message types shared by both sides.
- `web/src/sim/solver.ts`: `WorkerSolver` interface. Each scale implements it in
  `web/src/sim/solvers/<scale>.ts` by wrapping its wasm-bindgen struct from `crates/ed-wasm/src/<scale>.rs`.
- `web/src/scales/<scale>/view.ts`: the scale's `ScaleView` (DOM, WebGL, controls).
- `web/src/scales/<scale>/about.ts`: theory content (equations, assumptions, parameters, references).
  Views show it with `createAboutPanel(id)` from `web/src/ui/about.ts`; the content and the
  TeX→MathML renderer (Temml) load lazily on first hover/focus/open.
- Rust: `crates/ed-core` (constants, ions, numerics, membrane models), one crate per scale,
  and `crates/ed-wasm` (the only crate that depends on `wasm-bindgen`). Solver crates stay
  plain Rust so they can be unit-tested and benchmarked natively.

Deep links use the URL hash so GitHub Pages needs no rewrites. GitHub Pages cannot set
COOP/COEP headers, so `SharedArrayBuffer` is not available; use transferable buffers.

## Shared physics (thesis background, ch. 2)

Nernst-Planck flux for ion species k:

    J_k = -D_k ∇c_k - (D_k z_k / ψ) c_k ∇φ,     ψ = RT/F ≈ 25.85 mV at 300 K
    ∂c_k/∂t = -∇·J_k + f_k

| Ion | z | D (µm²/ms) |
|---|---|---|
| Na⁺ | +1 | 1.33 |
| K⁺ | +1 | 1.96 |
| Cl⁻ / X⁻ | −1 | 2.03 |
| Ca²⁺ | +2 | 0.71 |

Constants: R = 8.314 J/(K mol), F = 9.648·10⁴ C/mol, T = 300 K.

## Scale 1: KNP with point-source neurons (Publication 1)

- Electroneutral KNP: instead of Poisson's equation, require ∂ρ/∂t = −i_cap everywhere, i.e.
  F Σ_k z_k ∇·J_k = F Σ_k z_k f_k + i_cap, with ionic and capacitive currents positive when
  leaving the neuron. This gives a linear elliptic equation for φ each step:
  ∇·(σ∇φ) = −F Σ_k z_k ∇·(D_k ∇c_k) − F Σ_k z_k f_k − i_cap, with σ = (F/ψ) Σ_k D_k z_k² c_k.
- Porous medium: D̃_k = D_k/λ², λ = 1.6; sources f̃_k = f_k/α, α = 0.2.
- Species: Na⁺, K⁺, X⁻ (generic anion with Cl⁻ properties), Ca²⁺. ECS baseline (mM):
  Na 150, K 3, X 155.8, Ca 1.4.
- Neuron: in the paper, a detailed pyramidal cell simulated in NEURON supplied per-segment,
  per-ion transmembrane currents and the capacitive current as point sources. Here a
  multi-compartment Hodgkin-Huxley neuron runs in Rust alongside the KNP solver
  (one-way coupling, as in the paper).
- Numerics (this repo, `crates/ed-knp`): cell-centered finite volumes on a uniform 3-D grid
  (default 64×64×32 over 1×1×0.5 mm). Each step: backward Euler diffusion per species, then an
  electroneutral projection that solves for φ (multigrid-preconditioned CG, warm-started) and
  applies the drift, so Δt is not bound by a diffusion CFL limit. Sources are deposited into
  the grid cell containing each compartment centre.

## Scale 2: KNP-EMI (Publication 2)

- Separate intracellular (Ω_i) and extracellular (Ω_e) domains, each with KNP electroneutral
  bulk equations; coupled across the membrane Γ.
- φ_M = φ_i − φ_e, C_M ∂φ_M/∂t = I_M − I_ch.
- Ion-specific fluxes across Γ: F z_k J_k·n = I_ch^k + α_r^k (I_M − I_ch), with
  α_r^k = D_r^k (z^k)² [k]_r / Σ_l D_r^l (z^l)² [l]_r.
- Membrane: passive leak (g_L^Na 0.2, g_L^K 0.8, g_L^Cl 0 mS/cm²) plus Hodgkin-Huxley
  (ḡ_Na 120, ḡ_K 36 mS/cm²), E_k from Nernst with the local concentrations.
- Initial concentrations (mM): Na i/e 12/100, K 125/4, Cl 137/104; φ_M⁰ = −67.74 mV;
  m₀ 0.0379, h₀ 0.688, n₀ 0.276; C_M = 1 µF/cm².
- Stimulus: synaptic conductance g_syn = 4 mS/cm² with time constant 2 ms on part of an axon.
- Time steps: PDE Δt = 0.1 ms, membrane ODE substeps Δt/25.
- Geometry: bundle of unmyelinated axons modelled as rectangular blocks inside an ECS box.
- Geometry (paper): Ω = [0, 400] × [0, 1.4] × [0, 1.4] µm with 9 axons of 390 × 0.2 × 0.2 µm,
  0.1 µm apart; centre axon A at [5, 395] × [0.6, 0.8]². Closed outer boundary for all ions.
- Numerics (this repo, `crates/ed-emi`): P1 finite elements, following the paper's mortar scheme
  (which used FEniCS + MUMPS) with simplifications for speed:
  - Mesh (`mesh.rs`): a rectilinear grid whose planes include every cell face (spacing ≤
    `max_h` per axis), each hex split into 6 Kuhn tetrahedra. Ω_e and each Ω_i get their
    own P1 DOFs, so membrane vertices carry one DOF per side and the meshes share facets on Γ.
  - I_M lives in the P1 interface space Q_h with a lumped interface mass, so its DOFs are the
    membrane vertices and it can be eliminated: backward Euler on (29) gives
    I_M = (C_M/Δt)(φ_i − φ_e − φ_M*) at each membrane vertex (plus the implicit leak current for
    passive membranes), a Robin-type coupling between the two DOFs. Volume mass matrices are
    lumped too (an M-matrix for diffusion; the drift and membrane terms are explicit, so a
    step too large for the currents can still make a concentration negative, which `step`
    reports). Weighted P1 stiffness matrices (`fem.rs`) are
    reassembled in place each step from precomputed element edge coefficients.
  - Time step (backward Euler, Δt = 0.1 ms; implicit because the coupled system is stiff):
    1. Membrane ODE step (HH only): gates and φ_M with I_M = 0 over 5 substeps (exponential
       update for φ_M, Rush-Larsen for the gates, stable for any substep; the paper used 25
       forward Euler substeps), giving φ_M* and the per-ion channel currents averaged exactly
       over the step (so the ions moved match the capacitor discharge). Passive membranes skip it and
       put the leak currents implicitly into step 2, as the paper does (eq. 30).
    2. Potential: one SPD system for all φ DOFs, the charge equations (27)-(28) with σ and the
       diffusion-current term from [k]^{n−1}, plus a charge-correction term M Σ_k z_k [k]^{n−1}/Δt
       that pulls the bulk back to electroneutrality. Closed boundary ⇒ constant null space.
    3. Concentrations: per species, (M/Δt + D K)[k]^n = M [k]^{n−1}/Δt − drift(φ^n, [k]^{n−1})
       + membrane fluxes (I_ch^k + α_r^k (I_M − I_ch))/(F z_k), i.e. the paper's linearised flux.
    Steps 2-3 decouple the paper's monolithic solve so each system is SPD. Each is solved as a
    warm-started increment with PCG (relative tolerance 1e-2 of the step's change) and a
    two-level preconditioner (Gauss-Seidel plus one aggregate per x-plane and subdomain, which
    captures the weak coupling along the axons and between subdomains). A per-subdomain
    correction keeps ion amounts exactly consistent with the membrane fluxes.
  - Default geometry (`MeshSpec::default_bundle`): the paper's 3×3 cross-section with 200 µm
    axons on a 5 µm × ≤0.3 µm grid (≈4 k DOFs per field, ≈1.5 k membrane nodes). Native release
    build: ≈5 ms per step, ≈20 simulated ms per wall second
    (`cargo run --release -p ed-emi --example bench_step`). Results match a 0.1 µm cross-section
    grid to plotting accuracy.
  - Validation (crates/ed-emi/tests): rest stays at rest; a single axon's AP propagates at
    ≈0.2 m/s, within 3% of a converged 1-D HH cable with the same membrane and r_i + r_e
    (at Δt = 0.1 ms it is ≈5% slower, as expected for first-order time stepping; 0.17 m/s
    on the 390 µm paper bundle); [K]_e rises near the active membrane; charge is conserved
    exactly, each ion amount within the membrane-capacitor charge, and the bulk stays neutral
    to < 0.01 mM; a passive axon matches cable theory (λ = sqrt(r_m/(r_i + r_e)) ≈ 95 µm) to 2%.

## Scale 3: Brownian electrodiffusion (Publication 3)

- Each step, a particle moves r → r + X + (z D Δt / ψ) E, with X ~ N(0, 2DΔt) per axis.
  This is exact for unbounded motion in a constant field at any Δt.
- Ca²⁺: z = 2, D = 0.4 µm²/ms (4·10⁻¹⁰ m²/s).
- Dendritic spine meshes: spherical head r_h = 0.5 µm, neck radius 0.1 µm, neck length 1.0 µm
  (sharp junction), or 0.89 µm with a toroidal funnel of radius r_h/2 − ε (funneled).
- Walls reflect; the bottom of the neck absorbs. Quantity of interest: mean first passage
  time (MFPT) from the head centre, with and without a field along the neck. Boundary effects
  converge for Δt ≲ 10⁻⁴ ms.
