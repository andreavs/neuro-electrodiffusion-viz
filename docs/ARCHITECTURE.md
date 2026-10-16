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
  F Σ_k z_k ∇·J_k = F Σ_k z_k f_k − i_cap. This gives a linear elliptic equation for φ each step:
  ∇·(σ∇φ) = −F Σ_k z_k ∇·(D_k ∇c_k) + (source terms), with σ = (F/ψ) Σ_k D_k z_k² c_k.
- Porous medium: D̃_k = D_k/λ², λ = 1.6; sources f̃_k = f_k/α, α = 0.2.
- Species: Na⁺, K⁺, X⁻ (generic anion with Cl⁻ properties), Ca²⁺. ECS baseline (mM):
  Na 150, K 3, X 155.8, Ca 1.4.
- Neuron: in the paper, a detailed pyramidal cell simulated in NEURON supplied per-segment,
  per-ion transmembrane currents and the capacitive current as point sources. Here a
  multi-compartment Hodgkin-Huxley neuron runs in Rust alongside the KNP solver
  (one-way coupling, as in the paper).
- Numerics (this repo): cell-centered finite volumes on a uniform 3-D grid (e.g. 64×64×32 over
  ~1 mm), semi-implicit time stepping, potential solved with preconditioned CG warm-started from
  the previous step. Sources are deposited into the grid cell containing each compartment centre.

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
- Numerics (this repo): the paper used mortar FEM in FEniCS. Here we use a voxel finite-volume
  discretisation where each voxel is tagged by cell id (0 = ECS). Rectangular axons align with
  the grid exactly; membrane faces are the faces between differently tagged voxels.

## Scale 3: Brownian electrodiffusion (Publication 3)

- Each step, a particle moves r → r + X + (z D Δt / ψ) E, with X ~ N(0, 2DΔt) per axis.
  This is exact for unbounded motion in a constant field at any Δt.
- Ca²⁺: z = 2, D = 0.4 µm²/ms (4·10⁻¹⁰ m²/s).
- Dendritic spine meshes: spherical head r_h = 0.5 µm, neck radius 0.1 µm, neck length 1.0 µm
  (sharp junction), or 0.89 µm with a toroidal funnel of radius r_h/2 − ε (funneled).
- Walls reflect; the bottom of the neck absorbs. Quantity of interest: mean first passage
  time (MFPT) from the head centre, with and without a field along the neck. Boundary effects
  converge for Δt ≲ 10⁻⁴ ms.
