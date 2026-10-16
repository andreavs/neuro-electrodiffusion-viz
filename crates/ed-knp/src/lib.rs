//! Tissue-scale KNP electrodiffusion with point-source neurons (Publication 1).
//!
//! Cell-centred finite volumes on a uniform 3-D grid. Each step of length Δt:
//!
//! 1. Backward Euler diffusion with sources, per species:
//!    `(c* − cⁿ)/Δt = ∇·(D̃_k∇c*) + f̃_k`.
//! 2. Electroneutral drift: solve `∇·(σ∇φ) = (ρ − Σ z c*)/Δt` with the target charge ρ and
//!    `σ ≈ κ* = Σ_k D̃_k z_k² c*_k / ψ`, then `cⁿ⁺¹ = c* + Δt ∇·(D̃ z c* ∇φ / ψ)`.
//!
//! Step 2 is the KNP potential equation (divided by F, `q̃_cap = ĩ_cap / F`) written as a
//! projection: φ is the potential whose drift restores the charge density that diffusion and
//! the ionic sources disturbed, so `Σ z c` follows `∂ρ/∂t = −q̃_cap` to within
//! [`KnpConfig::charge_tol`], without accumulating error. The drift fluxes use the same face
//! coefficients as the potential operator, and both steps are written in flux form, so every
//! species is conserved to round-off with no-flux boundaries.
//!
//! Diffusion is unconditionally stable, and the drift only moves the charge mismatch of one
//! step, so Δt is not bound by a diffusion CFL limit (tests run 50 ms steps on 15.6 µm cells).
//! The drift uses central face concentrations, so a step that must move more charge than a
//! dilute species holds next to a steep gradient can still push it negative.
//!
//! With no-flux boundaries on every side, charge cannot leave the box: the net membrane
//! current (ionic plus capacitive) must sum to zero, as it does for a whole neuron. A net
//! imbalance is spread uniformly over the domain each step instead of being neutralised.
//!
//! Sign convention: ionic and capacitive currents are positive when they leave the neuron
//! into the extracellular space. Sources and sinks are per tissue volume and scaled by 1/α.

use ed_core::constants::{psi_mv, T_DEFAULT};
use ed_core::ions::{self, IonSpecies};
use ed_core::linalg::{Boundary, BoundaryConditions, CgStats, Grid};
use ed_core::Simulation;

pub mod cell;
mod diffusion;
mod electroneutral;
pub mod fv;
pub mod hay;
pub mod neuron;
pub mod population;
mod sources;
pub mod spreading;
pub mod tissue;

use diffusion::Diffusion;
use electroneutral::Projection;

/// Parameters of a [`KnpSolver`].
#[derive(Clone, Debug)]
pub struct KnpConfig {
    /// Cells per axis.
    pub n: [usize; 3],
    /// Domain size per axis (µm).
    pub size: [f64; 3],
    pub species: Vec<IonSpecies>,
    /// Baseline (initial and Dirichlet boundary) concentration per species (mM).
    pub baseline: Vec<f64>,
    /// Tortuosity λ: effective diffusion D̃ = D / λ².
    pub tortuosity: f64,
    /// Volume fraction α: sources are scaled by 1/α.
    pub porosity: f64,
    /// Temperature (K), for ψ = RT/F in the potential.
    pub temperature: f64,
    /// Time step (ms).
    pub dt: f64,
    /// Outer boundary: `Dirichlet` (baseline concentrations, φ = 0) or `Neumann` (no flux).
    /// Axes with a single cell always get no-flux faces.
    pub boundary: Boundary,
    /// Tolerance of the potential solve, as the RMS charge-density error Σ z c (mM) it may
    /// leave per step. The error does not accumulate: each step projects onto the target.
    /// The default, 1e-6 mM, keeps φ within 1e-4 mV of a tight solve on the default grid.
    pub charge_tol: f64,
}

impl Default for KnpConfig {
    /// 64×64×32 cells over 1×1×0.5 mm with Na⁺, K⁺, X⁻ and Ca²⁺ at ECS baseline.
    fn default() -> Self {
        let x = IonSpecies {
            name: "X-",
            ..ions::CL
        };
        Self {
            n: [64, 64, 32],
            size: [1000.0, 1000.0, 500.0],
            species: vec![ions::NA, ions::K, x, ions::CA],
            baseline: vec![150.0, 3.0, 155.8, 1.4],
            tortuosity: 1.6,
            porosity: 0.2,
            temperature: T_DEFAULT,
            dt: 0.1,
            boundary: Boundary::Dirichlet,
            charge_tol: 1e-6,
        }
    }
}

/// The fixed discretisation shared by the steps.
pub(crate) struct Model {
    grid: Grid,
    bc: BoundaryConditions,
    /// Σ 2/h² over each cell's Dirichlet faces (all zero with no-flux boundaries).
    boundary_weight: Vec<f64>,
    /// D̃_k, z_k and baseline per species.
    d: Vec<f64>,
    z: Vec<f64>,
    baseline: Vec<f64>,
    psi: f64,
}

/// Electroneutral KNP solver on a uniform grid. Fields are flat, x fastest.
pub struct KnpSolver {
    model: Model,
    dt: f64,
    time: f64,
    c: Vec<Vec<f64>>,
    /// Concentration sources per species and capacitive charge sink (mM/ms), and the cells
    /// that have any.
    src: Vec<Vec<f64>>,
    cap: Vec<f64>,
    has_source: Vec<bool>,
    source_cells: Vec<usize>,
    /// 1 / (α · cell volume): converts an amount rate to a concentration rate.
    source_scale: f64,
    diffusion: Diffusion,
    projection: Projection,
}

impl KnpSolver {
    pub fn new(cfg: &KnpConfig) -> Self {
        let ns = cfg.species.len();
        assert_eq!(cfg.baseline.len(), ns, "one baseline value per species");
        assert!(
            cfg.n.iter().all(|&n| n > 0) && cfg.size.iter().all(|&s| s > 0.0),
            "grid needs a positive size and at least one cell per axis"
        );
        assert!(cfg.tortuosity > 0.0 && cfg.porosity > 0.0 && cfg.temperature > 0.0);
        assert!(
            cfg.species.iter().all(|s| s.d >= 0.0),
            "negative diffusion coefficient"
        );
        let h = [0, 1, 2].map(|a| cfg.size[a] / cfg.n[a] as f64);
        let grid = Grid::new_3d(cfg.n, h);
        let len = grid.len();
        // Axes with a single cell are not resolved: give them no-flux faces so a 1-D or 2-D
        // configuration is not pinned to the baseline through its thin sides.
        let mut bc = BoundaryConditions::uniform(cfg.boundary);
        let mut boundary_weight = vec![0.0; len];
        for (a, ha) in h.iter().enumerate() {
            if cfg.n[a] == 1 {
                bc.lower[a] = Boundary::Neumann;
                bc.upper[a] = Boundary::Neumann;
            } else if cfg.boundary == Boundary::Dirichlet {
                let w = 2.0 / (ha * ha);
                fv::for_each_boundary_cell(&grid, a, |p| boundary_weight[p] += w);
            }
        }
        let tau2 = cfg.tortuosity * cfg.tortuosity;
        let model = Model {
            grid,
            bc,
            boundary_weight,
            d: cfg.species.iter().map(|s| s.d / tau2).collect(),
            z: cfg.species.iter().map(|s| s.z as f64).collect(),
            baseline: cfg.baseline.clone(),
            psi: psi_mv(cfg.temperature),
        };
        assert!(cfg.dt > 0.0, "time step must be positive");
        Self {
            diffusion: Diffusion::new(&model, cfg.dt),
            projection: Projection::new(&model, cfg.charge_tol, cfg.dt),
            dt: cfg.dt,
            time: 0.0,
            c: cfg.baseline.iter().map(|&b| vec![b; len]).collect(),
            src: vec![vec![0.0; len]; ns],
            cap: vec![0.0; len],
            has_source: vec![false; len],
            source_cells: Vec::new(),
            source_scale: 1.0 / (cfg.porosity * h[0] * h[1] * h[2]),
            model,
        }
    }

    pub fn grid(&self) -> &Grid {
        &self.model.grid
    }

    /// Sets the time step (ms). Allocates.
    pub fn set_dt(&mut self, dt: f64) {
        assert!(dt > 0.0, "time step must be positive");
        self.dt = dt;
        self.diffusion.set_dt(&self.model, dt);
        self.projection.set_dt(&self.model, dt);
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    pub fn num_species(&self) -> usize {
        self.c.len()
    }

    /// Concentration of species `k` (mM).
    pub fn concentration(&self, k: usize) -> &[f64] {
        &self.c[k]
    }

    /// Mutable concentrations of species `k`, for setting initial conditions. The charge
    /// density `Σ z c` after the edit becomes the one the solver keeps.
    pub fn concentration_mut(&mut self, k: usize) -> &mut [f64] {
        self.projection.reset_charge();
        &mut self.c[k]
    }

    /// Potential from the last step (mV).
    pub fn potential(&self) -> &[f64] {
        self.projection.potential()
    }

    /// Statistics of the last potential solve.
    pub fn last_solve(&self) -> CgStats {
        self.projection.last_solve()
    }
}

impl Simulation for KnpSolver {
    fn step(&mut self, n: usize) {
        for _ in 0..n {
            self.projection.prepare(&self.model, &self.c);
            self.diffusion.step(&self.model, &mut self.c, &self.src);
            self.projection
                .step(&self.model, self.dt, &mut self.c, &self.cap);
            self.time += self.dt;
        }
    }

    fn time_ms(&self) -> f64 {
        self.time
    }
}
