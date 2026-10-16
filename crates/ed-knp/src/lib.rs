//! Tissue-scale KNP electrodiffusion with point-source neurons (Publication 1).
//!
//! Cell-centred finite volumes on a uniform 3-D grid. Each step of length Δt advances every
//! species with backward Euler diffusion and its sources:
//! `(c* − cⁿ)/Δt = ∇·(D̃_k∇c*) + f̃_k`, which is unconditionally stable. The step is written
//! in flux form, so every species is conserved to round-off with no-flux boundaries.
//!
//! The electroneutral drift that couples the species through the KNP potential is added on
//! top of this step in a follow-up.
//!
//! Sources and sinks are per tissue volume and scaled by 1/α.

use ed_core::constants::T_DEFAULT;
use ed_core::ions::{self, IonSpecies};
use ed_core::linalg::{Boundary, BoundaryConditions, Grid};
use ed_core::Simulation;

mod diffusion;
pub mod fv;
pub mod neuron;
mod sources;

use diffusion::Diffusion;

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
    /// Temperature (K).
    pub temperature: f64,
    /// Time step (ms).
    pub dt: f64,
    /// Outer boundary: `Dirichlet` (baseline concentrations, φ = 0) or `Neumann` (no flux).
    /// Axes with a single cell always get no-flux faces.
    pub boundary: Boundary,
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
        }
    }
}

/// The fixed discretisation shared by the steps.
pub(crate) struct Model {
    grid: Grid,
    bc: BoundaryConditions,
    /// Σ 2/h² over each cell's Dirichlet faces (all zero with no-flux boundaries).
    boundary_weight: Vec<f64>,
    /// D̃_k and baseline per species.
    d: Vec<f64>,
    baseline: Vec<f64>,
}

/// Electroneutral KNP solver on a uniform grid. Fields are flat, x fastest.
pub struct KnpSolver {
    model: Model,
    dt: f64,
    time: f64,
    c: Vec<Vec<f64>>,
    /// Concentration sources per species (mM/ms), and the cells that have any.
    src: Vec<Vec<f64>>,
    has_source: Vec<bool>,
    source_cells: Vec<usize>,
    /// 1 / (α · cell volume): converts an amount rate to a concentration rate.
    source_scale: f64,
    diffusion: Diffusion,
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
            baseline: cfg.baseline.clone(),
        };
        assert!(cfg.dt > 0.0, "time step must be positive");
        Self {
            diffusion: Diffusion::new(&model, cfg.dt),
            dt: cfg.dt,
            time: 0.0,
            c: cfg.baseline.iter().map(|&b| vec![b; len]).collect(),
            src: vec![vec![0.0; len]; ns],
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

    /// Mutable concentrations of species `k`, for setting initial conditions.
    pub fn concentration_mut(&mut self, k: usize) -> &mut [f64] {
        &mut self.c[k]
    }
}

impl Simulation for KnpSolver {
    fn step(&mut self, n: usize) {
        for _ in 0..n {
            self.diffusion.step(&self.model, &mut self.c, &self.src);
            self.time += self.dt;
        }
    }

    fn time_ms(&self) -> f64 {
        self.time
    }
}
