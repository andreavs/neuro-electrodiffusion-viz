//! Tissue-scale KNP electrodiffusion with point-source neurons (Publication 1).
//!
//! The extracellular space is a porous medium on a uniform cell-centred finite-volume grid.
//! Each time step:
//!
//! 1. the electroneutral KNP potential equation (Publication 1, Eq 12)
//!    `∇·(σ∇φ + ∇b) + F Σ_k z_k f̃_k + ĩ_cap = 0`, with `σ = (F/ψ) Σ_k D̃_k z_k² c_k` and
//!    `b = F Σ_k z_k D̃_k c_k`, is solved for φ with multigrid-preconditioned CG, warm
//!    started from the previous step;
//! 2. the concentrations take an explicit (forward Euler) step with the Nernst-Planck
//!    fluxes `J_k = -D̃_k ∇c_k - (D̃_k z_k / ψ) c_k ∇φ` and the sources `f̃_k`.
//!
//! The potential operator's face conductivity is the harmonic mean of its cell values, and
//! the drift flux uses face concentrations scaled so that `Σ_k z_k J_k` matches that
//! operator exactly, so the charge flux cannot build up charge in a cell. The right-hand
//! side also asks each step to remove any local charge `Σ_k z_k c_k` left by earlier CG
//! residuals, so that error does not accumulate: bulk charge stays below
//! [`CHARGE_TOLERANCE`].
//!
//! Explicit stepping is stable for `Δt ≤ 1 / (2 D̃_max Σ_a h_a⁻²)`. Linearised around a
//! uniform state, the electroneutral drift turns the system into a diffusion whose
//! effective coefficients lie in `[0, D̃_max]`, so the pure-diffusion limit is also the
//! KNP limit. On the default grid (15.6 µm cells, `D̃_max ≈ 0.79 µm²/ms`) that limit is
//! about 50 ms, far above the 0.1 ms the app uses, so explicit drift and diffusion cost one
//! elliptic solve per step where a semi-implicit scheme would need one per species as well.
//! Finer grids are handled by splitting [`KnpConfig::dt`] into equal substeps.

use ed_core::constants::{psi_mv, T_DEFAULT};
use ed_core::ions::{IonSpecies, CA, CL, K, NA};
use ed_core::linalg::{
    BoundaryConditions, CgOptions, CgStats, DiffusionOperator, Grid, Multigrid, NullSpace, Pcg,
};
use ed_core::Simulation;
use fv::{add_net_inflow, for_each_face_run, stride};

pub mod fv;

/// Generic anion with the properties of Cl⁻ (Publication 1, Table 2).
pub const X: IonSpecies = IonSpecies { name: "X-", ..CL };

/// A simulated species and its initial, uniform concentration (mM).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Species {
    pub ion: IonSpecies,
    pub baseline: f64,
}

/// Parameters of a [`KnpSolver`]. The outer boundary has zero flux for every species, and
/// φ is defined up to a constant and has zero mean.
#[derive(Clone, Debug, PartialEq)]
pub struct KnpConfig {
    /// Cells per axis. Sizes of the form `k·2^m` keep the multigrid preconditioner fast.
    pub cells: [usize; 3],
    /// Size of the box (µm), lower corner at the origin.
    pub extent: [f64; 3],
    pub species: Vec<Species>,
    /// Tortuosity λ: `D̃ = D / λ²`.
    pub tortuosity: f64,
    /// ECS volume fraction α: sources are scaled by `1 / α`.
    pub volume_fraction: f64,
    /// Time step (ms) of one [`Simulation::step`].
    pub dt: f64,
    /// Temperature (K).
    pub temperature: f64,
}

impl Default for KnpConfig {
    /// 64×64×32 cells over 1 × 1 × 0.5 mm with the ECS baseline of Publication 1.
    fn default() -> Self {
        Self {
            cells: [64, 64, 32],
            extent: [1000.0, 1000.0, 500.0],
            species: vec![
                Species {
                    ion: NA,
                    baseline: 150.0,
                },
                Species {
                    ion: K,
                    baseline: 3.0,
                },
                Species {
                    ion: X,
                    baseline: 155.8,
                },
                Species {
                    ion: CA,
                    baseline: 1.4,
                },
            ],
            tortuosity: 1.6,
            volume_fraction: 0.2,
            dt: 0.1,
            temperature: T_DEFAULT,
        }
    }
}

/// Explicit steps use at most this fraction of the forward Euler diffusion limit.
const CFL_SAFETY: f64 = 0.9;

/// Bound (mM) on the bulk charge `Σ_k z_k c_k` left by CG residuals on the default
/// grid: below 1e-9 of the total ion concentration (≈ 310 mM).
pub const CHARGE_TOLERANCE: f64 = 1e-7;

/// The tissue-scale KNP solver. Fields are flat arrays in [`Grid`] order (x fastest).
pub struct KnpSolver {
    cfg: KnpConfig,
    grid: Grid,
    /// Effective diffusion coefficient D̃ per species (µm²/ms).
    d: Vec<f64>,
    z: Vec<f64>,
    psi: f64,
    c: Vec<Vec<f64>>,
    phi: Vec<f64>,
    /// `σ / F` per cell from the current concentrations.
    sigma: Vec<f64>,
    /// `b / F` per cell.
    b: Vec<f64>,
    rhs: Vec<f64>,
    /// `(φ_Q − φ_P) / (2ψ) · σ_face / σ_mean` on the upper face of each cell, per axis,
    /// where `σ_face = harm(σ_P, σ_Q)` is the potential operator's face conductivity.
    drift: [Vec<f64>; 3],
    /// Flux out of each cell through its upper face, per axis. Stays 0 on the last layer.
    flux: [Vec<f64>; 3],
    dc: Vec<f64>,
    op: DiffusionOperator,
    mg: Multigrid,
    pcg: Pcg,
    cg_opts: CgOptions,
    last_cg: CgStats,
    substeps: usize,
    time: f64,
}

impl KnpSolver {
    pub fn new(cfg: KnpConfig) -> Self {
        assert!(!cfg.species.is_empty(), "at least one species is needed");
        assert!(cfg.tortuosity > 0.0 && cfg.volume_fraction > 0.0 && cfg.dt > 0.0);
        let q: f64 = cfg
            .species
            .iter()
            .map(|s| s.ion.z as f64 * s.baseline)
            .sum();
        let total: f64 = cfg
            .species
            .iter()
            .map(|s| s.ion.z.abs() as f64 * s.baseline)
            .sum();
        assert!(
            q.abs() <= 1e-9 * total,
            "baseline is not electroneutral: Σ z c = {q} mM"
        );
        let h = [0, 1, 2].map(|a| cfg.extent[a] / cfg.cells[a] as f64);
        let grid = Grid::new_3d(cfg.cells, h);
        let n = grid.len();
        let d: Vec<f64> = cfg
            .species
            .iter()
            .map(|s| s.ion.d / (cfg.tortuosity * cfg.tortuosity))
            .collect();
        let z = cfg.species.iter().map(|s| s.ion.z as f64).collect();
        let op = DiffusionOperator::uniform(grid, BoundaryConditions::neumann(), 1.0);
        let mg = Multigrid::new(&op);

        let d_max = d.iter().cloned().fold(0.0, f64::max);
        // Axes with a single cell have no faces and do not limit the step.
        let inv_h2: f64 = (0..3)
            .filter(|&a| cfg.cells[a] > 1)
            .map(|a| 1.0 / (h[a] * h[a]))
            .sum();
        let dt_max = CFL_SAFETY / (2.0 * d_max * inv_h2);
        let substeps = if inv_h2 > 0.0 {
            (cfg.dt / dt_max).ceil().max(1.0) as usize
        } else {
            1
        };

        let mut solver = Self {
            grid,
            d,
            z,
            psi: psi_mv(cfg.temperature),
            c: cfg.species.iter().map(|s| vec![s.baseline; n]).collect(),
            phi: vec![0.0; n],
            sigma: vec![0.0; n],
            b: vec![0.0; n],
            rhs: vec![0.0; n],
            drift: [vec![0.0; n], vec![0.0; n], vec![0.0; n]],
            flux: [vec![0.0; n], vec![0.0; n], vec![0.0; n]],
            dc: vec![0.0; n],
            op,
            mg,
            pcg: Pcg::new(n),
            cg_opts: CgOptions {
                rtol: 1e-6,
                atol: 1e-12,
                max_iter: 200,
                null_space: NullSpace::Constant,
            },
            last_cg: CgStats {
                iterations: 0,
                residual: 0.0,
                converged: true,
            },
            substeps,
            time: 0.0,
            cfg,
        };
        solver.solve_potential();
        solver
    }

    pub fn config(&self) -> &KnpConfig {
        &self.cfg
    }

    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    /// Concentration of species `k` (mM).
    pub fn concentration(&self, k: usize) -> &[f64] {
        &self.c[k]
    }

    /// Mutable concentrations of species `k`, for setting initial conditions, which must be
    /// electroneutral, and every cell needs some ions (σ > 0). Steps remove local charge
    /// `Σ z_k c_k`, but with no-flux boundaries the box's total charge cannot change. Call
    /// [`solve_potential`](Self::solve_potential) to refresh [`potential`](Self::potential).
    pub fn concentration_mut(&mut self, k: usize) -> &mut [f64] {
        &mut self.c[k]
    }

    /// Extracellular potential (mV) from the last solve. After [`Simulation::step`] that is
    /// the potential that drove the last substep, i.e. from the concentrations before it.
    pub fn potential(&self) -> &[f64] {
        &self.phi
    }

    /// CG statistics of the last potential solve.
    pub fn last_solve(&self) -> CgStats {
        self.last_cg
    }

    /// Explicit substeps taken per [`Simulation::step`].
    pub fn substeps(&self) -> usize {
        self.substeps
    }

    /// Solves the KNP equation for φ from the current concentrations.
    pub fn solve_potential(&mut self) {
        self.solve(None);
    }

    /// Solves for φ. With `dt`, the right-hand side also removes any net charge in a cell
    /// over that step, so CG residuals do not accumulate into a charge drift.
    fn solve(&mut self, dt: Option<f64>) {
        let (z, d) = (&self.z, &self.d);
        let rho = &mut self.dc;
        self.sigma.fill(0.0);
        self.b.fill(0.0);
        rho.fill(0.0);
        for (k, c) in self.c.iter().enumerate() {
            let (zk, dz, dz2) = (z[k], d[k] * z[k], d[k] * z[k] * z[k]);
            for (cell, &c) in c.iter().enumerate() {
                self.sigma[cell] += dz2 * c;
                self.b[cell] += dz * c;
                rho[cell] += zk * c;
            }
        }
        let (inv_psi, inv_dt) = (1.0 / self.psi, dt.map_or(0.0, |dt| 1.0 / dt));
        for (cell, s) in self.sigma.iter_mut().enumerate() {
            *s *= inv_psi;
            self.rhs[cell] = rho[cell] * inv_dt;
        }
        // rhs = ∇²b / F.
        let (b, rhs) = (&self.b, &mut self.rhs);
        for a in 0..3 {
            let w = 1.0 / (self.grid.h[a] * self.grid.h[a]);
            let (flux, s) = (&mut self.flux[a], stride(&self.grid, a));
            for_each_face_run(&self.grid, a, |p, len| {
                for p in p..p + len {
                    flux[p] = w * (b[p] - b[p + s]);
                }
            });
        }
        add_net_inflow(&self.grid, &self.flux, rhs);
        self.op.update_sigma(&self.sigma);
        self.mg.update(&self.op);
        self.last_cg = self.pcg.solve(
            &self.op,
            &mut self.mg,
            &self.rhs,
            &mut self.phi,
            &self.cg_opts,
        );
    }

    fn substep(&mut self, dt: f64) {
        self.solve(Some(dt));
        debug_assert!(
            self.last_cg.converged,
            "KNP potential solve failed: {:?}",
            self.last_cg
        );
        let (grid, phi, sigma, op_sigma) = (&self.grid, &self.phi, &self.sigma, self.op.sigma());
        let inv_2psi = 0.5 / self.psi;
        for (a, drift) in self.drift.iter_mut().enumerate() {
            let s = stride(grid, a);
            for_each_face_run(grid, a, |p, len| {
                for p in p..p + len {
                    // Face concentrations (c_P + c_Q)/2 · σ_face / σ_mean make Σ_k z_k J_k
                    // use exactly the operator's face conductivity σ_face = harm(σ_P, σ_Q).
                    let (sp, sq) = (op_sigma[p], op_sigma[p + s]);
                    let sum = sigma[p] + sigma[p + s];
                    let ratio = if sum > 0.0 {
                        4.0 * sp * sq / ((sp + sq) * sum)
                    } else {
                        1.0
                    };
                    drift[p] = (phi[p + s] - phi[p]) * inv_2psi * ratio;
                }
            });
        }
        for k in 0..self.c.len() {
            let (c, dc, z) = (&mut self.c[k], &mut self.dc, self.z[k]);
            dc.fill(0.0);
            for a in 0..3 {
                let w = self.d[k] / (grid.h[a] * grid.h[a]);
                let (drift, flux, s) = (&self.drift[a], &mut self.flux[a], stride(grid, a));
                for_each_face_run(grid, a, |p, len| {
                    let (cp, cq) = (&c[p..p + len], &c[p + s..p + s + len]);
                    let (dr, fl) = (&drift[p..p + len], &mut flux[p..p + len]);
                    for i in 0..len {
                        fl[i] = w * ((cp[i] - cq[i]) - z * dr[i] * (cp[i] + cq[i]));
                    }
                });
            }
            add_net_inflow(grid, &self.flux, dc);
            c.iter_mut()
                .zip(dc.iter())
                .for_each(|(c, dc)| *c += dt * dc);
        }
        self.time += dt;
    }
}

impl Simulation for KnpSolver {
    fn step(&mut self, n: usize) {
        let dt = self.cfg.dt / self.substeps as f64;
        for _ in 0..n * self.substeps {
            self.substep(dt);
        }
    }

    fn time_ms(&self) -> f64 {
        self.time
    }
}

#[cfg(test)]
mod tests;
