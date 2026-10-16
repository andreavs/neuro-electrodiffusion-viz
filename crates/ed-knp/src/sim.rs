//! The simulator: potential solve and Nernst-Planck transport each step.

use crate::faces::{
    add_nernst_planck, apply_face_fluxes, for_each_boundary_run, for_each_face_run,
};
use crate::{OuterBoundary, Params};
use ed_core::constants::psi_mv;
use ed_core::linalg::{
    BoundaryConditions, CgOptions, CgStats, DiffusionOperator, Grid, Multigrid, NullSpace, Pcg,
};
use ed_core::Simulation;

/// Largest relative change of σ in any cell before the potential operator is rebuilt.
/// Well below the difference between harmonic and arithmetic face means at any
/// resolved gradient, so it adds no error beyond the discretisation's.
const SIGMA_TOLERANCE: f64 = 1e-3;

/// Root-mean-square charge error per cell, mM, that one potential solve may leave.
/// The charge feedback corrects it in the next step, so it does not accumulate.
const CHARGE_TOLERANCE: f64 = 1e-12;

/// Tissue-scale KNP simulator. Fields are flat arrays indexed by
/// [`Grid::index`] (x fastest).
pub struct KnpSim {
    pub(crate) params: Params,
    pub(crate) grid: Grid,
    psi: f64,
    /// Effective diffusion coefficient per species, µm²/ms.
    d: Vec<f64>,
    pub(crate) z: Vec<f64>,
    pub(crate) c: Vec<Vec<f64>>,
    /// Transport rate of the species being updated, mM/ms.
    dc: Vec<f64>,
    /// Source rate per species, mM/ms.
    pub(crate) src: Vec<Vec<f64>>,
    /// Capacitive current as a charge-density rate, mM/ms.
    pub(crate) cap: Vec<f64>,
    phi: Vec<f64>,
    /// φ from the step before, for extrapolating the CG starting guess.
    phi_prev: Vec<f64>,
    sigma: Vec<f64>,
    rhs: Vec<f64>,
    /// `Σ_k z_k D̃_k c_k` per cell.
    u: Vec<f64>,
    /// Charge concentration `Σ_k z_k c_k` that electroneutrality prescribes, mM.
    rho_target: Vec<f64>,
    /// Set when concentrations were edited, so `rho_target` is re-read from them.
    rho_dirty: bool,
    /// Per axis and face: drift factor `(σ_op,face / (σ_p + σ_q)) Δφ / (ψ h²)`.
    drift: [Vec<f64>; 3],
    /// Per-face scratch flux, mM/ms.
    flux: Vec<f64>,
    /// σ the operator was last built with.
    sigma_op: Vec<f64>,
    /// Per axis: harmonic-mean face value of `sigma_op` towards the upper neighbour.
    face_sigma: [Vec<f64>; 3],
    op: DiffusionOperator,
    mg: Multigrid,
    pcg: Pcg,
    cg: CgOptions,
    last_cg: CgStats,
    substeps: usize,
    time: f64,
}

impl KnpSim {
    pub fn new(params: Params) -> Self {
        assert!(params.dt > 0.0, "dt must be positive");
        assert!(params.tortuosity > 0.0 && params.volume_fraction > 0.0);
        assert!(!params.species.is_empty(), "need at least one species");
        let h = [0, 1, 2].map(|a| params.extent[a] / params.cells[a] as f64);
        let grid = Grid::new_3d(params.cells, h);
        let n = grid.len();
        let ns = params.species.len();
        let lambda2 = params.tortuosity * params.tortuosity;
        let d: Vec<f64> = params.species.iter().map(|s| s.ion.d / lambda2).collect();
        let z = params.species.iter().map(|s| s.ion.z as f64).collect();
        let c = params.species.iter().map(|s| vec![s.baseline; n]).collect();
        let bc = match params.boundary {
            OuterBoundary::Dirichlet => BoundaryConditions::dirichlet(),
            OuterBoundary::NoFlux => BoundaryConditions::neumann(),
        };
        let op = DiffusionOperator::uniform(grid, bc, 1.0);
        let mg = Multigrid::new(&op);
        let d_max = d.iter().cloned().fold(0.0, f64::max);
        let inv_h2: f64 = h.iter().map(|x| 1.0 / (x * x)).sum();
        let dt_stable = 0.9 / (2.0 * d_max * inv_h2);
        let substeps = (params.dt / dt_stable).ceil().max(1.0) as usize;
        let cg = CgOptions {
            rtol: params.cg_rtol,
            max_iter: 200,
            null_space: match params.boundary {
                OuterBoundary::Dirichlet => NullSpace::None,
                OuterBoundary::NoFlux => NullSpace::Constant,
            },
            ..CgOptions::default()
        };
        Self {
            psi: psi_mv(params.temperature),
            grid,
            d,
            z,
            c,
            dc: vec![0.0; n],
            src: vec![vec![0.0; n]; ns],
            cap: vec![0.0; n],
            phi: vec![0.0; n],
            phi_prev: vec![0.0; n],
            sigma: vec![0.0; n],
            rhs: vec![0.0; n],
            u: vec![0.0; n],
            rho_target: vec![0.0; n],
            rho_dirty: true,
            drift: [vec![0.0; n], vec![0.0; n], vec![0.0; n]],
            flux: vec![0.0; n],
            sigma_op: vec![-1.0; n],
            face_sigma: [vec![0.0; n], vec![0.0; n], vec![0.0; n]],
            op,
            mg,
            pcg: Pcg::new(n),
            cg,
            last_cg: CgStats {
                iterations: 0,
                residual: 0.0,
                converged: true,
            },
            substeps,
            time: 0.0,
            params,
        }
    }

    pub fn params(&self) -> &Params {
        &self.params
    }

    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    /// Concentration of species `k`, mM.
    pub fn concentration(&self, k: usize) -> &[f64] {
        &self.c[k]
    }

    /// Mutable concentration of species `k`, for setting initial conditions.
    pub fn concentration_mut(&mut self, k: usize) -> &mut [f64] {
        self.rho_dirty = true;
        &mut self.c[k]
    }

    /// Potential from the most recent step, mV.
    pub fn potential(&self) -> &[f64] {
        &self.phi
    }

    /// Statistics of the most recent potential solve.
    pub fn last_solve(&self) -> CgStats {
        self.last_cg
    }

    /// Substeps taken per [`Params::dt`] to stay within the stability limit.
    pub fn substeps(&self) -> usize {
        self.substeps
    }

    /// Total amount of species `k` in the extracellular volume, amol
    /// (mM·µm³), including the volume fraction α.
    pub fn total_amount(&self, k: usize) -> f64 {
        let h = self.grid.h;
        self.c[k].iter().sum::<f64>() * h[0] * h[1] * h[2] * self.params.volume_fraction
    }

    /// Charge concentration `Σ_k z_k c_k` in cell `i`, mM.
    pub fn charge_density(&self, i: usize) -> f64 {
        self.z.iter().zip(&self.c).map(|(z, c)| z * c[i]).sum()
    }

    /// Rebuilds the potential operator and multigrid hierarchy from the current σ.
    fn rebuild_operator(&mut self) {
        let g = self.grid;
        self.op.update_sigma(&self.sigma);
        self.mg.update(&self.op);
        self.sigma_op.copy_from_slice(&self.sigma);
        let sigma = &self.sigma_op;
        for a in 0..3 {
            let s = [1, g.n[0], g.n[0] * g.n[1]][a];
            let face = &mut self.face_sigma[a];
            for_each_face_run(&g, a, |r| {
                for p in r {
                    // Harmonic mean, as in DiffusionOperator.
                    let (sp, sq) = (sigma[p], sigma[p + s]);
                    face[p] = if sp + sq > 0.0 {
                        2.0 * sp * sq / (sp + sq)
                    } else {
                        0.0
                    };
                }
            });
        }
    }

    fn substep(&mut self, dt: f64) {
        let g = self.grid;
        let stride = [1, g.n[0], g.n[0] * g.n[1]];
        let dirichlet = self.params.boundary == OuterBoundary::Dirichlet;
        if self.rho_dirty {
            for i in 0..g.len() {
                self.rho_target[i] = self.charge_density(i);
            }
            self.rho_dirty = false;
        }

        // Per cell: σ, u = Σ z D̃ c (whose Laplacian is the diffusive charge term), and
        // the source and charge-correction parts of the right-hand side.
        self.sigma.fill(0.0);
        self.u.fill(0.0);
        // `flux` is scratch here: it collects the source part of the right-hand side
        // while `rhs` collects the charge density, so (ρ − ρ_target) is formed without
        // cancelling large terms against each other.
        self.rhs.fill(0.0);
        self.flux.copy_from_slice(&self.cap);
        let mut ub = 0.0;
        for k in 0..self.c.len() {
            let (z, zd) = (self.z[k], self.z[k] * self.d[k]);
            let s = zd * z / self.psi;
            ub += zd * self.params.species[k].baseline;
            let it = self.c[k].iter().zip(&self.src[k]);
            let out = self.sigma.iter_mut().zip(&mut self.u).zip(&mut self.rhs);
            for ((((sigma, u), rho), (c, src)), q) in out.zip(it).zip(&mut self.flux) {
                *sigma += s * c;
                *u += zd * c;
                *rho += z * c;
                *q += z * src;
            }
        }
        // Feeding the charge error back keeps the CG tolerance from accumulating.
        let it = self.rho_target.iter().zip(&self.flux);
        for (rhs, (target, q)) in self.rhs.iter_mut().zip(it) {
            *rhs = (*rhs - target) / dt + q;
        }
        let flux = &mut self.flux;
        for a in 0..3 {
            let w = 1.0 / (g.h[a] * g.h[a]);
            let u = &self.u;
            for_each_face_run(&g, a, |r| {
                let (up, uq) = (&u[r.clone()], &u[r.start + stride[a]..r.end + stride[a]]);
                for ((f, p), q) in flux[r].iter_mut().zip(up).zip(uq) {
                    *f = w * (p - q);
                }
            });
            apply_face_fluxes(&g, a, flux, &mut self.rhs);
            if dirichlet {
                for_each_boundary_run(&g, a, |r| {
                    for (rhs, u) in self.rhs[r.clone()].iter_mut().zip(&u[r]) {
                        *rhs -= 2.0 * w * (u - ub);
                    }
                });
            }
        }

        // Potential: A φ = rhs with A = −∇·(σ_op∇·)/F. Rebuilding the operator and its
        // multigrid hierarchy costs more than the rest of the step, so it is only done
        // once σ has drifted from σ_op by more than SIGMA_TOLERANCE in some cell. The
        // drift fluxes below use σ_op's face values, so transport stays consistent
        // with the potential equation either way.
        let stale = self
            .sigma
            .iter()
            .zip(&self.sigma_op)
            .any(|(s, o)| (s - o).abs() > SIGMA_TOLERANCE * o.abs().max(f64::MIN_POSITIVE));
        if stale {
            self.rebuild_operator();
        }
        // Absolute floor: a residual this small moves at most CHARGE_TOLERANCE of charge
        // (2-norm over cells) per step, and round-off in ρ − ρ_target stays far below it.
        self.cg.atol = CHARGE_TOLERANCE * (g.len() as f64).sqrt() / dt;
        // Linear extrapolation in time is a better starting guess than φ alone.
        for (phi, prev) in self.phi.iter_mut().zip(&mut self.phi_prev) {
            let p = *phi;
            *phi = 2.0 * p - *prev;
            *prev = p;
        }
        self.last_cg = self
            .pcg
            .solve(&self.op, &mut self.mg, &self.rhs, &mut self.phi, &self.cg);

        // Drift factor per face, `(σ_op,face / (σ_p + σ_q)) Δφ / (ψ h²)`: scaling each
        // species' arithmetic-mean face concentration by σ_op,face / σ_a,face makes the
        // charge carried equal the operator's face current exactly.
        let (sigma, phi) = (&self.sigma, &self.phi);
        for (a, &s) in stride.iter().enumerate() {
            let f = 1.0 / (self.psi * g.h[a] * g.h[a]);
            let (drift, face) = (&mut self.drift[a], &self.face_sigma[a]);
            for_each_face_run(&g, a, |r| {
                let (sp, sq) = (&sigma[r.clone()], &sigma[r.start + s..r.end + s]);
                let (pp, pq) = (&phi[r.clone()], &phi[r.start + s..r.end + s]);
                let it = drift[r.clone()].iter_mut().zip(&face[r]);
                for (((v, fs), (sp, sq)), (pp, pq)) in
                    it.zip(sp.iter().zip(sq)).zip(pp.iter().zip(pq))
                {
                    let sum = sp + sq;
                    *v = if sum > 0.0 {
                        fs / sum * (pq - pp) * f
                    } else {
                        0.0
                    };
                }
            });
        }

        // Nernst-Planck transport: diffusion plus drift through every face. Each
        // species only depends on its own concentration from here on, so it is
        // advanced right away.
        let (dc, flux) = (&mut self.dc, &mut self.flux);
        for (k, c) in self.c.iter_mut().enumerate() {
            let (d, dz) = (self.d[k], self.d[k] * self.z[k]);
            dc.fill(0.0);
            for a in 0..3 {
                let w = d / (g.h[a] * g.h[a]);
                add_nernst_planck(&g, a, c, &self.drift[a], w, dz, dc, flux);
                if dirichlet {
                    let cb = self.params.species[k].baseline;
                    let wd = 2.0 * dz / (self.psi * g.h[a] * g.h[a]);
                    let sigma_op = &self.sigma_op;
                    for_each_boundary_run(&g, a, |r| {
                        for p in r {
                            // The operator's boundary face uses σ_op of the cell.
                            let ratio = if sigma[p] > 0.0 {
                                sigma_op[p] / sigma[p]
                            } else {
                                0.0
                            };
                            dc[p] -= 2.0 * w * (c[p] - cb) + wd * c[p] * phi[p] * ratio;
                        }
                    });
                }
            }
            for ((c, dc), src) in c.iter_mut().zip(dc.iter()).zip(&self.src[k]) {
                *c += dt * (dc + src);
            }
        }
        for (target, cap) in self.rho_target.iter_mut().zip(&self.cap) {
            *target -= dt * cap;
        }
        self.time += dt;
    }
}

impl Simulation for KnpSim {
    fn step(&mut self, n: usize) {
        let dt = self.params.dt / self.substeps as f64;
        for _ in 0..n * self.substeps {
            self.substep(dt);
        }
    }

    fn time_ms(&self) -> f64 {
        self.time
    }
}
