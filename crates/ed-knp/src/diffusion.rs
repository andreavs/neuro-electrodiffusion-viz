//! Backward Euler diffusion with sources, one species at a time.

use crate::Model;
use ed_core::linalg::{CgOptions, DiffusionOperator, LinearOperator, Multigrid, NullSpace, Pcg};

/// RMS residual, in mM, at which a solve stops. Solver errors never break conservation: the
/// conservative correction that follows turns any approximate solution into a flux-form
/// update (and on the Jacobi path also shrinks the error by the contraction factor ≲ 0.1).
const TOL: f64 = 1e-8;

/// How a species' system is solved. When `1/(Δt D̃)` dominates the off-diagonal sum (the
/// usual case: diffusion moves far less than a cell per step), plain Jacobi sweeps converge in
/// one or two operator applications; otherwise multigrid-preconditioned CG.
enum Solver {
    /// Inverse diagonal of the operator.
    Jacobi(Vec<f64>),
    Multigrid(Multigrid),
}

/// Solves `(1/(Δt D̃) − ∇²) c* = (cⁿ + Δt f̃)/(Δt D̃)` for every species, with the baseline
/// as Dirichlet value where the boundary is Dirichlet.
pub(crate) struct Diffusion {
    /// Per species; `None` for species that do not diffuse.
    systems: Vec<Option<(DiffusionOperator, Solver)>>,
    dt: f64,
    pcg: Pcg,
    rhs: Vec<f64>,
    residual: Vec<f64>,
}

impl Diffusion {
    pub fn new(m: &Model, dt: f64) -> Self {
        let len = m.grid.len();
        let mut s = Self {
            systems: Vec::new(),
            dt: 0.0,
            pcg: Pcg::new(len),
            rhs: vec![0.0; len],
            residual: vec![0.0; len],
        };
        s.set_dt(m, dt);
        s
    }

    /// Rebuilds the systems for a new time step. Allocates.
    pub fn set_dt(&mut self, m: &Model, dt: f64) {
        let g = m.grid;
        self.dt = dt;
        // Off-diagonal row sum of −∇² (Dirichlet ghosts only add to the diagonal).
        let offdiag: f64 = (0..3)
            .filter(|&a| g.n[a] > 1)
            .map(|a| 2.0 / (g.h[a] * g.h[a]))
            .sum();
        self.systems =
            m.d.iter()
                .map(|&d| {
                    (d > 0.0).then(|| {
                        let shift = 1.0 / (dt * d);
                        let mut op = DiffusionOperator::uniform(g, m.bc, 1.0);
                        op.set_shift(&vec![shift; g.len()]);
                        let solver = if shift >= 10.0 * offdiag {
                            let mut inv = vec![0.0; g.len()];
                            op.diagonal(&mut inv);
                            inv.iter_mut().for_each(|d| *d = 1.0 / *d);
                            Solver::Jacobi(inv)
                        } else {
                            Solver::Multigrid(Multigrid::new(&op))
                        };
                        (op, solver)
                    })
                })
                .collect();
    }

    /// Advances every species `c[k]` by one step with sources `src[k]` (mM/ms).
    pub fn step(&mut self, m: &Model, c: &mut [Vec<f64>], src: &[Vec<f64>]) {
        let (dt, len) = (self.dt, m.grid.len());
        let opts = CgOptions {
            rtol: 1e-8,
            atol: 0.0,
            max_iter: 500,
            null_space: NullSpace::None,
        };
        for (k, c) in c.iter_mut().enumerate() {
            let Some((op, solver)) = &mut self.systems[k] else {
                c.iter_mut().zip(&src[k]).for_each(|(c, s)| *c += dt * s);
                continue;
            };
            let (shift, base) = (1.0 / (dt * m.d[k]), m.baseline[k]);
            let terms = c.iter().zip(&src[k]).zip(&m.boundary_weight);
            for (b, ((c, s), w)) in self.rhs.iter_mut().zip(terms) {
                *b = (c + dt * s) * shift + w * base;
            }
            // Writes r = b − A c into `r` and returns ‖r‖².
            let (b, r) = (&self.rhs, &mut self.residual);
            let residual = |c: &[f64], r: &mut [f64]| {
                op.apply(c, r);
                // Eight partial sums so the reduction vectorises.
                let mut acc = [0.0; 8];
                for (r, b) in r.chunks_mut(8).zip(b.chunks(8)) {
                    for ((r, b), a) in r.iter_mut().zip(b).zip(&mut acc) {
                        *r = b - *r;
                        *a += *r * *r;
                    }
                }
                acc.iter().sum::<f64>()
            };
            // Warm start from cⁿ.
            match solver {
                Solver::Jacobi(inv) => {
                    let tol = (TOL * shift).powi(2) * len as f64;
                    for sweep in 0..=50 {
                        if residual(c, r) <= tol || sweep == 50 {
                            break;
                        }
                        let update = c.iter_mut().zip(r.iter()).zip(inv.iter());
                        update.for_each(|((c, r), d)| *c += r * d);
                    }
                }
                Solver::Multigrid(mg) => {
                    let stats = self.pcg.solve(op, mg, b, c, &opts);
                    debug_assert!(stats.converged, "diffusion solve: {stats:?}");
                    residual(c, r);
                }
            }
            // c* + r/shift equals cⁿ + Δt (f̃ + ∇·D̃∇c*) with conservative face fluxes, so the
            // species is conserved to round-off whatever the solver tolerance.
            c.iter_mut()
                .zip(r.iter())
                .for_each(|(c, r)| *c += r / shift);
        }
    }
}
