//! Explicit Nernst-Planck face fluxes on a cell-centred grid.
//!
//! The outflow per volume of one species through the face from cell `i` to its upper
//! neighbour `j` along axis `a` is
//!
//! ```text
//! D̃ (c_i − c_j) / h_a² − D̃ z (c_i + c_j) g_a,     g_a = σ_op,h / (σ_i + σ_j) · (φ_j − φ_i) / (h_a² ψ)
//! ```
//!
//! where `σ = Σ D̃ z² c / ψ` is the current conductivity and `σ_op,h` the harmonic-mean
//! face conductivity of the potential operator. Summed over species with weights `z`,
//! the drift part is exactly `−σ_op,h (φ_j − φ_i) / h_a²`, the operator's own face flux,
//! so the charge balance of a step matches the potential equation that was solved.
//! Dirichlet faces use a ghost value at `h/2` with the baseline concentration and `φ = 0`.

use ed_core::linalg::Grid;

/// Drift factors `g_a` per axis, stored on the face to the upper neighbour (0 where there
/// is none), and their sum over each cell's Dirichlet faces.
pub(crate) struct DriftFactors {
    faces: [Vec<f64>; 3],
    ghost: Vec<f64>,
}

impl DriftFactors {
    pub fn new(n: usize) -> Self {
        Self {
            faces: [vec![0.0; n], vec![0.0; n], vec![0.0; n]],
            ghost: vec![0.0; n],
        }
    }

    /// Recomputes the factors. `ghost` is `Σ 2/h²` over each cell's Dirichlet faces.
    pub fn update(
        &mut self,
        grid: &Grid,
        psi: f64,
        sigma: &[f64],
        sigma_op: &[f64],
        phi: &[f64],
        ghost: &[f64],
    ) {
        let [nx, ny, _] = grid.n;
        let n = grid.len();
        let stride = [1, nx, nx * ny];
        for (a, g) in self.faces.iter_mut().enumerate() {
            let (s, w) = (stride[a], 1.0 / (grid.h[a] * grid.h[a] * psi));
            let m = n - s;
            for (i, g) in g[..m].iter_mut().enumerate() {
                let (s0, s1) = (sigma[i], sigma[i + s]);
                let (o0, o1) = (sigma_op[i], sigma_op[i + s]);
                *g = if s0 + s1 > 0.0 && o0 + o1 > 0.0 {
                    2.0 * o0 * o1 / ((o0 + o1) * (s0 + s1)) * (phi[i + s] - phi[i]) * w
                } else {
                    0.0
                };
            }
            // Clear the faces that wrap around to the next row or plane.
            match a {
                0 => g.iter_mut().skip(nx - 1).step_by(nx).for_each(|g| *g = 0.0),
                1 => (g.chunks_mut(nx * ny)).for_each(|p| p[nx * (ny - 1)..].fill(0.0)),
                _ => {}
            }
        }
        for ((((g, gh), p), s), o) in (self.ghost.iter_mut())
            .zip(ghost)
            .zip(phi)
            .zip(sigma)
            .zip(sigma_op)
        {
            *g = if *s > 0.0 {
                gh * p * o / (s * psi)
            } else {
                0.0
            };
        }
    }
}

/// Coefficients of one species' face fluxes.
pub(crate) struct FaceFluxes {
    /// D̃ and D̃ z.
    pub d: f64,
    pub dz: f64,
    /// Concentration on Dirichlet faces.
    pub base: f64,
    pub inv_h2: [f64; 3],
}

impl FaceFluxes {
    /// Writes `out = c + Δt (f̃ − ∇·J)`, one x-row at a time.
    #[allow(clippy::too_many_arguments)]
    pub fn advance(
        &self,
        grid: &Grid,
        drift: &DriftFactors,
        ghost: &[f64],
        c: &[f64],
        src: &[f64],
        dt: f64,
        out: &mut [f64],
    ) {
        let [nx, ny, nz] = grid.n;
        let (d, dz, g) = (self.d, self.dz, &drift.faces);
        for k in 0..nz {
            for j in 0..ny {
                let r = nx * (j + ny * k);
                let row = r..r + nx;
                let (acc, cr) = (&mut out[row.clone()], &c[row.clone()]);
                for (((o, &c), gh), p) in (acc.iter_mut())
                    .zip(cr)
                    .zip(&ghost[row.clone()])
                    .zip(&drift.ghost[row])
                {
                    *o = gh * d * (c - self.base) + dz * c * p;
                }
                let (gx, w) = (&g[0][r..r + nx], d * self.inv_h2[0]);
                for i in 0..nx - 1 {
                    let f = w * (cr[i] - cr[i + 1]) - dz * (cr[i] + cr[i + 1]) * gx[i];
                    acc[i] += f;
                    acc[i + 1] -= f;
                }
                for (a, s, has_lo, has_hi) in
                    [(1, nx, j > 0, j + 1 < ny), (2, nx * ny, k > 0, k + 1 < nz)]
                {
                    let w = d * self.inv_h2[a];
                    if has_hi {
                        let (cn, g) = (&c[r + s..r + s + nx], &g[a][r..r + nx]);
                        for (((o, &ci), &cj), &g) in acc.iter_mut().zip(cr).zip(cn).zip(g) {
                            *o += w * (ci - cj) - dz * (ci + cj) * g;
                        }
                    }
                    if has_lo {
                        let (cn, g) = (&c[r - s..r - s + nx], &g[a][r - s..r - s + nx]);
                        for (((o, &ci), &cj), &g) in acc.iter_mut().zip(cr).zip(cn).zip(g) {
                            *o += w * (ci - cj) + dz * (ci + cj) * g;
                        }
                    }
                }
                for ((o, &c), &s) in acc.iter_mut().zip(cr).zip(&src[r..r + nx]) {
                    *o = c + dt * (s - *o);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed_core::linalg::{BoundaryConditions, DiffusionOperator, LinearOperator};

    fn field(n: usize, seed: u64, lo: f64, hi: f64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                lo + (hi - lo) * ((s >> 11) as f64 / (1u64 << 53) as f64)
            })
            .collect()
    }

    /// Two species (z = +1, −1) on a small anisotropic grid with random fields: the
    /// charge outflow `Σ z (c − c_new)/Δt` of one step must equal the operator applied to
    /// φ plus the diffusive charge outflow, for both boundary kinds.
    #[test]
    fn charge_outflow_matches_potential_operator() {
        let grid = Grid::new_3d([5, 4, 3], [1.0, 1.5, 0.7]);
        let n = grid.len();
        let (psi, dt, base) = (25.0, 0.01, [10.0, 12.0]);
        let (d, z) = ([1.3, 2.0], [1.0, -1.0]);
        let c = [field(n, 1, 8.0, 12.0), field(n, 2, 9.0, 14.0)];
        let phi = field(n, 3, -1.0, 1.0);
        let sigma: Vec<f64> = (0..n)
            .map(|i| (d[0] * c[0][i] + d[1] * c[1][i]) / psi)
            .collect();
        // A lagged operator conductivity, as between operator refreshes.
        let sigma_op: Vec<f64> = sigma
            .iter()
            .zip(field(n, 4, 0.9, 1.1))
            .map(|(s, f)| s * f)
            .collect();
        for bc in [
            BoundaryConditions::neumann(),
            BoundaryConditions::dirichlet(),
        ] {
            let laplace = DiffusionOperator::uniform(grid, bc, 1.0);
            let mut ghost = vec![0.0; n];
            laplace.apply(&vec![1.0; n], &mut ghost);
            let mut drift = DriftFactors::new(n);
            drift.update(&grid, psi, &sigma, &sigma_op, &phi, &ghost);

            let mut charge_out = vec![0.0; n];
            let mut expected = vec![0.0; n];
            DiffusionOperator::new(grid, bc, &sigma_op).apply(&phi, &mut expected);
            for k in 0..2 {
                let fx = FaceFluxes {
                    d: d[k],
                    dz: d[k] * z[k],
                    base: base[k],
                    inv_h2: grid.h.map(|h| 1.0 / (h * h)),
                };
                let mut out = vec![0.0; n];
                fx.advance(&grid, &drift, &ghost, &c[k], &vec![0.0; n], dt, &mut out);
                // Diffusive part: D̃ (−Δ)(c − c_baseline).
                let shifted: Vec<f64> = c[k].iter().map(|c| c - base[k]).collect();
                let mut lap = vec![0.0; n];
                laplace.apply(&shifted, &mut lap);
                for i in 0..n {
                    charge_out[i] += z[k] * (c[k][i] - out[i]) / dt;
                    expected[i] += z[k] * d[k] * lap[i];
                }
            }
            for (a, b) in charge_out.iter().zip(&expected) {
                assert!(
                    (a - b).abs() < 1e-9 * (1.0 + b.abs()),
                    "{a} vs {b} ({bc:?})"
                );
            }
        }
    }

    #[test]
    fn no_flux_step_conserves_amount() {
        let grid = Grid::new_3d([6, 5, 4], [1.0, 1.0, 2.0]);
        let n = grid.len();
        let c = field(n, 5, 1.0, 2.0);
        let sigma = field(n, 6, 0.5, 1.0);
        let mut drift = DriftFactors::new(n);
        drift.update(
            &grid,
            25.0,
            &sigma,
            &sigma,
            &field(n, 7, -5.0, 5.0),
            &vec![0.0; n],
        );
        let fx = FaceFluxes {
            d: 0.8,
            dz: 1.6,
            base: 0.0,
            inv_h2: grid.h.map(|h| 1.0 / (h * h)),
        };
        let mut out = vec![0.0; n];
        fx.advance(
            &grid,
            &drift,
            &vec![0.0; n],
            &c,
            &vec![0.0; n],
            0.05,
            &mut out,
        );
        let (before, after): (f64, f64) = (c.iter().sum(), out.iter().sum());
        assert!((before - after).abs() < 1e-12 * before);
        assert!(c.iter().zip(&out).any(|(a, b)| (a - b).abs() > 1e-3));
    }
}
