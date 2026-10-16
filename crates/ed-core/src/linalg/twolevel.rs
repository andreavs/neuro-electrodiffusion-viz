use super::{BandCholesky, CsrMatrix, LinearOperator, Preconditioner};

/// Two-level aggregation preconditioner for a symmetric positive (semi-)definite CSR matrix.
///
/// Each fine unknown belongs to one aggregate; the coarse space is spanned by the indicator
/// vectors of the aggregates (piecewise-constant prolongation `P`), and `A_c = Pᵀ A P` is
/// factorised with [`BandCholesky`]. One application is a forward Gauss-Seidel sweep, a
/// coarse-grid correction and a backward Gauss-Seidel sweep, which is symmetric and so
/// usable with CG.
///
/// This targets unstructured or anisotropic finite element systems where geometric
/// multigrid does not apply. It works well when the slow error modes are close to constant
/// on each aggregate, e.g. aggregates that are thin slabs across strongly coupled
/// directions, or one aggregate per weakly coupled subdomain. Number the aggregates so
/// that coupled aggregates have nearby indices: the coarse factorisation costs
/// `n_c · b²` for coarse bandwidth `b`.
///
/// A singular `A` (e.g. a pure-Neumann problem) gives a singular `A_c`, whose zero pivot
/// [`BandCholesky`] replaces; the result is still SPD, and CG's null-space projection
/// removes the constant component.
#[derive(Clone, Debug)]
pub struct TwoLevel {
    a: CsrMatrix,
    inv_diag: Vec<f64>,
    aggregate: Vec<u32>,
    coarse_a: CsrMatrix,
    /// Index into `coarse_a`'s values of each stored entry of `a`.
    coarse_pos: Vec<u32>,
    chol: BandCholesky,
    work: Vec<f64>,
    coarse: Vec<f64>,
}

impl TwoLevel {
    /// Builds the preconditioner for `a` with the given aggregate of each unknown.
    ///
    /// # Panics
    /// If `aggregate` has the wrong length or leaves an aggregate index unused.
    pub fn new(a: &CsrMatrix, aggregate: Vec<u32>) -> Self {
        let n = a.n_rows();
        assert_eq!(aggregate.len(), n, "one aggregate per unknown");
        let n_coarse = aggregate.iter().map(|&g| g as usize + 1).max().unwrap_or(0);
        let mut used = vec![false; n_coarse];
        for &g in &aggregate {
            used[g as usize] = true;
        }
        assert!(used.iter().all(|&u| u), "every aggregate needs an unknown");
        let (rp, ci) = (a.row_ptr(), a.col_idx());
        let mut triplets = Vec::with_capacity(a.nnz());
        for r in 0..n {
            for &c in &ci[rp[r]..rp[r + 1]] {
                triplets.push((aggregate[r] as usize, aggregate[c] as usize, 1.0));
            }
        }
        let coarse_a = CsrMatrix::from_triplets(n_coarse, n_coarse, &triplets);
        let coarse_pos = triplets
            .iter()
            .map(|&(i, j, _)| coarse_a.position(i, j).expect("in pattern") as u32)
            .collect();
        let chol = BandCholesky::new(&coarse_a);
        let mut tl = Self {
            a: a.clone(),
            inv_diag: vec![0.0; n],
            aggregate,
            coarse_a,
            coarse_pos,
            chol,
            work: vec![0.0; n],
            coarse: vec![0.0; n_coarse],
        };
        tl.update(a);
        tl
    }

    /// Number of coarse unknowns.
    pub fn coarse_len(&self) -> usize {
        self.coarse.len()
    }

    /// Bandwidth of the coarse matrix.
    pub fn coarse_bandwidth(&self) -> usize {
        self.chol.bandwidth()
    }

    /// Refreshes the preconditioner after the values of `a` changed (same sparsity pattern).
    pub fn update(&mut self, a: &CsrMatrix) {
        assert_eq!(a.nnz(), self.a.nnz(), "sparsity pattern changed");
        self.a.values_mut().copy_from_slice(a.values());
        self.a.diagonal(&mut self.inv_diag);
        for d in &mut self.inv_diag {
            *d = if *d > 0.0 { 1.0 / *d } else { 0.0 };
        }
        let coarse = self.coarse_a.values_mut();
        coarse.fill(0.0);
        for (&p, &v) in self.coarse_pos.iter().zip(self.a.values()) {
            coarse[p as usize] += v;
        }
        self.chol.factor(&self.coarse_a);
    }

    fn gauss_seidel(&self, r: &[f64], z: &mut [f64], forward: bool) {
        let (rp, ci, v) = (self.a.row_ptr(), self.a.col_idx(), self.a.values());
        let n = self.a.n_rows();
        let mut sweep = |i: usize| {
            let (s, e) = (rp[i], rp[i + 1]);
            let ax: f64 = ci[s..e].iter().zip(&v[s..e]).map(|(&c, &a)| a * z[c]).sum();
            z[i] += (r[i] - ax) * self.inv_diag[i];
        };
        if forward {
            for i in 0..n {
                sweep(i);
            }
        } else {
            for i in (0..n).rev() {
                sweep(i);
            }
        }
    }
}

impl Preconditioner for TwoLevel {
    fn apply(&mut self, r: &[f64], z: &mut [f64]) {
        z.fill(0.0);
        self.gauss_seidel(r, z, true);
        let mut work = std::mem::take(&mut self.work);
        self.a.matvec(z, &mut work);
        self.coarse.fill(0.0);
        for (i, (&ri, &azi)) in r.iter().zip(&work).enumerate() {
            self.coarse[self.aggregate[i] as usize] += ri - azi;
        }
        self.work = work;
        self.chol.solve_in_place(&mut self.coarse);
        for (zi, &g) in z.iter_mut().zip(&self.aggregate) {
            *zi += self.coarse[g as usize];
        }
        self.gauss_seidel(r, z, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{dot, CgOptions, Jacobi, NullSpace, Pcg};

    /// Anisotropic 2-D Laplacian (weak coupling `eps` along x, strong along y) on an
    /// `nx × ny` grid with Neumann ends, plus `shift` on the diagonal.
    fn aniso(nx: usize, ny: usize, eps: f64, shift: f64) -> CsrMatrix {
        let id = |i: usize, j: usize| i + nx * j;
        let mut t = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                t.push((id(i, j), id(i, j), shift));
                let mut link = |a: usize, b: usize, w: f64| {
                    t.push((a, a, w));
                    t.push((b, b, w));
                    t.push((a, b, -w));
                    t.push((b, a, -w));
                };
                if i + 1 < nx {
                    link(id(i, j), id(i + 1, j), eps);
                }
                if j + 1 < ny {
                    link(id(i, j), id(i, j + 1), 1.0);
                }
            }
        }
        CsrMatrix::from_triplets(nx * ny, nx * ny, &t)
    }

    fn iterations(a: &CsrMatrix, m: &mut dyn Preconditioner, null: NullSpace) -> usize {
        let n = a.n_rows();
        let mut b: Vec<f64> = (0..n).map(|i| ((i * 7919) % 101) as f64 - 50.0).collect();
        if null == NullSpace::Constant {
            let mean = b.iter().sum::<f64>() / n as f64;
            b.iter_mut().for_each(|v| *v -= mean);
        }
        let mut x = vec![0.0; n];
        let opts = CgOptions {
            rtol: 1e-10,
            max_iter: 5000,
            null_space: null,
            ..Default::default()
        };
        let stats = Pcg::new(n).solve(a, m, &b, &mut x, &opts);
        assert!(stats.converged, "{stats:?}");
        let mut ax = vec![0.0; n];
        a.matvec(&x, &mut ax);
        let res: Vec<f64> = ax.iter().zip(&b).map(|(p, q)| p - q).collect();
        assert!(dot(&res, &res).sqrt() < 1e-8 * dot(&b, &b).sqrt());
        stats.iterations
    }

    #[test]
    fn column_aggregates_fix_anisotropy() {
        let (nx, ny) = (200, 12);
        for (shift, null) in [(1e-3, NullSpace::None), (0.0, NullSpace::Constant)] {
            let a = aniso(nx, ny, 1e-2, shift);
            let agg: Vec<u32> = (0..nx * ny).map(|k| (k % nx) as u32).collect();
            let mut tl = TwoLevel::new(&a, agg);
            assert_eq!((tl.coarse_len(), tl.coarse_bandwidth()), (nx, 1));
            let it_tl = iterations(&a, &mut tl, null);
            let it_j = iterations(&a, &mut Jacobi::new(&a), null);
            assert!(it_tl * 10 < it_j, "two-level {it_tl}, Jacobi {it_j}");
        }
    }

    #[test]
    fn update_tracks_new_values() {
        let a = aniso(30, 4, 0.1, 1.0);
        let agg: Vec<u32> = (0..120).map(|k| (k % 30) as u32 / 3).collect();
        let mut tl = TwoLevel::new(&aniso(30, 4, 0.1, 5.0), agg);
        tl.update(&a);
        let fresh = TwoLevel::new(&a, tl.aggregate.clone());
        assert_eq!(tl.coarse_a, fresh.coarse_a);
        assert!(iterations(&a, &mut tl, NullSpace::None) < 30);
    }
}
