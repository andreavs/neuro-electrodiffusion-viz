use super::{CsrMatrix, Preconditioner};

/// Banded Cholesky factorisation `A ≈ L Lᵀ` of a symmetric positive (semi-)definite
/// matrix, used as an exact direct solve or as a preconditioner.
///
/// Storage and work scale with the half-bandwidth `w` (largest `|i − j|` of a nonzero):
/// `n·(w+1)` values, `n·w²/2` multiply-adds to factor and `2·n·w` per solve. On a grid
/// stored with `x` fastest the bandwidth is `nx·ny`, so put the longest axis last.
///
/// Pivots that vanish relative to their diagonal entry (a singular matrix such as a
/// pure-Neumann operator) are replaced by that diagonal entry. The result is then the
/// factor of `A` plus a rank-one term, which is still an SPD preconditioner for CG with
/// [`NullSpace::Constant`](super::NullSpace::Constant).
///
/// A typical use is to factor once and keep the factor as a preconditioner while the
/// coefficients drift slowly, refactoring with [`factor`](Self::factor) when CG needs
/// more iterations.
#[derive(Clone, Debug)]
pub struct BandCholesky {
    n: usize,
    w: usize,
    /// Row `i` holds `L[i][i−w..=i]` at `l[i·(w+1)..(i+1)·(w+1)]` (zero-padded on top).
    l: Vec<f64>,
}

/// Relative size below which a pivot counts as zero.
const PIVOT_TOL: f64 = 1e-10;

impl BandCholesky {
    /// Factors `a`, which must be square and symmetric (only the lower triangle is read).
    pub fn new(a: &CsrMatrix) -> Self {
        assert_eq!(a.n_rows(), a.n_cols(), "matrix must be square");
        let n = a.n_rows();
        let w = (0..n)
            .flat_map(|i| a.row(i).map(move |(j, _)| i.abs_diff(j)))
            .max()
            .unwrap_or(0);
        let mut f = Self {
            n,
            w,
            l: vec![0.0; n * (w + 1)],
        };
        f.factor(a);
        f
    }

    /// Size of the factored matrix.
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Half-bandwidth `w`.
    pub fn bandwidth(&self) -> usize {
        self.w
    }

    /// Refactors in place for a matrix with the same size and a bandwidth no larger than
    /// at construction. Does not allocate.
    pub fn factor(&mut self, a: &CsrMatrix) {
        assert_eq!(a.n_rows(), self.n, "matrix size changed");
        let (n, w) = (self.n, self.w);
        let s = w + 1;
        self.l.fill(0.0);
        for i in 0..n {
            for (j, v) in a.row(i) {
                if j <= i {
                    assert!(i - j <= w, "matrix bandwidth grew since construction");
                    self.l[i * s + w + j - i] = v;
                }
            }
        }
        for i in 0..n {
            let lo = i.saturating_sub(w);
            for j in lo..=i {
                // L[i][j] = (A[i][j] − Σ_{p<j} L[i][p] L[j][p]) / L[j][j], p ≥ max(i, j) − w.
                let p0 = lo.max(j.saturating_sub(w));
                let ri = &self.l[i * s + w + p0 - i..i * s + w + j - i];
                let rj = &self.l[j * s + w + p0 - j..j * s + w];
                let dot: f64 = ri.iter().zip(rj).map(|(x, y)| x * y).sum();
                let idx = i * s + w + j - i;
                let v = self.l[idx] - dot;
                self.l[idx] = if j < i {
                    v / self.l[j * s + w]
                } else {
                    let a_ii = a.get(i, i);
                    if v > PIVOT_TOL * a_ii.abs() {
                        v.sqrt()
                    } else {
                        a_ii.abs().max(f64::MIN_POSITIVE).sqrt()
                    }
                };
            }
        }
    }

    /// Solves `L Lᵀ x = b` in place.
    pub fn solve_in_place(&self, x: &mut [f64]) {
        assert_eq!(x.len(), self.n);
        let (n, w) = (self.n, self.w);
        let s = w + 1;
        for i in 0..n {
            let lo = i.saturating_sub(w);
            let row = &self.l[i * s + w + lo - i..i * s + w];
            let dot: f64 = row.iter().zip(&x[lo..i]).map(|(l, y)| l * y).sum();
            x[i] = (x[i] - dot) / self.l[i * s + w];
        }
        for i in (0..n).rev() {
            x[i] /= self.l[i * s + w];
            let xi = x[i];
            let lo = i.saturating_sub(w);
            let row = &self.l[i * s + w + lo - i..i * s + w];
            for (y, l) in x[lo..i].iter_mut().zip(row) {
                *y -= l * xi;
            }
        }
    }
}

impl Preconditioner for BandCholesky {
    fn apply(&mut self, r: &[f64], z: &mut [f64]) {
        z.copy_from_slice(r);
        self.solve_in_place(z);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{
        BoundaryConditions, CgOptions, DiffusionOperator, Grid, LinearOperator, NullSpace, Pcg,
    };

    fn operator(bc: BoundaryConditions) -> DiffusionOperator {
        let grid = Grid::new_3d([4, 3, 6], [0.5, 0.7, 1.0]);
        let sigma: Vec<f64> = (0..grid.len())
            .map(|c| 1.0 + 0.5 * (c as f64 * 0.37).sin())
            .collect();
        DiffusionOperator::new(grid, bc, &sigma)
    }

    #[test]
    fn solves_spd_system_exactly() {
        let op = operator(BoundaryConditions::dirichlet());
        let chol = BandCholesky::new(&op.to_csr());
        assert_eq!(chol.bandwidth(), 12);
        let exact: Vec<f64> = (0..op.len()).map(|i| (i as f64 * 0.21).cos()).collect();
        let mut x = vec![0.0; op.len()];
        op.apply(&exact, &mut x);
        chol.solve_in_place(&mut x);
        for (u, v) in x.iter().zip(&exact) {
            assert!((u - v).abs() < 1e-10, "{u} vs {v}");
        }
    }

    #[test]
    fn preconditions_singular_neumann_system_in_one_step() {
        let op = operator(BoundaryConditions::neumann());
        let mut chol = BandCholesky::new(&op.to_csr());
        let mut b: Vec<f64> = (0..op.len()).map(|i| (i as f64 * 0.9).sin()).collect();
        crate::linalg::remove_mean(&mut b);
        let mut x = vec![0.0; op.len()];
        let opts = CgOptions {
            rtol: 1e-10,
            null_space: NullSpace::Constant,
            ..Default::default()
        };
        let stats = Pcg::new(op.len()).solve(&op, &mut chol, &b, &mut x, &opts);
        assert!(stats.converged && stats.iterations <= 3, "{stats:?}");
    }
}
