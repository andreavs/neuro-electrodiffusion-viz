use super::{dot, remove_mean, LinearOperator, Preconditioner};

/// Null space of the system matrix that CG should project out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NullSpace {
    /// The matrix is nonsingular.
    None,
    /// The constants span the null space (pure-Neumann diffusion problems). The right-hand
    /// side is projected onto zero mean and the returned solution has zero mean.
    Constant,
}

/// Stopping criteria for [`Pcg::solve`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CgOptions {
    /// Stop when `‖b − A x‖ ≤ rtol · ‖b‖` …
    pub rtol: f64,
    /// … or when `‖b − A x‖ ≤ atol`.
    pub atol: f64,
    pub max_iter: usize,
    pub null_space: NullSpace,
}

impl Default for CgOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            atol: 0.0,
            max_iter: 1000,
            null_space: NullSpace::None,
        }
    }
}

/// Result of a CG solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CgStats {
    /// Iterations performed (0 if the initial guess already satisfied the tolerance).
    pub iterations: usize,
    /// Final relative residual `‖b − A x‖ / ‖b‖` (absolute if `b = 0`).
    pub residual: f64,
    pub converged: bool,
}

/// Preconditioned conjugate gradient solver with preallocated work vectors.
///
/// Create one per system size and reuse it every time step: [`solve`](Self::solve) does
/// not allocate. The initial contents of `x` are the starting guess (warm start).
#[derive(Clone, Debug)]
pub struct Pcg {
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    q: Vec<f64>,
}

impl Pcg {
    pub fn new(n: usize) -> Self {
        Self {
            r: vec![0.0; n],
            z: vec![0.0; n],
            p: vec![0.0; n],
            q: vec![0.0; n],
        }
    }

    pub fn len(&self) -> usize {
        self.r.len()
    }

    pub fn is_empty(&self) -> bool {
        self.r.is_empty()
    }

    /// Solves `A x = b` for symmetric positive (semi-)definite `A`, starting from `x`.
    pub fn solve<A, M>(
        &mut self,
        a: &A,
        m: &mut M,
        b: &[f64],
        x: &mut [f64],
        opts: &CgOptions,
    ) -> CgStats
    where
        A: LinearOperator + ?Sized,
        M: Preconditioner + ?Sized,
    {
        let n = self.len();
        assert_eq!(a.len(), n, "operator size does not match the solver");
        assert_eq!(b.len(), n);
        assert_eq!(x.len(), n);
        let project = opts.null_space == NullSpace::Constant;
        let Self { r, z, p, q } = self;

        // ‖b‖ of the projected right-hand side, computed without copying b.
        let b_norm = if project {
            let mean = b.iter().sum::<f64>() / n.max(1) as f64;
            b.iter()
                .map(|v| (v - mean) * (v - mean))
                .sum::<f64>()
                .sqrt()
        } else {
            dot(b, b).sqrt()
        };
        if b_norm == 0.0 {
            x.fill(0.0);
            return CgStats {
                iterations: 0,
                residual: 0.0,
                converged: true,
            };
        }
        let tol = (opts.rtol * b_norm).max(opts.atol);

        if project {
            remove_mean(x);
        }
        a.apply(x, r);
        r.iter_mut().zip(b).for_each(|(ri, bi)| *ri = bi - *ri);
        if project {
            remove_mean(r);
        }
        let mut r_norm = dot(r, r).sqrt();
        if r_norm <= tol {
            return CgStats {
                iterations: 0,
                residual: r_norm / b_norm,
                converged: true,
            };
        }

        m.apply(r, z);
        if project {
            remove_mean(z);
        }
        p.copy_from_slice(z);
        let mut rz = dot(r, z);

        for it in 1..=opts.max_iter {
            a.apply(p, q);
            let pq = dot(p, q);
            if pq <= 0.0 || !pq.is_finite() {
                // Breakdown: the operator or preconditioner is not SPD on this subspace.
                return CgStats {
                    iterations: it - 1,
                    residual: r_norm / b_norm,
                    converged: false,
                };
            }
            let alpha = rz / pq;
            let mut rr = 0.0;
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * q[i];
                rr += r[i] * r[i];
            }
            r_norm = rr.sqrt();
            if r_norm <= tol {
                if project {
                    remove_mean(x);
                }
                return CgStats {
                    iterations: it,
                    residual: r_norm / b_norm,
                    converged: true,
                };
            }
            m.apply(r, z);
            if project {
                remove_mean(z);
            }
            let rz_new = dot(r, z);
            let beta = rz_new / rz;
            rz = rz_new;
            p.iter_mut()
                .zip(z.iter())
                .for_each(|(pi, zi)| *pi = zi + beta * *pi);
        }
        if project {
            remove_mean(x);
        }
        CgStats {
            iterations: opts.max_iter,
            residual: r_norm / b_norm,
            converged: false,
        }
    }
}

/// No preconditioning: `z = r`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Identity;

impl Preconditioner for Identity {
    fn apply(&mut self, r: &[f64], z: &mut [f64]) {
        z.copy_from_slice(r);
    }
}

/// Jacobi (diagonal) preconditioner.
#[derive(Clone, Debug)]
pub struct Jacobi {
    inv_diag: Vec<f64>,
}

impl Jacobi {
    pub fn new<A: LinearOperator + ?Sized>(a: &A) -> Self {
        let mut j = Self {
            inv_diag: vec![0.0; a.len()],
        };
        j.update(a);
        j
    }

    /// Recomputes the diagonal after the operator changed, without reallocating.
    pub fn update<A: LinearOperator + ?Sized>(&mut self, a: &A) {
        a.diagonal(&mut self.inv_diag);
        for d in &mut self.inv_diag {
            *d = if *d != 0.0 { 1.0 / *d } else { 1.0 };
        }
    }
}

impl Preconditioner for Jacobi {
    fn apply(&mut self, r: &[f64], z: &mut [f64]) {
        for ((zi, ri), di) in z.iter_mut().zip(r).zip(&self.inv_diag) {
            *zi = ri * di;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::CsrMatrix;

    /// Dense SPD matrix `Bᵀ B + n I` with a deterministic pseudo-random `B`.
    fn spd_matrix(n: usize) -> Vec<Vec<f64>> {
        let mut s: u64 = 42;
        let mut rand = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let b: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| rand()).collect()).collect();
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let bb: f64 = (0..n).map(|k| b[k][i] * b[k][j]).sum();
                        bb + if i == j { 0.1 * n as f64 } else { 0.0 }
                    })
                    .collect()
            })
            .collect()
    }

    /// Gaussian elimination with partial pivoting.
    fn dense_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut m: Vec<Vec<f64>> = a.to_vec();
        let mut x = b.to_vec();
        for col in 0..n {
            let piv = (col..n)
                .max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))
                .unwrap();
            m.swap(col, piv);
            x.swap(col, piv);
            let (top, rest) = m.split_at_mut(col + 1);
            let pivot_row = &top[col];
            for (r, row) in rest.iter_mut().enumerate() {
                let f = row[col] / pivot_row[col];
                for (v, p) in row[col..].iter_mut().zip(&pivot_row[col..]) {
                    *v -= f * p;
                }
                x[col + 1 + r] -= f * x[col];
            }
        }
        for row in (0..n).rev() {
            let s: f64 = (row + 1..n).map(|k| m[row][k] * x[k]).sum();
            x[row] = (x[row] - s) / m[row][row];
        }
        x
    }

    #[test]
    fn cg_matches_direct_solve() {
        let n = 30;
        let dense = spd_matrix(n);
        let mut t = Vec::new();
        for (i, row) in dense.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                t.push((i, j, v));
            }
        }
        let a = CsrMatrix::from_triplets(n, n, &t);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let exact = dense_solve(&dense, &b);

        let opts = CgOptions {
            rtol: 1e-12,
            ..Default::default()
        };
        let mut cg = Pcg::new(n);
        for jacobi in [false, true] {
            let mut x = vec![0.0; n];
            let stats = if jacobi {
                cg.solve(&a, &mut Jacobi::new(&a), &b, &mut x, &opts)
            } else {
                cg.solve(&a, &mut Identity, &b, &mut x, &opts)
            };
            assert!(stats.converged, "{stats:?}");
            for (u, v) in x.iter().zip(&exact) {
                assert!((u - v).abs() < 1e-9, "{u} vs {v}");
            }
        }
    }

    #[test]
    fn warm_start_at_solution_takes_no_iterations() {
        let a =
            CsrMatrix::from_triplets(2, 2, &[(0, 0, 2.0), (1, 1, 4.0), (0, 1, 1.0), (1, 0, 1.0)]);
        let b = [3.0, 5.0];
        let mut x = [1.0, 1.0];
        let stats = Pcg::new(2).solve(&a, &mut Identity, &b, &mut x, &CgOptions::default());
        assert_eq!(stats.iterations, 0);
        assert!(stats.converged);
    }

    #[test]
    fn zero_rhs_gives_zero_solution() {
        let a = CsrMatrix::from_triplets(2, 2, &[(0, 0, 1.0), (1, 1, 1.0)]);
        let mut x = [5.0, -2.0];
        let stats = Pcg::new(2).solve(
            &a,
            &mut Identity,
            &[0.0, 0.0],
            &mut x,
            &CgOptions::default(),
        );
        assert_eq!(x, [0.0, 0.0]);
        assert!(stats.converged);
    }

    #[test]
    fn reports_non_convergence() {
        let n = 30;
        let dense = spd_matrix(n);
        let t: Vec<_> = dense
            .iter()
            .enumerate()
            .flat_map(|(i, row)| row.iter().enumerate().map(move |(j, &v)| (i, j, v)))
            .collect();
        let a = CsrMatrix::from_triplets(n, n, &t);
        let b = vec![1.0; n];
        let mut x = vec![0.0; n];
        let opts = CgOptions {
            rtol: 1e-14,
            max_iter: 2,
            ..Default::default()
        };
        let stats = Pcg::new(n).solve(&a, &mut Identity, &b, &mut x, &opts);
        assert_eq!(stats.iterations, 2);
        assert!(!stats.converged);
    }
}
