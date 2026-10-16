//! Sparse linear algebra for the continuum solvers.
//!
//! - [`CsrMatrix`]: compressed sparse row matrices built from triplets.
//! - [`Grid`], [`DiffusionOperator`]: cell-centred finite-volume discretisation of
//!   `-∇·(σ∇u)` on uniform 2-D and 3-D grids, applied matrix-free or assembled into CSR.
//! - [`Pcg`]: preconditioned conjugate gradients with preallocated work vectors, warm
//!   starts and support for the singular pure-Neumann case.
//! - [`Jacobi`] and [`Multigrid`]: preconditioners. The geometric multigrid V-cycle is
//!   the fast option for structured grids.
//!
//! Sign convention: operators are assembled as `A u = -∇·(σ∇u)`, which is symmetric
//! positive (semi-)definite, so `∇·(σ∇u) = g` is solved as `A u = -g`.

mod cg;
mod csr;
mod grid;
mod multigrid;

pub use cg::{CgOptions, CgStats, Identity, Jacobi, NullSpace, Pcg};
pub use csr::CsrMatrix;
pub use grid::{Boundary, BoundaryConditions, DiffusionOperator, Face, Grid, Side};
pub use multigrid::Multigrid;

/// A square linear operator `y = A x`.
pub trait LinearOperator {
    /// Number of rows (and columns).
    fn len(&self) -> usize;
    /// Whether the operator has no rows.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Computes `y = A x`. Both slices have length [`len`](Self::len).
    fn apply(&self, x: &[f64], y: &mut [f64]);
    /// Writes the diagonal of `A` into `out`.
    fn diagonal(&self, out: &mut [f64]);
}

/// A preconditioner `z ≈ A⁻¹ r`. It must be symmetric positive definite for use with CG.
pub trait Preconditioner {
    /// Computes `z = M⁻¹ r`.
    fn apply(&mut self, r: &[f64], z: &mut [f64]);
}

/// Euclidean inner product.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    // Independent partial sums let the additions pipeline and vectorise.
    let mut acc = [0.0; 4];
    let (ca, cb) = (a.chunks_exact(4), b.chunks_exact(4));
    let tail: f64 = ca
        .remainder()
        .iter()
        .zip(cb.remainder())
        .map(|(x, y)| x * y)
        .sum();
    for (x, y) in ca.zip(cb) {
        for l in 0..4 {
            acc[l] += x[l] * y[l];
        }
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

/// Euclidean norm.
pub fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Subtracts the arithmetic mean from `v`.
pub fn remove_mean(v: &mut [f64]) {
    if v.is_empty() {
        return;
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    v.iter_mut().for_each(|x| *x -= mean);
}
