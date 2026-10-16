//! Sparse linear algebra for the continuum solvers.
//!
//! - [`CsrMatrix`]: compressed sparse row matrices built from triplets.
//! - [`Grid`], [`DiffusionOperator`]: cell-centred finite-volume discretisation of
//!   `-∇·(σ∇u)` on uniform 2-D and 3-D grids, applied matrix-free or assembled into CSR.
//! - [`Pcg`]: preconditioned conjugate gradients with preallocated work vectors, warm
//!   starts and support for the singular pure-Neumann case.
//! - [`Jacobi`] and [`Multigrid`]: preconditioners. The geometric multigrid V-cycle is
//!   the fast option for structured grids. [`TwoLevel`] (aggregation coarse space plus
//!   Gauss-Seidel) is for finite element systems on unstructured or anisotropic meshes.
//! - [`BandCholesky`]: banded direct factorisation, for long thin grids or as a
//!   preconditioner for operators with high-contrast faces that multigrid handles poorly.
//!
//! Sign convention: operators are assembled as `A u = -∇·(σ∇u)`, which is symmetric
//! positive (semi-)definite, so `∇·(σ∇u) = g` is solved as `A u = -g`.

mod band;
mod cg;
mod csr;
mod grid;
mod multigrid;
mod twolevel;

pub use band::BandCholesky;
pub use cg::{CgOptions, CgStats, Identity, Jacobi, NullSpace, Pcg};
pub use csr::CsrMatrix;
pub use grid::{Boundary, BoundaryConditions, DiffusionOperator, Grid, Side};
pub use multigrid::Multigrid;
pub use twolevel::TwoLevel;

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
    a.iter().zip(b).map(|(x, y)| x * y).sum()
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
