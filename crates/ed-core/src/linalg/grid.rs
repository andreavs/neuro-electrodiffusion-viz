use super::{CsrMatrix, LinearOperator};

/// A uniform cell-centred grid with its lower corner at the origin.
///
/// Cells are stored with `i` (x) fastest: `index = i + nx * (j + ny * k)`.
/// A 2-D grid has `nz == 1` and no z-faces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    /// Cells per axis.
    pub n: [usize; 3],
    /// Cell size per axis (µm).
    pub h: [f64; 3],
    /// Number of active axes: 2 or 3.
    pub dim: usize,
}

impl Grid {
    /// A 3-D grid of `n[0] × n[1] × n[2]` cells of size `h`.
    pub fn new_3d(n: [usize; 3], h: [f64; 3]) -> Self {
        assert!(
            n.iter().all(|&m| m > 0),
            "grid needs at least one cell per axis"
        );
        assert!(h.iter().all(|&d| d > 0.0), "cell sizes must be positive");
        Self { n, h, dim: 3 }
    }

    /// A 2-D grid of `nx × ny` cells of size `hx × hy`.
    pub fn new_2d(nx: usize, ny: usize, hx: f64, hy: f64) -> Self {
        let mut g = Self::new_3d([nx, ny, 1], [hx, hy, 1.0]);
        g.dim = 2;
        g
    }

    /// Total number of cells.
    pub fn len(&self) -> usize {
        self.n[0] * self.n[1] * self.n[2]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Linear index of cell `(i, j, k)`.
    #[inline]
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.n[0] * (j + self.n[1] * k)
    }

    /// Centre of cell `(i, j, k)`. The z-coordinate is 0 on 2-D grids.
    pub fn cell_center(&self, i: usize, j: usize, k: usize) -> [f64; 3] {
        let z = if self.dim == 2 {
            0.0
        } else {
            (k as f64 + 0.5) * self.h[2]
        };
        [
            (i as f64 + 0.5) * self.h[0],
            (j as f64 + 0.5) * self.h[1],
            z,
        ]
    }

    /// Physical extent of the domain along each axis.
    pub fn extent(&self) -> [f64; 3] {
        [
            self.n[0] as f64 * self.h[0],
            self.n[1] as f64 * self.h[1],
            self.n[2] as f64 * self.h[2],
        ]
    }

    /// Calls `f(i, j, k, index)` for every cell in storage order.
    pub fn for_each_cell(&self, mut f: impl FnMut(usize, usize, usize, usize)) {
        let mut c = 0;
        for k in 0..self.n[2] {
            for j in 0..self.n[1] {
                for i in 0..self.n[0] {
                    f(i, j, k, c);
                    c += 1;
                }
            }
        }
    }
}

/// Boundary condition kind on one face of the domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    /// Zero normal flux.
    Neumann,
    /// Prescribed value on the face; values enter through
    /// [`DiffusionOperator::add_dirichlet_rhs`].
    Dirichlet,
}

/// Which end of an axis a face is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Lower,
    Upper,
}

/// Boundary kinds for the lower and upper face of each axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundaryConditions {
    pub lower: [Boundary; 3],
    pub upper: [Boundary; 3],
}

impl BoundaryConditions {
    /// The same kind on every face.
    pub fn uniform(b: Boundary) -> Self {
        Self {
            lower: [b; 3],
            upper: [b; 3],
        }
    }

    pub fn neumann() -> Self {
        Self::uniform(Boundary::Neumann)
    }

    pub fn dirichlet() -> Self {
        Self::uniform(Boundary::Dirichlet)
    }

    /// The kind on `axis`, `side`.
    pub fn get(&self, axis: usize, side: Side) -> Boundary {
        match side {
            Side::Lower => self.lower[axis],
            Side::Upper => self.upper[axis],
        }
    }

    /// True when every active face of `grid` is Neumann, so the operator is singular
    /// with the constants as its null space.
    pub fn is_pure_neumann(&self, grid: &Grid) -> bool {
        (0..grid.dim)
            .all(|a| self.lower[a] == Boundary::Neumann && self.upper[a] == Boundary::Neumann)
    }
}

/// Cell-centred finite-volume discretisation of `A u = -∇·(σ∇u)` on a [`Grid`].
///
/// `σ` is given per cell; face coefficients use the harmonic mean of the two neighbouring
/// cells. Rows are scaled by the cell volume, so `A u` approximates `-∇·(σ∇u)` pointwise.
/// Dirichlet faces use a ghost value at distance `h/2`. The operator is symmetric and,
/// unless every face is Neumann, positive definite.
#[derive(Clone, Debug)]
pub struct DiffusionOperator {
    grid: Grid,
    bc: BoundaryConditions,
    pub(super) sigma: Vec<f64>,
    /// `c[a][cell]` couples `cell` to its upper neighbour along axis `a` (0 on the last layer).
    c: [Vec<f64>; 3],
    diag: Vec<f64>,
    /// `1 / diag`, or 0 for rows with no coupling at all.
    inv_diag: Vec<f64>,
    /// `nx` zeros, standing in for missing neighbour rows.
    zero_row: Vec<f64>,
}

impl DiffusionOperator {
    /// Builds the operator for conductivity/diffusivity `sigma` (one value per cell, ≥ 0).
    pub fn new(grid: Grid, bc: BoundaryConditions, sigma: &[f64]) -> Self {
        assert_eq!(
            sigma.len(),
            grid.len(),
            "sigma must have one value per cell"
        );
        let n = grid.len();
        let mut op = Self {
            grid,
            bc,
            sigma: sigma.to_vec(),
            c: [vec![0.0; n], vec![0.0; n], vec![0.0; n]],
            diag: vec![0.0; n],
            inv_diag: vec![0.0; n],
            zero_row: vec![0.0; grid.n[0]],
        };
        op.recompute();
        op
    }

    /// Constant coefficient `sigma` everywhere.
    pub fn uniform(grid: Grid, bc: BoundaryConditions, sigma: f64) -> Self {
        Self::new(grid, bc, &vec![sigma; grid.len()])
    }

    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    pub fn boundary_conditions(&self) -> &BoundaryConditions {
        &self.bc
    }

    pub fn sigma(&self) -> &[f64] {
        &self.sigma
    }

    /// Replaces the coefficients without reallocating.
    pub fn update_sigma(&mut self, sigma: &[f64]) {
        assert_eq!(sigma.len(), self.sigma.len());
        self.sigma.copy_from_slice(sigma);
        self.recompute();
    }

    /// Rebuilds face coefficients and the diagonal from `self.sigma`.
    pub(super) fn recompute(&mut self) {
        let g = self.grid;
        let stride = [1, g.n[0], g.n[0] * g.n[1]];
        let sigma = &self.sigma;
        self.diag.fill(0.0);
        for a in 0..3 {
            self.c[a].fill(0.0);
            if a >= g.dim {
                continue;
            }
            let inv_h2 = 1.0 / (g.h[a] * g.h[a]);
            let (c, diag) = (&mut self.c[a], &mut self.diag);
            let lower_dirichlet = self.bc.lower[a] == Boundary::Dirichlet;
            let upper_dirichlet = self.bc.upper[a] == Boundary::Dirichlet;
            g.for_each_cell(|i, j, k, cell| {
                let pos = [i, j, k][a];
                if pos + 1 < g.n[a] {
                    let nb = cell + stride[a];
                    let w = harmonic_mean(sigma[cell], sigma[nb]) * inv_h2;
                    c[cell] = w;
                    diag[cell] += w;
                    diag[nb] += w;
                } else if upper_dirichlet {
                    diag[cell] += 2.0 * sigma[cell] * inv_h2;
                }
                if pos == 0 && lower_dirichlet {
                    diag[cell] += 2.0 * sigma[cell] * inv_h2;
                }
            });
        }
        for (inv, &d) in self.inv_diag.iter_mut().zip(&self.diag) {
            *inv = if d != 0.0 { 1.0 / d } else { 0.0 };
        }
    }

    /// Adds the Dirichlet boundary contributions to a right-hand side `b`.
    ///
    /// `g(axis, side, point)` returns the boundary value at the centre `point` of each
    /// Dirichlet face. Call this after filling `b` with the source term.
    pub fn add_dirichlet_rhs(
        &self,
        b: &mut [f64],
        mut g: impl FnMut(usize, Side, [f64; 3]) -> f64,
    ) {
        assert_eq!(b.len(), self.grid.len());
        let grid = self.grid;
        let ext = grid.extent();
        for a in 0..grid.dim {
            let inv_h2 = 1.0 / (grid.h[a] * grid.h[a]);
            for side in [Side::Lower, Side::Upper] {
                if self.bc.get(a, side) != Boundary::Dirichlet {
                    continue;
                }
                let layer = match side {
                    Side::Lower => 0,
                    Side::Upper => grid.n[a] - 1,
                };
                grid.for_each_cell(|i, j, k, cell| {
                    if [i, j, k][a] != layer {
                        return;
                    }
                    let mut p = grid.cell_center(i, j, k);
                    p[a] = match side {
                        Side::Lower => 0.0,
                        Side::Upper => ext[a],
                    };
                    b[cell] += 2.0 * self.sigma[cell] * inv_h2 * g(a, side, p);
                });
            }
        }
    }

    /// Writes `out[i] = Σ_nb w_nb x_nb` over the y- and z-neighbours of the x-row
    /// `(·, j, k)` that starts at cell `row`.
    #[inline(always)]
    fn transverse_row(&self, x: &[f64], j: usize, k: usize, row: usize, out: &mut [f64]) {
        let [nx, ny, nz] = self.grid.n;
        let out = &mut out[..nx];
        out.fill(0.0);
        let mut add = |w: &[f64], xn: &[f64]| {
            for ((o, w), xn) in out.iter_mut().zip(w).zip(xn) {
                *o += w * xn;
            }
        };
        let (sy, sz) = (nx, nx * ny);
        let [_, cy, cz] = &self.c;
        if j > 0 {
            add(&cy[row - sy..row - sy + nx], &x[row - sy..row - sy + nx]);
        }
        if j + 1 < ny {
            add(&cy[row..row + nx], &x[row + sy..row + sy + nx]);
        }
        if k > 0 {
            add(&cz[row - sz..row - sz + nx], &x[row - sz..row - sz + nx]);
        }
        if k + 1 < nz {
            add(&cz[row..row + nx], &x[row + sz..row + sz + nx]);
        }
    }

    /// Writes `out[i] = Σ_nb w_nb x_nb` over all neighbours of the x-row at `row`.
    #[inline(always)]
    fn offdiag_row(&self, x: &[f64], j: usize, k: usize, row: usize, out: &mut [f64]) {
        let nx = self.grid.n[0];
        self.transverse_row(x, j, k, row, out);
        if nx > 1 {
            let cx = &self.c[0][row..row + nx - 1];
            let xr = &x[row..row + nx];
            for ((o, w), xn) in out[..nx - 1].iter_mut().zip(cx).zip(&xr[1..]) {
                *o += w * xn;
            }
            for ((o, w), xn) in out[1..nx].iter_mut().zip(cx).zip(&xr[..nx - 1]) {
                *o += w * xn;
            }
        }
    }

    /// Calls `f(j, k, row_start)` for every x-row.
    #[inline(always)]
    fn for_each_row(&self, mut f: impl FnMut(usize, usize, usize)) {
        let [nx, ny, nz] = self.grid.n;
        for k in 0..nz {
            for j in 0..ny {
                f(j, k, nx * (j + ny * k));
            }
        }
    }

    /// One red-black Gauss-Seidel sweep for `A x = b`: red cells (`i + j + k` even) then
    /// black cells when `forward`, the reverse order otherwise, so a forward sweep followed
    /// by a backward sweep is symmetric.
    pub(super) fn gauss_seidel(&self, b: &[f64], x: &mut [f64], forward: bool) {
        let colors = if forward { [0, 1] } else { [1, 0] };
        for color in colors {
            self.relax_color(b, x, color);
        }
    }

    /// Same as a forward [`gauss_seidel`](Self::gauss_seidel) sweep starting from `x = 0`,
    /// but skips the neighbour sums of the red half-sweep, which are all zero.
    pub(super) fn gauss_seidel_from_zero(&self, b: &[f64], x: &mut [f64]) {
        let [nx, ny, nz] = self.grid.n;
        for k in 0..nz {
            for j in 0..ny {
                let row = nx * (j + ny * k);
                let xr = &mut x[row..row + nx];
                let (b, d) = (&b[row..row + nx], &self.inv_diag[row..row + nx]);
                let red = (j + k) % 2;
                for (i, ((x, b), d)) in xr.iter_mut().zip(b).zip(d).enumerate() {
                    *x = if i % 2 == red { b * d } else { 0.0 };
                }
            }
        }
        self.relax_color(b, x, 1);
    }

    /// Gauss-Seidel update of every cell with `(i + j + k) % 2 == color`. Those cells only
    /// couple to the other colour, so the update order within the colour does not matter.
    fn relax_color(&self, b: &[f64], x: &mut [f64], color: usize) {
        let [nx, ny, nz] = self.grid.n;
        let (sy, sz) = (nx, nx * ny);
        let [cx, cy, cz] = &self.c;
        let zeros = &self.zero_row[..];
        for k in 0..nz {
            for j in 0..ny {
                let row = nx * (j + ny * k);
                // Split x so the row being relaxed is mutable while neighbour rows are read.
                let (below, rest) = x.split_at_mut(row);
                let (xr, above) = rest.split_at_mut(nx);
                let (xym, wym) = if j > 0 {
                    (
                        &below[row - sy..row - sy + nx],
                        &cy[row - sy..row - sy + nx],
                    )
                } else {
                    (zeros, zeros)
                };
                let (xyp, wyp) = if j + 1 < ny {
                    (&above[sy - nx..sy], &cy[row..row + nx])
                } else {
                    (zeros, zeros)
                };
                let (xzm, wzm) = if k > 0 {
                    (
                        &below[row - sz..row - sz + nx],
                        &cz[row - sz..row - sz + nx],
                    )
                } else {
                    (zeros, zeros)
                };
                let (xzp, wzp) = if k + 1 < nz {
                    (&above[sz - nx..sz], &cz[row..row + nx])
                } else {
                    (zeros, zeros)
                };
                let (b, d, wx) = (
                    &b[row..row + nx],
                    &self.inv_diag[row..row + nx],
                    &cx[row..row + nx],
                );
                let mut i = (j + k + color) % 2;
                while i < nx {
                    let mut v = b[i]
                        + wym[i] * xym[i]
                        + wyp[i] * xyp[i]
                        + wzm[i] * xzm[i]
                        + wzp[i] * xzp[i];
                    if i > 0 {
                        v += wx[i - 1] * xr[i - 1];
                    }
                    if i + 1 < nx {
                        v += wx[i] * xr[i + 1];
                    }
                    xr[i] = v * d[i];
                    i += 2;
                }
            }
        }
    }

    /// `r = b − A x`.
    pub(super) fn residual(&self, b: &[f64], x: &[f64], r: &mut [f64]) {
        let nx = self.grid.n[0];
        self.for_each_row(|j, k, row| {
            let out = &mut r[row..row + nx];
            self.offdiag_row(x, j, k, row, out);
            let (b, d, x) = (
                &b[row..row + nx],
                &self.diag[row..row + nx],
                &x[row..row + nx],
            );
            for (((o, b), d), x) in out.iter_mut().zip(b).zip(d).zip(x) {
                *o += b - d * x;
            }
        });
    }

    /// Assembles the operator into a CSR matrix.
    pub fn to_csr(&self) -> CsrMatrix {
        let g = self.grid;
        let stride = [1, g.n[0], g.n[0] * g.n[1]];
        let mut t = Vec::with_capacity(g.len() * (1 + 2 * g.dim));
        for cell in 0..g.len() {
            t.push((cell, cell, self.diag[cell]));
        }
        for a in 0..g.dim {
            g.for_each_cell(|i, j, k, cell| {
                if [i, j, k][a] + 1 < g.n[a] {
                    let w = self.c[a][cell];
                    t.push((cell, cell + stride[a], -w));
                    t.push((cell + stride[a], cell, -w));
                }
            });
        }
        CsrMatrix::from_triplets(g.len(), g.len(), &t)
    }
}

impl LinearOperator for DiffusionOperator {
    fn len(&self) -> usize {
        self.grid.len()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.len());
        assert_eq!(y.len(), self.len());
        let nx = self.grid.n[0];
        self.for_each_row(|j, k, row| {
            let out = &mut y[row..row + nx];
            self.offdiag_row(x, j, k, row, out);
            let (d, x) = (&self.diag[row..row + nx], &x[row..row + nx]);
            for ((o, d), x) in out.iter_mut().zip(d).zip(x) {
                *o = d * x - *o;
            }
        });
    }

    fn diagonal(&self, out: &mut [f64]) {
        out.copy_from_slice(&self.diag);
    }
}

fn harmonic_mean(a: f64, b: f64) -> f64 {
    if a + b > 0.0 {
        2.0 * a * b / (a + b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    }

    #[test]
    fn matrix_free_matches_csr_and_is_symmetric() {
        let grid = Grid::new_3d([5, 4, 3], [0.5, 0.7, 1.1]);
        let bc = BoundaryConditions {
            lower: [Boundary::Dirichlet, Boundary::Neumann, Boundary::Dirichlet],
            upper: [Boundary::Neumann, Boundary::Dirichlet, Boundary::Neumann],
        };
        let sigma: Vec<f64> = pseudo_random(grid.len(), 1)
            .iter()
            .map(|v| 1.0 + v)
            .collect();
        let op = DiffusionOperator::new(grid, bc, &sigma);
        let a = op.to_csr();
        let x = pseudo_random(grid.len(), 2);
        let (mut y1, mut y2) = (vec![0.0; grid.len()], vec![0.0; grid.len()]);
        op.apply(&x, &mut y1);
        a.matvec(&x, &mut y2);
        for (u, v) in y1.iter().zip(&y2) {
            assert!((u - v).abs() < 1e-12);
        }
        for r in 0..a.n_rows() {
            for (c, v) in a.row(r) {
                assert_eq!(v, a.get(c, r));
            }
        }
    }

    #[test]
    fn neumann_operator_annihilates_constants() {
        let grid = Grid::new_2d(6, 5, 0.3, 0.2);
        let sigma = pseudo_random(grid.len(), 3)
            .iter()
            .map(|v| 2.0 + v)
            .collect::<Vec<_>>();
        let op = DiffusionOperator::new(grid, BoundaryConditions::neumann(), &sigma);
        let mut y = vec![1.0; grid.len()];
        op.apply(&vec![3.0; grid.len()], &mut y);
        assert!(y.iter().all(|v| v.abs() < 1e-12));
        assert!(BoundaryConditions::neumann().is_pure_neumann(&grid));
    }

    #[test]
    fn red_black_sweep_from_zero_matches_general_sweep() {
        let grid = Grid::new_3d([7, 6, 5], [1.0, 0.8, 1.3]);
        let sigma: Vec<f64> = pseudo_random(grid.len(), 4)
            .iter()
            .map(|v| 1.0 + v)
            .collect();
        let op = DiffusionOperator::new(grid, BoundaryConditions::dirichlet(), &sigma);
        let b = pseudo_random(grid.len(), 5);
        let (mut x1, mut x2) = (vec![0.0; grid.len()], vec![7.0; grid.len()]);
        op.gauss_seidel(&b, &mut x1, true);
        op.gauss_seidel_from_zero(&b, &mut x2);
        assert_eq!(x1, x2);
    }

    #[test]
    fn gauss_seidel_converges_to_solution() {
        let grid = Grid::new_2d(6, 5, 1.0, 1.0);
        let op = DiffusionOperator::uniform(grid, BoundaryConditions::dirichlet(), 1.0);
        let exact = pseudo_random(grid.len(), 6);
        let mut b = vec![0.0; grid.len()];
        op.apply(&exact, &mut b);
        let mut x = vec![0.0; grid.len()];
        for _ in 0..200 {
            op.gauss_seidel(&b, &mut x, true);
        }
        for (u, v) in x.iter().zip(&exact) {
            assert!((u - v).abs() < 1e-10);
        }
    }

    #[test]
    fn two_d_grid_has_no_z_coupling() {
        let grid = Grid::new_2d(4, 4, 1.0, 1.0);
        let op = DiffusionOperator::uniform(grid, BoundaryConditions::dirichlet(), 1.0);
        let mut d = vec![0.0; grid.len()];
        op.diagonal(&mut d);
        // Interior 2-D cell: 4 neighbours of weight 1, no z contribution.
        assert_eq!(d[grid.index(1, 1, 0)], 4.0);
        // Corner cell: 2 neighbours plus two Dirichlet faces of weight 2.
        assert_eq!(d[grid.index(0, 0, 0)], 6.0);
    }
}
