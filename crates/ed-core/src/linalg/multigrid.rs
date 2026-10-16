use super::{DiffusionOperator, Grid, LinearOperator, Preconditioner};

/// Geometric multigrid V-cycle preconditioner for [`DiffusionOperator`].
///
/// Coarse levels halve every active axis that has an even cell count ≥ 4 (semi-coarsening
/// otherwise) until a level has at most [`COARSEST_CELLS`] cells. Of those axes, only the
/// ones with the shortest cells (within a factor 1.5) are halved, so anisotropic cells are
/// first coarsened along their short sides until they are roughly cubic. Coarse operators are
/// rediscretised from the cell-averaged coefficients with the same boundary kinds. If the
/// fine operator has [per-face coefficients](DiffusionOperator::set_face_sigma), each
/// coarse face gets the mean of the fine faces it covers; fine faces inside a coarse cell
/// are dropped, so a cut that does not lie on a coarse face is lost on that level.
/// Restriction averages the children; prolongation injects the parent value. The
/// smoother is red-black Gauss-Seidel, red-then-black before and black-then-red after the
/// coarse correction. The coarsest level is solved exactly with a banded Cholesky
/// factorisation when that costs at most [`DIRECT_FLOPS`] (a semidefinite one: rows with
/// no coupling and the constants of a pure-Neumann level are left at zero), otherwise
/// with [`coarse_sweeps`](Self::coarse_sweeps) symmetric sweep pairs. Either way one
/// V-cycle is a symmetric operator and can precondition CG.
///
/// The preconditioner keeps its own copy of the operator: call [`update`](Self::update)
/// after every [`DiffusionOperator::update_sigma`] or [`DiffusionOperator::set_shift`],
/// or CG keeps working but converges more slowly.
///
/// Coarsening needs even cell counts, so grids of size `k·2^m` with small `k` (such as
/// 64×64×32) work best. An axis with an odd count stops being coarsened, and the hierarchy
/// ends early with a larger coarsest level; CG still converges, with 2–6× more iterations.
/// Strongly anisotropic cells whose short axes cannot be halved also cost extra iterations.
/// [`coarsest_len`](Self::coarsest_len) shows where the hierarchy ended.
#[derive(Clone, Debug)]
pub struct Multigrid {
    ops: Vec<DiffusionOperator>,
    /// Coarsening factor (1 or 2) per axis from level `l` to `l + 1`.
    factors: Vec<[usize; 3]>,
    b: Vec<Vec<f64>>,
    x: Vec<Vec<f64>>,
    r: Vec<Vec<f64>>,
    /// Smoothing sweeps before and after each coarse correction.
    pub sweeps: usize,
    /// Symmetric Gauss-Seidel sweeps on the coarsest level when it is too large to factor.
    pub coarse_sweeps: usize,
    /// Direct factorisation of the coarsest level, if cheap enough.
    direct: Option<BandCholesky>,
}

/// Stop coarsening once a level has at most this many cells.
pub const COARSEST_CELLS: usize = 64;

/// Largest cost (cells × bandwidth², in multiply-adds) of factorising the coarsest level
/// directly. Larger coarsest levels are smoothed instead.
pub const DIRECT_FLOPS: usize = 1 << 20;

impl Multigrid {
    /// Builds the level hierarchy for `fine`. Allocates; reuse across solves and call
    /// [`update`](Self::update) when the coefficients change.
    pub fn new(fine: &DiffusionOperator) -> Self {
        let mut ops = vec![fine.clone()];
        let mut factors = Vec::new();
        loop {
            let g = *ops.last().expect("fine level").grid();
            if g.len() <= COARSEST_CELLS {
                break;
            }
            // Coarsen the axes that can be halved, but only those whose cells are not much
            // longer than the shortest such axis: coupling is weak along long cells, and
            // point smoothing cannot smooth along weak axes.
            let halvable = |a: usize| g.n[a] >= 4 && g.n[a] % 2 == 0;
            let h_min = (0..g.dim)
                .filter(|&a| halvable(a))
                .map(|a| g.h[a])
                .fold(f64::INFINITY, f64::min);
            let mut f = [1usize; 3];
            for (a, fa) in f.iter_mut().enumerate().take(g.dim) {
                if halvable(a) && g.h[a] < 1.5 * h_min {
                    *fa = 2;
                }
            }
            if f == [1, 1, 1] {
                break;
            }
            let mut cg = g;
            for ((n, h), &fa) in cg.n.iter_mut().zip(&mut cg.h).zip(&f) {
                *n /= fa;
                *h *= fa as f64;
            }
            let parent = ops.last().expect("fine level");
            let (mut sigma, mut shift) = (vec![0.0; cg.len()], vec![0.0; cg.len()]);
            restrict(&g, &cg, f, &parent.sigma, &mut sigma);
            restrict(&g, &cg, f, &parent.shift, &mut shift);
            let mut op = DiffusionOperator::new(cg, *fine.boundary_conditions(), &sigma);
            if let Some(fs) = &parent.face_sigma {
                let mut cf = [
                    vec![0.0; cg.len()],
                    vec![0.0; cg.len()],
                    vec![0.0; cg.len()],
                ];
                restrict_faces(&g, &cg, f, fs, &mut cf);
                op.face_sigma = Some(cf);
            }
            op.set_shift(&shift);
            ops.push(op);
            factors.push(f);
        }
        let alloc = |ops: &[DiffusionOperator]| -> Vec<Vec<f64>> {
            ops.iter().map(|o| vec![0.0; o.grid().len()]).collect()
        };
        let direct = BandCholesky::try_new(ops.last().expect("fine level"));
        Self {
            b: alloc(&ops),
            x: alloc(&ops),
            r: alloc(&ops),
            ops,
            factors,
            sweeps: 2,
            coarse_sweeps: 16,
            direct,
        }
    }

    /// Number of levels including the finest.
    pub fn levels(&self) -> usize {
        self.ops.len()
    }

    /// Number of cells on the coarsest level.
    pub fn coarsest_len(&self) -> usize {
        self.ops.last().expect("at least one level").len()
    }

    /// Re-derives all levels after the fine operator's coefficients changed. Does not
    /// allocate. `fine` must have the same grid and boundary kinds as at construction.
    pub fn update(&mut self, fine: &DiffusionOperator) {
        assert_eq!(fine.grid(), self.ops[0].grid());
        assert_eq!(
            fine.boundary_conditions(),
            self.ops[0].boundary_conditions()
        );
        self.ops[0].sigma.copy_from_slice(&fine.sigma);
        self.ops[0].shift.copy_from_slice(&fine.shift);
        assert_eq!(
            fine.face_sigma.is_some(),
            self.ops[0].face_sigma.is_some(),
            "face coefficients were set or cleared after the hierarchy was built"
        );
        if let (Some(dst), Some(src)) = (&mut self.ops[0].face_sigma, &fine.face_sigma) {
            for (d, s) in dst.iter_mut().zip(src) {
                d.copy_from_slice(s);
            }
        }
        self.ops[0].recompute();
        for l in 0..self.factors.len() {
            let (head, tail) = self.ops.split_at_mut(l + 1);
            let (f_op, c_op) = (&head[l], &mut tail[0]);
            let cg = *c_op.grid();
            restrict(
                f_op.grid(),
                &cg,
                self.factors[l],
                &f_op.sigma,
                &mut c_op.sigma,
            );
            restrict(
                f_op.grid(),
                &cg,
                self.factors[l],
                &f_op.shift,
                &mut c_op.shift,
            );
            if let (Some(src), Some(dst)) = (&f_op.face_sigma, &mut c_op.face_sigma) {
                restrict_faces(f_op.grid(), &cg, self.factors[l], src, dst);
            }
            c_op.recompute();
        }
        if let Some(d) = &mut self.direct {
            d.factor(self.ops.last().expect("fine level"));
        }
    }
}

impl Preconditioner for Multigrid {
    fn apply(&mut self, r: &[f64], z: &mut [f64]) {
        let last = self.ops.len() - 1;
        self.b[0].copy_from_slice(r);

        for l in 0..last {
            let op = &self.ops[l];
            let x = &mut self.x[l];
            for sweep in 0..self.sweeps {
                if sweep == 0 {
                    op.gauss_seidel_from_zero(&self.b[l], x);
                } else {
                    op.gauss_seidel(&self.b[l], x, true);
                }
            }
            if self.sweeps == 0 {
                x.fill(0.0);
            }
            op.residual(&self.b[l], x, &mut self.r[l]);
            restrict(
                op.grid(),
                self.ops[l + 1].grid(),
                self.factors[l],
                &self.r[l],
                &mut self.b[l + 1],
            );
        }

        let op = &self.ops[last];
        let x = &mut self.x[last];
        if let Some(d) = &self.direct {
            d.solve(&self.b[last], x);
        } else {
            x.fill(0.0);
            for _ in 0..self.coarse_sweeps {
                op.gauss_seidel(&self.b[last], x, true);
                op.gauss_seidel(&self.b[last], x, false);
            }
        }

        for l in (0..last).rev() {
            let (xf, xc) = self.x.split_at_mut(l + 1);
            prolong_add(self.ops[l].grid(), self.factors[l], &xc[0], &mut xf[l]);
            for _ in 0..self.sweeps {
                self.ops[l].gauss_seidel(&self.b[l], &mut xf[l], false);
            }
        }

        z.copy_from_slice(&self.x[0]);
    }
}

/// Banded Cholesky factorisation `A = L Lᵀ` of a symmetric positive semidefinite matrix.
/// A pivot that vanishes (relative to its diagonal entry) marks a direction in the null
/// space: its row and column are dropped and the matching solution entry is set to zero,
/// which keeps the solve a symmetric operator.
#[derive(Clone, Debug)]
struct BandCholesky {
    n: usize,
    band: usize,
    /// `l[i * (band + 1) + d]` is `L[i][i − d]`.
    l: Vec<f64>,
    /// `1 / L_ii`, or 0 for dropped pivots.
    inv_pivot: Vec<f64>,
}

impl BandCholesky {
    /// Factorises `op` if that costs at most [`DIRECT_FLOPS`] multiply-adds.
    fn try_new(op: &DiffusionOperator) -> Option<Self> {
        let (n, band) = (op.len(), op.bandwidth());
        if n * band * band > DIRECT_FLOPS {
            return None;
        }
        let mut c = Self {
            n,
            band,
            l: vec![0.0; n * (band + 1)],
            inv_pivot: vec![0.0; n],
        };
        c.factor(op);
        Some(c)
    }

    fn factor(&mut self, op: &DiffusionOperator) {
        let (n, b) = (self.n, self.band);
        let w = b + 1;
        op.fill_lower_band(b, &mut self.l);
        let l = &mut self.l;
        for i in 0..n {
            for j in i.saturating_sub(b)..=i {
                let k0 = i.saturating_sub(b);
                let mut s = l[i * w + (i - j)];
                for k in k0..j {
                    s -= l[i * w + (i - k)] * l[j * w + (j - k)];
                }
                if j < i {
                    l[i * w + (i - j)] = s * self.inv_pivot[j];
                } else {
                    let a_ii = op.diag[i];
                    if s <= 1e-10 * a_ii || s <= 0.0 {
                        self.inv_pivot[i] = 0.0;
                        l[i * w] = 0.0;
                    } else {
                        let p = s.sqrt();
                        self.inv_pivot[i] = 1.0 / p;
                        l[i * w] = p;
                    }
                }
            }
        }
    }

    fn solve(&self, rhs: &[f64], x: &mut [f64]) {
        let (n, b, l) = (self.n, self.band, &self.l);
        let w = b + 1;
        for i in 0..n {
            let mut s = rhs[i];
            for k in i.saturating_sub(b)..i {
                s -= l[i * w + (i - k)] * x[k];
            }
            x[i] = s * self.inv_pivot[i];
        }
        for i in (0..n).rev() {
            let mut s = x[i];
            for k in i + 1..(i + b + 1).min(n) {
                s -= l[k * w + (k - i)] * x[k];
            }
            x[i] = s * self.inv_pivot[i];
        }
    }
}

/// Calls `f(fine_row, coarse_row)` for every x-row of `fine`, where `coarse_row` is the
/// start of the parent row.
#[inline(always)]
fn for_each_row_pair(fine: &Grid, f: [usize; 3], mut g: impl FnMut(usize, usize)) {
    let [nx, ny, nz] = fine.n;
    let (cnx, cny) = (nx / f[0], ny / f[1]);
    let (sy, sz) = (f[1] - 1, f[2] - 1);
    for k in 0..nz {
        for j in 0..ny {
            g(nx * (j + ny * k), cnx * ((j >> sy) + cny * (k >> sz)));
        }
    }
}

/// `coarse[parent] = mean of fine children`. Coarsening factors are 1 or 2.
fn restrict(fine: &Grid, coarse: &Grid, f: [usize; 3], src: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(coarse.len() * f[0] * f[1] * f[2], fine.len());
    dst.fill(0.0);
    let (nx, sx) = (fine.n[0], f[0] - 1);
    for_each_row_pair(fine, f, |row, crow| {
        let (src, dst) = (&src[row..row + nx], &mut dst[crow..crow + (nx >> sx)]);
        for (i, v) in src.iter().enumerate() {
            dst[i >> sx] += v;
        }
    });
    let w = 1.0 / (f[0] * f[1] * f[2]) as f64;
    dst.iter_mut().for_each(|v| *v *= w);
}

/// Coarse face coefficients: each coarse face gets the mean of the fine faces that lie on
/// it. Along a halved axis those are the faces between the two children of neighbouring
/// parents (odd fine positions); along an axis that is not coarsened, every face.
fn restrict_faces(
    fine: &Grid,
    coarse: &Grid,
    f: [usize; 3],
    src: &[Vec<f64>; 3],
    dst: &mut [Vec<f64>; 3],
) {
    let [cnx, cny, _] = coarse.n;
    for (a, d) in dst.iter_mut().enumerate() {
        d.fill(0.0);
        if a >= fine.dim {
            continue;
        }
        let w = f[a] as f64 / (f[0] * f[1] * f[2]) as f64;
        fine.for_each_cell(|i, j, k, cell| {
            let p = [i, j, k][a];
            if p + 1 >= fine.n[a] || (f[a] == 2 && p % 2 == 0) {
                return;
            }
            let parent = i / f[0] + cnx * (j / f[1] + cny * (k / f[2]));
            d[parent] += w * src[a][cell];
        });
    }
}

/// `fine[child] += coarse[parent]`. Coarsening factors are 1 or 2.
fn prolong_add(fine: &Grid, f: [usize; 3], src: &[f64], dst: &mut [f64]) {
    let (nx, sx) = (fine.n[0], f[0] - 1);
    for_each_row_pair(fine, f, |row, crow| {
        let (dst, src) = (&mut dst[row..row + nx], &src[crow..crow + (nx >> sx)]);
        for (i, v) in dst.iter_mut().enumerate() {
            *v += src[i >> sx];
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{dot, BoundaryConditions, LinearOperator};

    #[test]
    fn hierarchy_for_knp_grid() {
        let grid = Grid::new_3d([64, 64, 32], [15.0; 3]);
        let op = DiffusionOperator::uniform(grid, BoundaryConditions::neumann(), 1.0);
        let mg = Multigrid::new(&op);
        assert_eq!(mg.levels(), 5);
        assert_eq!(mg.ops.last().unwrap().grid().n, [4, 4, 2]);
    }

    #[test]
    fn band_cholesky_solves_regular_singular_and_decoupled_systems() {
        let grid = Grid::new_3d([5, 4, 3], [1.0, 0.7, 1.3]);
        let n = grid.len();
        let sigma: Vec<f64> = (0..n)
            .map(|c| {
                if c % 7 == 3 {
                    0.0
                } else {
                    1.0 + 0.3 * (c as f64).sin()
                }
            })
            .collect();
        for bc in [
            BoundaryConditions::dirichlet(),
            BoundaryConditions::neumann(),
        ] {
            let op = DiffusionOperator::new(grid, bc, &sigma);
            let chol = BandCholesky::try_new(&op).expect("small enough");
            assert_eq!(chol.band, 20);
            // A right-hand side in the range of A: b = A u, zero on the decoupled rows.
            let u: Vec<f64> = (0..n).map(|c| (c as f64 * 0.37).cos()).collect();
            let mut b = vec![0.0; n];
            op.apply(&u, &mut b);
            let mut x = vec![0.0; n];
            chol.solve(&b, &mut x);
            let mut ax = vec![0.0; n];
            op.apply(&x, &mut ax);
            for (p, q) in ax.iter().zip(&b) {
                assert!((p - q).abs() < 1e-10, "{p} vs {q}");
            }
            for c in (0..n).filter(|c| c % 7 == 3) {
                assert_eq!(x[c], 0.0);
            }
        }
    }

    #[test]
    fn anisotropic_cells_coarsen_along_short_axes_first() {
        let grid = Grid::new_3d([64, 12, 12], [4.0, 1.0, 1.0]);
        let op = DiffusionOperator::uniform(grid, BoundaryConditions::neumann(), 1.0);
        let mg = Multigrid::new(&op);
        let dims: Vec<[usize; 3]> = mg.ops.iter().map(|o| o.grid().n).collect();
        assert_eq!(
            dims,
            [
                [64, 12, 12],
                [64, 6, 6],
                [64, 3, 3],
                [32, 3, 3],
                [16, 3, 3],
                [8, 3, 3],
                [4, 3, 3]
            ]
        );
    }

    #[test]
    fn coarse_faces_average_the_fine_faces_they_cover() {
        let grid = Grid::new_2d(16, 8, 1.0, 1.0);
        let n = grid.len();
        let mut op = DiffusionOperator::uniform(grid, BoundaryConditions::neumann(), 1.0);
        // Cut the x-faces between columns 3 and 4 (on a coarse face) in the lower half,
        // and between columns 4 and 5 (inside coarse cells) everywhere.
        let mut fx = vec![1.0; n];
        for j in 0..8 {
            if j < 4 {
                fx[grid.index(3, j, 0)] = 0.0;
            }
            fx[grid.index(4, j, 0)] = 0.0;
        }
        let ones = vec![1.0; n];
        op.set_face_sigma([&fx, &ones, &ones]);
        let mg = Multigrid::new(&op);
        let c1 = &mg.ops[1];
        let fs = c1.face_sigma().expect("coarse face sigma");
        let g1 = *c1.grid();
        assert_eq!(g1.n, [8, 4, 1]);
        // Coarse face between coarse columns 1 and 2: rows 0-1 fully cut, rows 2-3 open.
        assert_eq!(fs[0][g1.index(1, 0, 0)], 0.0);
        assert_eq!(fs[0][g1.index(1, 1, 0)], 0.0);
        assert_eq!(fs[0][g1.index(1, 3, 0)], 1.0);
        // The cut inside coarse column 2 is dropped.
        assert_eq!(fs[0][g1.index(2, 0, 0)], 1.0);
        assert_eq!(fs[1][g1.index(0, 0, 0)], 1.0);
    }

    #[test]
    fn v_cycle_is_symmetric() {
        let grid = Grid::new_2d(16, 12, 1.0, 1.0);
        let sigma: Vec<f64> = (0..grid.len())
            .map(|c| 1.0 + 0.5 * ((c as f64) * 0.3).sin())
            .collect();
        let op = DiffusionOperator::new(grid, BoundaryConditions::dirichlet(), &sigma);
        let mut mg = Multigrid::new(&op);
        assert!(mg.levels() > 1);
        let n = op.len();
        let u: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).cos()).collect();
        let v: Vec<f64> = (0..n).map(|i| (i as f64 * 1.3).sin()).collect();
        let (mut mu, mut mv) = (vec![0.0; n], vec![0.0; n]);
        mg.apply(&u, &mut mu);
        mg.apply(&v, &mut mv);
        let (a, b) = (dot(&v, &mu), dot(&u, &mv));
        assert!((a - b).abs() < 1e-10 * a.abs().max(1.0), "{a} vs {b}");
        assert!(dot(&u, &mu) > 0.0);
    }
}
