use super::{DiffusionOperator, Grid, LinearOperator, Preconditioner};

/// Geometric multigrid V-cycle preconditioner for [`DiffusionOperator`].
///
/// Coarse levels halve every active axis that has an even cell count ≥ 4 (semi-coarsening
/// otherwise) until a level has at most [`COARSEST_CELLS`] cells. Coarse operators are
/// rediscretised from the cell-averaged coefficients with the same boundary kinds. With
/// explicit face coefficients ([`DiffusionOperator::set_face_sigma`]), a coarse face takes
/// the mean of the fine faces on it; faces inside a coarse cell drop out, so thin
/// high-contrast features such as membranes are not represented on coarse levels and CG
/// needs more iterations.
/// Restriction averages the children; prolongation injects the parent value. The
/// smoother is red-black Gauss-Seidel, red-then-black before and black-then-red after the
/// coarse correction, and the coarsest level uses symmetric sweep pairs, so one V-cycle
/// is a symmetric operator and can precondition CG.
///
/// The preconditioner keeps its own copy of the operator: call [`update`](Self::update)
/// after every [`DiffusionOperator::update_sigma`] or [`DiffusionOperator::set_shift`],
/// or CG keeps working but converges more slowly.
///
/// Coarsening needs even cell counts, so grids of size `k·2^m` with small `k` (such as
/// 64×64×32) work best. An axis with an odd count stops being coarsened, and the hierarchy
/// ends early with a larger coarsest level; CG still converges, with 2–6× more iterations.
/// Strongly anisotropic cells (`h` differing per axis) also cost extra iterations.
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
    /// Symmetric Gauss-Seidel sweeps on the coarsest level.
    pub coarse_sweeps: usize,
}

/// Stop coarsening once a level has at most this many cells.
pub const COARSEST_CELLS: usize = 64;

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
            let mut f = [1usize; 3];
            for (a, fa) in f.iter_mut().enumerate().take(g.dim) {
                if g.n[a] >= 4 && g.n[a] % 2 == 0 {
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
            op.set_shift(&shift);
            if let Some(ff) = &parent.face_sigma {
                let mut cf = [
                    vec![0.0; cg.len()],
                    vec![0.0; cg.len()],
                    vec![0.0; cg.len()],
                ];
                restrict_faces(&g, &cg, f, ff, &mut cf);
                op.set_face_sigma([&cf[0], &cf[1], &cf[2]]);
            }
            ops.push(op);
            factors.push(f);
        }
        let alloc = |ops: &[DiffusionOperator]| -> Vec<Vec<f64>> {
            ops.iter().map(|o| vec![0.0; o.grid().len()]).collect()
        };
        Self {
            b: alloc(&ops),
            x: alloc(&ops),
            r: alloc(&ops),
            ops,
            factors,
            sweeps: 2,
            coarse_sweeps: 16,
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
            "face coefficients must be set before building the multigrid"
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
            if let (Some(ff), Some(cf)) = (&f_op.face_sigma, &mut c_op.face_sigma) {
                restrict_faces(f_op.grid(), &cg, self.factors[l], ff, cf);
            }
            c_op.recompute();
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
        x.fill(0.0);
        for _ in 0..self.coarse_sweeps {
            op.gauss_seidel(&self.b[last], x, true);
            op.gauss_seidel(&self.b[last], x, false);
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

/// Coarse face coefficients: each coarse face (between a coarse cell and its upper
/// neighbour along an axis) takes the mean of the fine faces that lie on it.
fn restrict_faces(
    fine: &Grid,
    coarse: &Grid,
    f: [usize; 3],
    src: &[Vec<f64>; 3],
    dst: &mut [Vec<f64>; 3],
) {
    for (a, dst) in dst.iter_mut().enumerate() {
        let w = f[a] as f64 / (f[0] * f[1] * f[2]) as f64;
        coarse.for_each_cell(|ci, cj, ck, c| {
            let cpos = [ci, cj, ck];
            if a >= coarse.dim || cpos[a] + 1 >= coarse.n[a] {
                dst[c] = 0.0;
                return;
            }
            let mut sum = 0.0;
            for dk in 0..f[2] {
                for dj in 0..f[1] {
                    for di in 0..f[0] {
                        // Only the last child layer along `a` touches the coarse face.
                        if [di, dj, dk][a] + 1 != f[a] {
                            continue;
                        }
                        let p = [ci * f[0] + di, cj * f[1] + dj, ck * f[2] + dk];
                        sum += src[a][fine.index(p[0], p[1], p[2])];
                    }
                }
            }
            dst[c] = sum * w;
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
    fn face_restriction_averages_fine_faces_on_coarse_face() {
        let fine = Grid::new_2d(4, 4, 1.0, 1.0);
        let coarse = Grid::new_2d(2, 2, 2.0, 2.0);
        let mut src = [vec![0.0; 16], vec![0.0; 16], vec![0.0; 16]];
        // x-faces between i = 1 and i = 2 in rows j = 0, 1 carry 1 and 3.
        src[0][fine.index(1, 0, 0)] = 1.0;
        src[0][fine.index(1, 1, 0)] = 3.0;
        // An interior face of coarse cell 0 must not count.
        src[0][fine.index(0, 0, 0)] = 100.0;
        let mut dst = [vec![0.0; 4], vec![0.0; 4], vec![0.0; 4]];
        restrict_faces(&fine, &coarse, [2, 2, 1], &src, &mut dst);
        assert_eq!(dst[0][coarse.index(0, 0, 0)], 2.0);
        assert_eq!(dst[0][coarse.index(1, 0, 0)], 0.0);
        assert!(dst[1].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn preconditions_operator_with_face_coefficients() {
        use crate::linalg::{CgOptions, Pcg};
        let grid = Grid::new_3d([16, 16, 8], [1.0; 3]);
        let mut op = DiffusionOperator::uniform(grid, BoundaryConditions::dirichlet(), 1.0);
        // A weakly conducting wall between x = 7 and x = 8.
        let ones = vec![1.0; grid.len()];
        let mut fx = ones.clone();
        grid.for_each_cell(|i, _, _, c| {
            if i == 7 {
                fx[c] = 1e-3;
            }
        });
        op.set_face_sigma([&ones, &ones, &ones]);
        let mut mg = Multigrid::new(&op);
        op.set_face_sigma([&fx, &ones, &ones]);
        mg.update(&op);
        let b: Vec<f64> = (0..grid.len()).map(|c| (c as f64 * 0.3).sin()).collect();
        let mut x = vec![0.0; grid.len()];
        let stats = Pcg::new(grid.len()).solve(&op, &mut mg, &b, &mut x, &CgOptions::default());
        assert!(stats.converged && stats.iterations < 30, "{stats:?}");
        let mut r = vec![0.0; grid.len()];
        op.apply(&x, &mut r);
        let err = r
            .iter()
            .zip(&b)
            .map(|(u, v)| (u - v).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-6, "{err}");
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
