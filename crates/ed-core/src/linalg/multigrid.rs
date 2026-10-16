use super::{DiffusionOperator, Grid, LinearOperator, Preconditioner};

/// Geometric multigrid V-cycle preconditioner for [`DiffusionOperator`].
///
/// Coarse levels halve every active axis that has an even cell count ≥ 4 (semi-coarsening
/// otherwise) until a level has at most [`COARSEST_CELLS`] cells. Coarse operators are
/// rediscretised from the cell-averaged coefficients with the same boundary kinds.
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
/// With anisotropic cells, only the axes with the finest spacing (within a factor 1.5) are
/// coarsened until the cells are close to isotropic (semi-coarsening).
///
/// If the fine operator has face overrides ([`DiffusionOperator::set_face_sigma`]), every
/// coarse face coefficient is derived from the fine face coefficients instead: the fine
/// faces crossed by each child column are combined in series (half weight for the faces
/// inside the two coarse cells) and the columns in parallel. Impermeable or weakly
/// conducting fine faces such as membranes therefore stay (nearly) closed on coarse levels.
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
        let face_mode = !fine.face_sigma().is_empty();
        loop {
            let g = *ops.last().expect("fine level").grid();
            if g.len() <= COARSEST_CELLS {
                break;
            }
            let f = coarsening_factors(&g);
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
            if face_mode {
                op.explicit_faces = true;
                coarsen_faces(parent, f, &mut op.c);
            }
            op.set_shift(&shift);
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
        assert_eq!(
            fine.face_sigma().len(),
            self.ops[0].face_sigma().len(),
            "the number of overridden faces changed: rebuild the multigrid"
        );
        let top = &mut self.ops[0];
        top.sigma.copy_from_slice(&fine.sigma);
        top.shift.copy_from_slice(&fine.shift);
        top.overrides.copy_from_slice(&fine.overrides);
        top.recompute();
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
            if c_op.explicit_faces {
                coarsen_faces(f_op, self.factors[l], &mut c_op.c);
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

/// Coarsening factor per axis for the level below `g`: halve every active axis with an even
/// count ≥ 4 whose spacing is within 1.5× of the finest spacing; if none qualifies, every
/// even axis ≥ 4.
fn coarsening_factors(g: &Grid) -> [usize; 3] {
    let h_min = g.h[..g.dim].iter().copied().fold(f64::INFINITY, f64::min);
    let even = |a: usize| g.n[a] >= 4 && g.n[a] % 2 == 0;
    for max_h in [1.5 * h_min, f64::INFINITY] {
        let mut f = [1usize; 3];
        for (a, fa) in f.iter_mut().enumerate().take(g.dim) {
            if even(a) && g.h[a] <= max_h {
                *fa = 2;
            }
        }
        if f != [1, 1, 1] {
            return f;
        }
    }
    [1, 1, 1]
}

/// Coarse face weights from the fine operator's face weights (series along the axis,
/// parallel across it), written into `coarse_c`.
fn coarsen_faces(fine: &DiffusionOperator, f: [usize; 3], coarse_c: &mut [Vec<f64>; 3]) {
    let g = *fine.grid();
    let (cnx, cny) = (g.n[0] / f[0], g.n[1] / f[1]);
    let stride = [1, g.n[0], g.n[0] * g.n[1]];
    // Face weight times cell volume is the face conductance.
    let vol = g.h[0] * g.h[1] * if g.dim == 3 { g.h[2] } else { 1.0 };
    let coarse_vol = vol * (f[0] * f[1] * f[2]) as f64;
    for a in 0..3 {
        coarse_c[a].fill(0.0);
        if a >= g.dim {
            continue;
        }
        let c = &fine.c[a];
        let out = &mut coarse_c[a];
        g.for_each_cell(|i, j, k, cell| {
            let pos = [i, j, k][a];
            // Only the last fine layer of a coarse cell touches the next coarse cell.
            if pos % f[a] != f[a] - 1 || pos + 1 >= g.n[a] {
                return;
            }
            // Series resistance of this column (1/0 = ∞ makes a closed face).
            let mut r = 1.0 / c[cell];
            if f[a] == 2 {
                r += 0.5 / c[cell - stride[a]] + 0.5 / c[cell + stride[a]];
            }
            let parent = i / f[0] + cnx * (j / f[1] + cny * (k / f[2]));
            out[parent] += vol / r / coarse_vol;
        });
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
    fn anisotropic_grid_is_semi_coarsened() {
        let g = Grid::new_3d([32, 16, 16], [2.0, 0.5, 0.5]);
        assert_eq!(coarsening_factors(&g), [1, 2, 2]);
        let g = Grid::new_3d([32, 16, 16], [2.0, 1.0, 1.0]);
        assert_eq!(coarsening_factors(&g), [1, 2, 2]);
        let g = Grid::new_3d([32, 3, 3], [2.0, 1.0, 1.0]);
        assert_eq!(coarsening_factors(&g), [2, 1, 1]);
        let g = Grid::new_3d([32, 16, 16], [1.2, 1.0, 1.0]);
        assert_eq!(coarsening_factors(&g), [2, 2, 2]);
    }

    #[test]
    fn face_coarsening_reproduces_uniform_coefficients() {
        // Overriding every face with the value it already has must not change the levels.
        let grid = Grid::new_3d([16, 8, 4], [1.0, 0.5, 0.5]);
        let plain = DiffusionOperator::uniform(grid, BoundaryConditions::neumann(), 2.0);
        let mut faces = Vec::new();
        grid.for_each_cell(|i, j, k, cell| {
            for (axis, p) in [i, j, k].into_iter().enumerate() {
                if p + 1 < grid.n[axis] {
                    faces.push((crate::linalg::Face { axis, cell }, 2.0));
                }
            }
        });
        let mut over = plain.clone();
        over.set_face_sigma(&faces);
        let (a, b) = (Multigrid::new(&plain), Multigrid::new(&over));
        assert_eq!(a.levels(), b.levels());
        for (pa, pb) in a.ops.iter().zip(&b.ops) {
            for axis in 0..3 {
                for (x, y) in pa.c[axis].iter().zip(&pb.c[axis]) {
                    assert!((x - y).abs() < 1e-12 * x.abs().max(1.0), "{x} vs {y}");
                }
            }
        }
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
