//! Loops over the faces of a uniform cell-centred grid (x fastest, see [`Grid`]), used by
//! the finite-volume flux updates. All helpers treat the grid as 3-D: on a grid from
//! [`Grid::new_2d`] (`nz = 1`) axis 2 has no interior faces, and its "boundary" layers are
//! the whole grid, so 2-D callers skip axis 2.

use ed_core::linalg::Grid;

/// Distance in storage between a cell and its upper neighbour along `axis`.
pub fn stride(grid: &Grid, axis: usize) -> usize {
    [1, grid.n[0], grid.n[0] * grid.n[1]][axis]
}

/// Calls `f(start, len)` for each contiguous run of cells `p` that have an upper neighbour
/// `p + stride(axis)`: one run per x-row for axis 0, per z-layer for axis 1, and a single
/// run for axis 2.
pub fn for_each_face_run(grid: &Grid, axis: usize, mut f: impl FnMut(usize, usize)) {
    let [nx, ny, nz] = grid.n;
    match axis {
        0 => (0..ny * nz).for_each(|r| f(r * nx, nx - 1)),
        1 => (0..nz).for_each(|k| f(k * nx * ny, nx * (ny - 1))),
        2 => f(0, nx * ny * (nz - 1)),
        _ => panic!("axis {axis} out of range"),
    }
}

/// Adds the net inflow `Σ_a (flux_a[p − s_a] − flux_a[p])` to every cell `p`, where
/// `flux_a[p]` is the outflow through the upper face of `p` along axis `a`.
///
/// Entries of `flux_a` on the last layer along `a` (cells without an upper face) are read
/// and must be 0: fill fluxes with [`for_each_face_run`] into zero-initialised buffers.
pub fn add_net_inflow(grid: &Grid, flux: &[Vec<f64>; 3], out: &mut [f64]) {
    let n = grid.len();
    assert_eq!(out.len(), n);
    assert!(
        flux.iter().all(|f| f.len() == n),
        "one flux per cell and axis"
    );
    #[cfg(debug_assertions)]
    grid.for_each_cell(|i, j, k, p| {
        for (a, f) in flux.iter().enumerate() {
            let last = [i, j, k][a] + 1 == grid.n[a];
            debug_assert!(!last || f[p] == 0.0, "flux on the last layer of axis {a}");
        }
    });
    let [fx, fy, fz] = flux;
    let (sy, sz) = (stride(grid, 1), stride(grid, 2));
    // Cells of the first z-layer, where lower neighbours may be missing.
    for p in 0..sz.min(n) {
        let mut v = -(fx[p] + fy[p] + fz[p]);
        if p >= 1 {
            v += fx[p - 1];
        }
        if p >= sy {
            v += fy[p - sy];
        }
        out[p] += v;
    }
    if sz >= n {
        return;
    }
    let (o, x, xm) = (&mut out[sz..], &fx[sz..], &fx[sz - 1..n - 1]);
    let (y, ym, z, zm) = (&fy[sz..], &fy[sz - sy..n - sy], &fz[sz..], &fz[..n - sz]);
    for i in 0..o.len() {
        o[i] += (xm[i] - x[i]) + (ym[i] - y[i]) + (zm[i] - z[i]);
    }
}

/// Calls `f(p)` for every cell on the lower and the upper boundary layer along `axis`
/// (twice for a cell that is on both, i.e. when the axis has a single cell), lower layer
/// first in each pair. Callers whose boundary terms differ between the two sides need to
/// tell them apart by position.
pub fn for_each_boundary_cell(grid: &Grid, axis: usize, mut f: impl FnMut(usize)) {
    let [nx, ny, _] = grid.n;
    let offset = (grid.n[axis] - 1) * stride(grid, axis);
    let mut layer = grid.n;
    layer[axis] = 1;
    for k in 0..layer[2] {
        for j in 0..layer[1] {
            let row = nx * (j + ny * k);
            for i in 0..layer[0] {
                f(row + i);
                f(row + i + offset);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_runs_cover_every_interior_face_once() {
        for (grid, axis) in SHAPES.iter().flat_map(|&n| (0..3).map(move |a| (n, a))) {
            let grid = Grid::new_3d(grid, [1.0; 3]);
            let s = stride(&grid, axis);
            let mut seen = vec![0; grid.len()];
            for_each_face_run(&grid, axis, |p, len| {
                (p..p + len).for_each(|p| seen[p] += 1)
            });
            grid.for_each_cell(|i, j, k, cell| {
                let has_upper = [i, j, k][axis] + 1 < grid.n[axis];
                assert_eq!(seen[cell], has_upper as i32, "axis {axis}, cell {cell}");
                if has_upper {
                    let mut up = [i, j, k];
                    up[axis] += 1;
                    assert_eq!(cell + s, grid.index(up[0], up[1], up[2]));
                }
            });
        }
    }

    #[test]
    fn net_inflow_conserves_and_matches_a_direct_sum() {
        for shape in SHAPES {
            net_inflow_on(Grid::new_3d(shape, [1.0; 3]));
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "last layer")]
    fn net_inflow_rejects_flux_on_the_last_layer() {
        let grid = Grid::new_3d([3, 2, 2], [1.0; 3]);
        let mut flux = [vec![0.0; 12], vec![0.0; 12], vec![0.0; 12]];
        flux[0][2] = 1.0;
        add_net_inflow(&grid, &flux, &mut [0.0; 12]);
    }

    /// Grid shapes including single-cell axes and a single z-layer.
    const SHAPES: [[usize; 3]; 7] = [
        [5, 4, 3],
        [1, 1, 1],
        [1, 5, 1],
        [5, 1, 1],
        [3, 4, 1],
        [3, 1, 4],
        [1, 3, 4],
    ];

    fn net_inflow_on(grid: Grid) {
        let n = grid.len();
        let mut flux = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
        for (a, f) in flux.iter_mut().enumerate() {
            for_each_face_run(&grid, a, |p, len| {
                (p..p + len).for_each(|p| f[p] = ((p * 7 + a * 3) % 11) as f64 - 5.0)
            });
        }
        let mut out = vec![1.0; n];
        add_net_inflow(&grid, &flux, &mut out);
        let mut expected = vec![1.0; n];
        for (a, f) in flux.iter().enumerate() {
            let s = stride(&grid, a);
            for_each_face_run(&grid, a, |p, len| {
                for p in p..p + len {
                    expected[p] -= f[p];
                    expected[p + s] += f[p];
                }
            });
        }
        assert_eq!(out, expected, "{:?}", grid.n);
        assert!((out.iter().sum::<f64>() - n as f64).abs() < 1e-12);
    }

    #[test]
    fn boundary_cells_are_the_outer_layers() {
        for (grid, axis) in SHAPES.iter().flat_map(|&n| (0..3).map(move |a| (n, a))) {
            let grid = Grid::new_3d(grid, [1.0; 3]);
            let mut seen = vec![0; grid.len()];
            for_each_boundary_cell(&grid, axis, |p| seen[p] += 1);
            grid.for_each_cell(|i, j, k, cell| {
                let pos = [i, j, k][axis];
                let expected = (pos == 0) as i32 + (pos + 1 == grid.n[axis]) as i32;
                assert_eq!(seen[cell], expected, "axis {axis}, cell {cell}");
            });
        }
    }
}
