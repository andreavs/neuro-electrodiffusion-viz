//! Finite-volume face loops on the grid, written over contiguous runs of cells so
//! they vectorise.

use ed_core::linalg::Grid;
use std::ops::Range;

/// Adds the net Nernst-Planck flux through every interior face along `axis` to `dc`:
/// `w (c_p − c_q) − dz (c_p + c_q) drift_p` leaves cell `p` for its upper neighbour `q`.
/// `drift[p]` belongs to the face between `p` and its upper neighbour. Works on pairs
/// of x-rows so the loops vectorise; `buf` (at least `nx − 1` long) holds one row of
/// fluxes and is only used along axis 0.
#[allow(clippy::too_many_arguments)]
pub fn add_nernst_planck(
    g: &Grid,
    axis: usize,
    c: &[f64],
    drift: &[f64],
    w: f64,
    dz: f64,
    dc: &mut [f64],
    buf: &mut [f64],
) {
    let [nx, ny, nz] = g.n;
    let flux = |p: &f64, q: &f64, v: &f64| w * (p - q) - dz * (p + q) * v;
    if axis == 0 {
        if nx == 1 {
            return;
        }
        let buf = &mut buf[..nx - 1];
        for row in (0..ny * nz).map(|r| r * nx) {
            let (cr, v) = (&c[row..row + nx], &drift[row..row + nx - 1]);
            for ((f, v), (p, q)) in buf.iter_mut().zip(v).zip(cr.iter().zip(&cr[1..])) {
                *f = flux(p, q, v);
            }
            let dr = &mut dc[row..row + nx];
            dr[..nx - 1]
                .iter_mut()
                .zip(&*buf)
                .for_each(|(d, f)| *d -= f);
            dr[1..].iter_mut().zip(&*buf).for_each(|(d, f)| *d += f);
        }
        return;
    }
    let s = [1, nx, nx * ny][axis];
    let rows = (0..nz).flat_map(|k| (0..ny).map(move |j| (j, k)));
    for (j, k) in rows {
        if (axis == 1 && j + 1 == ny) || (axis == 2 && k + 1 == nz) {
            continue;
        }
        let p = nx * (j + ny * k);
        let (lo, hi) = dc.split_at_mut(p + s);
        let (dp, dq) = (&mut lo[p..p + nx], &mut hi[..nx]);
        let (cp, cq, v) = (&c[p..p + nx], &c[p + s..p + s + nx], &drift[p..p + nx]);
        let it = dp.iter_mut().zip(dq.iter_mut()).zip(v);
        for (((dp, dq), v), (p, q)) in it.zip(cp.iter().zip(cq)) {
            let f = flux(p, q, v);
            *dp -= f;
            *dq += f;
        }
    }
}

/// Calls `f(run)` for runs of cells `p` that have an upper neighbour `p + stride`
/// along `axis`. Runs are contiguous, so the loops over them vectorise.
pub fn for_each_face_run(g: &Grid, axis: usize, mut f: impl FnMut(Range<usize>)) {
    let [nx, ny, nz] = g.n;
    match axis {
        0 => (0..ny * nz).for_each(|row| f(row * nx..row * nx + nx - 1)),
        1 => (0..nz).for_each(|k| f(k * nx * ny..k * nx * ny + nx * (ny - 1))),
        _ => f(0..nx * ny * (nz - 1)),
    }
}

/// Moves `flux[p]` from cell `p` to its upper neighbour along `axis`, for every face.
pub fn apply_face_fluxes(g: &Grid, axis: usize, flux: &[f64], dc: &mut [f64]) {
    let s = [1, g.n[0], g.n[0] * g.n[1]][axis];
    for_each_face_run(g, axis, |r| {
        let f = &flux[r.clone()];
        dc[r.clone()].iter_mut().zip(f).for_each(|(d, f)| *d -= f);
        dc[r.start + s..r.end + s]
            .iter_mut()
            .zip(f)
            .for_each(|(d, f)| *d += f);
    });
}

/// Calls `f(run)` for runs of cells on the lower and then the upper boundary layer
/// along `axis`, so a cell on both layers (a single-cell axis) is visited twice.
pub fn for_each_boundary_run(g: &Grid, axis: usize, mut f: impl FnMut(Range<usize>)) {
    let [nx, ny, nz] = g.n;
    let plane = nx * ny;
    for upper in [false, true] {
        match axis {
            0 => (0..ny * nz).for_each(|row| {
                let p = row * nx + if upper { nx - 1 } else { 0 };
                f(p..p + 1)
            }),
            1 => (0..nz).for_each(|k| {
                let p = k * plane + if upper { plane - nx } else { 0 };
                f(p..p + nx)
            }),
            _ => {
                let p = if upper { (nz - 1) * plane } else { 0 };
                f(p..p + plane)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A general grid and grids with single-cell axes.
    const GRIDS: [[usize; 3]; 3] = [[5, 4, 3], [4, 1, 3], [1, 4, 1]];

    #[test]
    fn face_runs_cover_each_interior_face_once() {
        for n in GRIDS {
            let g = Grid::new_3d(n, [1.0; 3]);
            for a in 0..3 {
                let mut seen = vec![0; g.len()];
                for_each_face_run(&g, a, |r| r.for_each(|p| seen[p] += 1));
                g.for_each_cell(|i, j, k, p| {
                    let interior = [i, j, k][a] + 1 < g.n[a];
                    assert_eq!(seen[p], interior as i32, "{n:?}, axis {a}, cell {p}");
                });
            }
        }
    }

    #[test]
    fn boundary_runs_cover_both_layers() {
        for n in GRIDS {
            let g = Grid::new_3d(n, [1.0; 3]);
            for a in 0..3 {
                let mut seen = vec![0; g.len()];
                for_each_boundary_run(&g, a, |r| r.for_each(|p| seen[p] += 1));
                g.for_each_cell(|i, j, k, p| {
                    let pos = [i, j, k][a];
                    let expected = (pos == 0) as i32 + (pos + 1 == g.n[a]) as i32;
                    assert_eq!(seen[p], expected, "{n:?}, axis {a}, cell {p}");
                });
            }
        }
    }

    #[test]
    fn nernst_planck_matches_face_by_face_reference() {
        for n in GRIDS {
            let g = Grid::new_3d(n, [1.0; 3]);
            let len = g.len();
            let c: Vec<f64> = (0..len).map(|i| (i as f64 * 0.37).sin() + 2.0).collect();
            let drift: Vec<f64> = (0..len).map(|i| (i as f64 * 0.11).cos()).collect();
            let stride = [1, n[0], n[0] * n[1]];
            for (a, &s) in stride.iter().enumerate() {
                let (w, dz) = (0.7, 0.3);
                let mut dc = vec![0.0; len];
                let mut buf = vec![0.0; len];
                add_nernst_planck(&g, a, &c, &drift, w, dz, &mut dc, &mut buf);
                let mut expected = vec![0.0; len];
                let mut flux = vec![0.0; len];
                for_each_face_run(&g, a, |r| {
                    for p in r {
                        let q = p + s;
                        flux[p] = w * (c[p] - c[q]) - dz * (c[p] + c[q]) * drift[p];
                    }
                });
                apply_face_fluxes(&g, a, &flux, &mut expected);
                for (got, want) in dc.iter().zip(&expected) {
                    assert!(
                        (got - want).abs() < 1e-14,
                        "{n:?}, axis {a}: {got} vs {want}"
                    );
                }
                // Nothing moves along a single-cell axis; otherwise something must.
                assert_eq!(
                    dc.iter().any(|v| v.abs() > 1e-3),
                    n[a] > 1,
                    "{n:?}, axis {a}"
                );
            }
        }
    }
}
