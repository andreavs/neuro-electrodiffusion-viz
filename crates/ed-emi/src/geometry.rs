//! Voxel geometry: a grid whose voxels are tagged with a cell id, and the membrane faces
//! between the cells and the extracellular space.

use ed_core::linalg::Grid;

/// Tag of extracellular voxels.
pub const ECS: u16 = 0;

/// A face between an intracellular voxel and an extracellular voxel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MembraneFace {
    /// Intracellular voxel.
    pub inner: usize,
    /// Extracellular voxel.
    pub outer: usize,
    /// Axis the face is normal to.
    pub axis: u8,
    /// Cell id of the intracellular voxel.
    pub cell: u16,
}

/// A voxel grid with one cell id per voxel ([`ECS`] = extracellular).
#[derive(Clone, Debug)]
pub struct Geometry {
    pub grid: Grid,
    /// Cell id per voxel, `x` fastest.
    pub tags: Vec<u16>,
}

impl Geometry {
    /// Wraps a tagged grid. Each cell must be connected and so must the extracellular
    /// space; faces between two different cells are treated as insulating.
    pub fn new(grid: Grid, tags: Vec<u16>) -> Self {
        assert_eq!(grid.dim, 3, "KNP-EMI geometries are 3-D");
        assert_eq!(tags.len(), grid.len(), "one tag per voxel");
        assert!(tags.contains(&ECS), "geometry needs extracellular space");
        Self { grid, tags }
    }

    /// Number of cells (largest cell id).
    pub fn cell_count(&self) -> usize {
        self.tags.iter().copied().max().unwrap_or(0) as usize
    }

    /// Calls `f(axis, lower, upper)` for every interior face of the grid.
    pub fn for_each_face(&self, mut f: impl FnMut(usize, usize, usize)) {
        let g = &self.grid;
        let stride = [1, g.n[0], g.n[0] * g.n[1]];
        g.for_each_cell(|i, j, k, c| {
            let pos = [i, j, k];
            for a in 0..3 {
                if pos[a] + 1 < g.n[a] {
                    f(a, c, c + stride[a]);
                }
            }
        });
    }

    /// All membrane faces, in storage order of their lower voxel.
    pub fn membrane_faces(&self) -> Vec<MembraneFace> {
        let mut faces = Vec::new();
        self.for_each_face(|axis, lo, hi| {
            let (tl, th) = (self.tags[lo], self.tags[hi]);
            if (tl == ECS) == (th == ECS) {
                return;
            }
            let (inner, outer) = if tl == ECS { (hi, lo) } else { (lo, hi) };
            faces.push(MembraneFace {
                inner,
                outer,
                axis: axis as u8,
                cell: self.tags[inner],
            });
        });
        faces
    }
}

/// A bundle of rectangular axons running along `z` inside an extracellular box, laid out
/// on a `rows × cols` lattice in the `xy`-plane. Lengths are in µm and are rounded to
/// whole voxels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxonBundle {
    pub rows: usize,
    pub cols: usize,
    /// Side of the square axon cross-section.
    pub side: f64,
    /// Extracellular gap between neighbouring axons.
    pub gap: f64,
    /// Extracellular margin between the outer axons and the box walls.
    pub margin: f64,
    /// Axon length.
    pub length: f64,
    /// Extracellular space beyond each axon end.
    pub end_gap: f64,
    /// Voxel size across the axons (x and y).
    pub h_xy: f64,
    /// Voxel size along the axons (z).
    pub h_z: f64,
}

impl Default for AxonBundle {
    /// A 3×3 bundle of 1 µm axons, 150 µm long, on a coarse grid that runs in real time.
    fn default() -> Self {
        Self {
            rows: 3,
            cols: 3,
            side: 1.0,
            gap: 1.0,
            margin: 1.0,
            length: 150.0,
            end_gap: 5.0,
            h_xy: 1.0,
            h_z: 5.0,
        }
    }
}

fn voxels(len: f64, h: f64, what: &str) -> usize {
    let n = (len / h).round();
    assert!(
        n >= 1.0,
        "{what} ({len} µm) must span at least one voxel of {h} µm"
    );
    n as usize
}

impl AxonBundle {
    /// Builds the voxel geometry. Axon ids run `1..=rows·cols`, row-major from the
    /// lower corner.
    pub fn build(&self) -> Geometry {
        assert!(self.rows > 0 && self.cols > 0, "need at least one axon");
        let side = voxels(self.side, self.h_xy, "axon side");
        let gap = voxels(self.gap, self.h_xy, "gap");
        let margin = voxels(self.margin, self.h_xy, "margin");
        let length = voxels(self.length, self.h_z, "axon length");
        let end = voxels(self.end_gap, self.h_z, "end gap");
        let span = |m: usize| 2 * margin + m * side + (m - 1) * gap;
        let n = [span(self.cols), span(self.rows), length + 2 * end];
        let grid = Grid::new_3d(n, [self.h_xy, self.h_xy, self.h_z]);
        // Lattice slot (0-based) of a transverse voxel index, if inside an axon.
        let slot = |v: usize, m: usize| {
            let v = v.checked_sub(margin)?;
            let (s, r) = (v / (side + gap), v % (side + gap));
            (s < m && r < side).then_some(s)
        };
        let mut tags = vec![ECS; grid.len()];
        grid.for_each_cell(|i, j, k, c| {
            if let (Some(col), Some(row), true) = (
                slot(i, self.cols),
                slot(j, self.rows),
                (end..end + length).contains(&k),
            ) {
                tags[c] = (1 + row * self.cols + col) as u16;
            }
        });
        Geometry::new(grid, tags)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_bundle_has_nine_axons_with_expected_membrane_area() {
        let b = AxonBundle::default();
        let g = b.build();
        assert_eq!(g.grid.n, [7, 7, 32]);
        assert_eq!(g.cell_count(), 9);
        let faces = g.membrane_faces();
        // Each 1×1-voxel axon of 30 voxels: 4 sides × 30 plus 2 end caps.
        assert_eq!(faces.len(), 9 * (4 * 30 + 2));
        for f in &faces {
            assert_ne!(g.tags[f.inner], ECS);
            assert_eq!(g.tags[f.outer], ECS);
            assert_eq!(g.tags[f.inner], f.cell);
        }
        let axon_voxels = g.tags.iter().filter(|&&t| t != ECS).count();
        assert_eq!(axon_voxels, 9 * 30);
    }

    #[test]
    fn finer_bundle_resolves_cross_sections() {
        let b = AxonBundle {
            rows: 1,
            cols: 2,
            h_xy: 0.5,
            h_z: 2.0,
            length: 20.0,
            end_gap: 4.0,
            ..Default::default()
        };
        let g = b.build();
        // margin 2 + side 2 + gap 2 + side 2 + margin 2 voxels.
        assert_eq!(g.grid.n, [10, 6, 14]);
        let faces = g.membrane_faces();
        // Per axon: 4 sides × 2 voxels × 10 + 2 caps × 4 voxels.
        assert_eq!(faces.len(), 2 * (4 * 2 * 10 + 2 * 4));
        assert!(faces.iter().any(|f| f.cell == 2));
    }
}
