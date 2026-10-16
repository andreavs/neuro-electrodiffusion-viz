//! Voxel geometry: a grid whose voxels are tagged with a cell id (0 = extracellular), and
//! the membrane faces between differently tagged voxels.

use ed_core::linalg::{Face, Grid};

/// Tag of extracellular voxels.
pub const ECS: u32 = 0;

/// A tagged voxel grid. Axis 0 (x) is the axon axis for [`AxonBundle`] geometries.
#[derive(Clone, Debug)]
pub struct VoxelGeometry {
    pub grid: Grid,
    /// Cell id per voxel, `0` = extracellular space, x fastest.
    pub tags: Vec<u32>,
}

/// A membrane patch: the face between an intracellular and an extracellular voxel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MembraneFace {
    /// The grid face (between `face.cell` and its upper neighbour along `face.axis`).
    pub face: Face,
    /// Intracellular voxel.
    pub inside: usize,
    /// Extracellular voxel.
    pub outside: usize,
    /// Cell id of the intracellular voxel.
    pub cell: u32,
}

impl MembraneFace {
    /// Centre of the face, µm.
    pub fn center(&self, grid: &Grid) -> [f64; 3] {
        let g = grid;
        let c = self.face.cell;
        let (i, j, k) = (c % g.n[0], (c / g.n[0]) % g.n[1], c / (g.n[0] * g.n[1]));
        let mut p = g.cell_center(i, j, k);
        p[self.face.axis] += 0.5 * g.h[self.face.axis];
        p
    }
}

impl VoxelGeometry {
    /// Wraps a tag array; `tags.len()` must equal `grid.len()`.
    pub fn new(grid: Grid, tags: Vec<u32>) -> Self {
        assert_eq!(tags.len(), grid.len(), "one tag per voxel");
        assert_eq!(grid.dim, 3, "KNP-EMI geometries are 3-D");
        Self { grid, tags }
    }

    /// Calls `f(face, lower, upper)` for every interior face whose voxels have different tags.
    pub fn for_each_interface(&self, mut f: impl FnMut(Face, usize, usize)) {
        let g = &self.grid;
        let stride = [1, g.n[0], g.n[0] * g.n[1]];
        g.for_each_cell(|i, j, k, cell| {
            let pos = [i, j, k];
            for axis in 0..3 {
                if pos[axis] + 1 < g.n[axis] {
                    let nb = cell + stride[axis];
                    if self.tags[cell] != self.tags[nb] {
                        f(Face { axis, cell }, cell, nb);
                    }
                }
            }
        });
    }

    /// Membrane faces between a cell and the extracellular space, in storage order.
    ///
    /// Faces between two different cells are not membranes in this model: they are
    /// treated as impermeable and get no patch (see [`cell_contacts`](Self::cell_contacts)).
    pub fn membrane_faces(&self) -> Vec<MembraneFace> {
        let mut out = Vec::new();
        self.for_each_interface(|face, lo, hi| {
            let (inside, outside) = match (self.tags[lo], self.tags[hi]) {
                (ECS, _) => (hi, lo),
                (_, ECS) => (lo, hi),
                _ => return,
            };
            out.push(MembraneFace {
                face,
                inside,
                outside,
                cell: self.tags[inside],
            });
        });
        out
    }

    /// Faces where two different cells touch directly.
    pub fn cell_contacts(&self) -> Vec<Face> {
        let mut out = Vec::new();
        self.for_each_interface(|face, lo, hi| {
            if self.tags[lo] != ECS && self.tags[hi] != ECS {
                out.push(face);
            }
        });
        out
    }

    /// Number of voxels per tag, indexed by tag.
    pub fn voxel_counts(&self) -> Vec<usize> {
        let max = self.tags.iter().copied().max().unwrap_or(0) as usize;
        let mut counts = vec![0; max + 1];
        for &t in &self.tags {
            counts[t as usize] += 1;
        }
        counts
    }

    /// Area of a face along `axis`, µm².
    pub fn face_area(&self, axis: usize) -> f64 {
        let h = self.grid.h;
        h[(axis + 1) % 3] * h[(axis + 2) % 3]
    }

    /// Voxel volume, µm³.
    pub fn voxel_volume(&self) -> f64 {
        self.grid.h.iter().product()
    }
}

/// A bundle of rectangular axons along x, arranged on a `rows × cols` lattice in the y-z
/// plane, inside an extracellular box. Lengths are in µm and are rounded to whole voxels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxonBundle {
    /// Number of axons; they fill a near-square lattice row by row.
    pub count: usize,
    /// Side of the square axon cross-section.
    pub side: f64,
    /// Extracellular gap between neighbouring axons.
    pub gap: f64,
    /// Axon length.
    pub length: f64,
    /// Extracellular margin between the bundle and the box, laterally.
    pub margin: f64,
    /// Extracellular margin beyond each axon end.
    pub end_margin: f64,
    /// Voxel size across the axons (y and z).
    pub h: f64,
    /// Voxel size along the axons (x).
    pub hx: f64,
}

impl Default for AxonBundle {
    /// Nine 1 µm axons, 120 µm long, at 1 µm resolution across and 2 µm along the axons.
    fn default() -> Self {
        Self {
            count: 9,
            side: 1.0,
            gap: 1.0,
            length: 120.0,
            margin: 1.0,
            end_margin: 4.0,
            h: 1.0,
            hx: 2.0,
        }
    }
}

/// An axon's voxel extent in a built [`AxonBundle`]: half-open index ranges per axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AxonExtent {
    pub lo: [usize; 3],
    pub hi: [usize; 3],
}

impl AxonBundle {
    /// Lattice shape `(rows, cols)` used for `count` axons.
    pub fn lattice(&self) -> (usize, usize) {
        let cols = (self.count as f64).sqrt().ceil().max(1.0) as usize;
        (self.count.div_ceil(cols), cols)
    }

    /// Builds the voxel geometry. Axon `a` gets tag `a + 1`. Cell counts are rounded up to
    /// `2^m` or `3·2^m` (adding extracellular margin) so multigrid can coarsen well.
    pub fn build(&self) -> (VoxelGeometry, Vec<AxonExtent>) {
        assert!(self.count > 0, "need at least one axon");
        assert!(
            self.h > 0.0 && self.hx > 0.0,
            "voxel sizes must be positive"
        );
        let vox = |len: f64, h: f64| (len / h).round() as usize;
        let side = vox(self.side, self.h).max(1);
        let gap = vox(self.gap, self.h).max(1);
        let margin = vox(self.margin, self.h).max(1);
        let len = vox(self.length, self.hx).max(1);
        let end = vox(self.end_margin, self.hx).max(1);
        let (rows, cols) = self.lattice();
        let span_y = cols * side + (cols - 1) * gap;
        let span_z = rows * side + (rows - 1) * gap;
        let ny = multigrid_size(span_y + 2 * margin);
        let nz = multigrid_size(span_z + 2 * margin);
        let nx = multigrid_size(len + 2 * end);
        // Centre the bundle in the (possibly enlarged) box.
        let (oy, oz, ox) = ((ny - span_y) / 2, (nz - span_z) / 2, (nx - len) / 2);
        let grid = Grid::new_3d([nx, ny, nz], [self.hx, self.h, self.h]);
        let mut tags = vec![ECS; grid.len()];
        let extents: Vec<AxonExtent> = (0..self.count)
            .map(|a| {
                let (r, c) = (a / cols, a % cols);
                let y0 = oy + c * (side + gap);
                let z0 = oz + r * (side + gap);
                AxonExtent {
                    lo: [ox, y0, z0],
                    hi: [ox + len, y0 + side, z0 + side],
                }
            })
            .collect();
        for (a, e) in extents.iter().enumerate() {
            for k in e.lo[2]..e.hi[2] {
                for j in e.lo[1]..e.hi[1] {
                    for i in e.lo[0]..e.hi[0] {
                        tags[grid.index(i, j, k)] = a as u32 + 1;
                    }
                }
            }
        }
        (VoxelGeometry::new(grid, tags), extents)
    }
}

/// Smallest `n' ≥ n` of the form `2^m` or `3·2^m`.
fn multigrid_size(n: usize) -> usize {
    let p = n.max(1).next_power_of_two();
    if 3 * p / 4 >= n && p >= 4 {
        3 * p / 4
    } else {
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_layout_and_membrane_area() {
        let bundle = AxonBundle::default();
        let (geo, axons) = bundle.build();
        assert_eq!(axons.len(), 9);
        assert_eq!(geo.grid.n, [64, 12, 12]);
        assert_eq!(
            [7, 9, 12, 13, 24, 25].map(multigrid_size),
            [8, 12, 12, 16, 24, 32]
        );
        let counts = geo.voxel_counts();
        assert_eq!(counts.len(), 10);
        assert!(counts[1..].iter().all(|&c| c == 60));
        assert!(geo.cell_contacts().is_empty());
        // Each axon: 4 sides × 60 voxels of 2 µm² plus two 1 µm² end caps.
        let faces = geo.membrane_faces();
        let area: f64 = faces.iter().map(|f| geo.face_area(f.face.axis)).sum();
        assert!(
            (area - 9.0 * (4.0 * 120.0 + 2.0)).abs() < 1e-9,
            "area {area}"
        );
        for f in &faces {
            assert_ne!(geo.tags[f.inside], ECS);
            assert_eq!(geo.tags[f.outside], ECS);
            assert_eq!(geo.tags[f.inside], f.cell);
        }
    }

    #[test]
    fn touching_cells_are_contacts_not_membranes() {
        let grid = Grid::new_3d([4, 2, 2], [1.0; 3]);
        let mut tags = vec![ECS; grid.len()];
        tags[grid.index(1, 0, 0)] = 1;
        tags[grid.index(2, 0, 0)] = 2;
        let geo = VoxelGeometry::new(grid, tags);
        assert_eq!(geo.cell_contacts().len(), 1);
        // Each cell voxel touches ECS across one x-face, one y-face and one z-face.
        assert_eq!(geo.membrane_faces().len(), 6);
    }
}
