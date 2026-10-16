//! Conforming tetrahedral meshes of a box containing axis-aligned cells.
//!
//! The box is cut into a rectilinear grid whose planes include every cell face, and each
//! hexahedron is split into six Kuhn (path) tetrahedra. All hexes use the same split, so the
//! tetrahedra are conforming across hex faces, and every cell face is a union of mesh facets.
//!
//! Each subdomain (the extracellular space, tag 0, and each cell, tag `1..=n`) has its own
//! P1 degrees of freedom. A grid vertex on a membrane therefore carries two DOFs, one per
//! side, as in the mortar scheme of Publication 2 where Ω_i and Ω_e have separate meshes with
//! matching facets on Γ. The interface space Q_h is P1 on the membrane facets; with a lumped
//! interface mass matrix its DOFs are the membrane vertices, stored in [`Mesh::membrane`].

use std::fmt;

/// An axis-aligned box `[lo, hi]` in µm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub lo: [f64; 3],
    pub hi: [f64; 3],
}

impl Aabb {
    pub fn new(lo: [f64; 3], hi: [f64; 3]) -> Self {
        Self { lo, hi }
    }

    fn contains(&self, p: [f64; 3]) -> bool {
        (0..3).all(|a| p[a] > self.lo[a] && p[a] < self.hi[a])
    }

    pub fn volume(&self) -> f64 {
        (0..3).map(|a| self.hi[a] - self.lo[a]).product()
    }

    pub fn surface_area(&self) -> f64 {
        let l = [0, 1, 2].map(|a| self.hi[a] - self.lo[a]);
        2.0 * (l[0] * l[1] + l[1] * l[2] + l[0] * l[2])
    }
}

/// Geometry to mesh: the extracellular box `[0, extent]` with cells inside it.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshSpec {
    /// Size of the whole domain Ω, µm.
    pub extent: [f64; 3],
    /// The cells Ω_i^1 … Ω_i^n. They must lie inside Ω and must not touch each other.
    pub cells: Vec<Aabb>,
    /// Largest grid spacing along each axis, µm.
    pub max_h: [f64; 3],
}

impl MeshSpec {
    /// A 3×3 bundle of square axons along x, as in Publication 2 (fig. 1): axons of side
    /// `side`, separated by `gap`, with `margin` of extracellular space around the bundle
    /// and `end_gap` between the axon ends and the box ends. Cell 5 (tag 5) is the centre.
    pub fn axon_bundle(length: f64, side: f64, gap: f64, margin: f64, end_gap: f64) -> Self {
        let mut cells = Vec::with_capacity(9);
        for iz in 0..3 {
            for iy in 0..3 {
                let y0 = margin + iy as f64 * (side + gap);
                let z0 = margin + iz as f64 * (side + gap);
                cells.push(Aabb::new(
                    [end_gap, y0, z0],
                    [length + end_gap, y0 + side, z0 + side],
                ));
            }
        }
        let width = 2.0 * margin + 3.0 * side + 2.0 * gap;
        Self {
            extent: [length + 2.0 * end_gap, width, width],
            cells,
            max_h: [4.0, 0.1, 0.1],
        }
    }

    /// The default bundle for the app: the paper's 3×3 cross-section with 200 µm axons and
    /// a coarse grid (5 µm along the axons, ≤ 0.3 µm across, which only splits the cell
    /// faces) that runs at about 20 simulated ms per wall second natively.
    pub fn default_bundle() -> Self {
        let mut spec = Self::axon_bundle(200.0, 0.2, 0.1, 0.3, 5.0);
        spec.max_h = [5.0, 0.3, 0.3];
        spec
    }

    /// The paper's bundle: 9 axons of 390 × 0.2 × 0.2 µm, 0.1 µm apart, in 400 × 1.4² µm.
    pub fn paper_bundle() -> Self {
        Self::axon_bundle(390.0, 0.2, 0.1, 0.3, 5.0)
    }
}

/// Why a [`MeshSpec`] could not be meshed.
#[derive(Debug, Clone, PartialEq)]
pub enum MeshError {
    /// A cell is empty or sticks out of the domain.
    CellOutsideDomain(usize),
    /// Two cells share a vertex; the KNP-EMI model needs extracellular space between cells.
    CellsTouch(usize, usize),
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CellOutsideDomain(c) => write!(f, "cell {c} is empty or outside the domain"),
            Self::CellsTouch(a, b) => write!(f, "cells {a} and {b} touch"),
        }
    }
}

impl std::error::Error for MeshError {}

/// No DOF.
pub const NONE: u32 = u32::MAX;

/// One membrane vertex: the P1 interface DOF and the bulk DOFs on either side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MembraneNode {
    /// DOF on the intracellular side.
    pub inner: u32,
    /// DOF on the extracellular side.
    pub outer: u32,
    /// Lumped membrane area (row sum of the interface mass matrix), µm².
    pub area: f64,
    /// Cell tag (≥ 1).
    pub cell: u16,
}

/// A tetrahedral mesh with per-subdomain P1 DOFs.
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Grid plane coordinates along each axis, µm. Vertex `(i, j, k)` has index
    /// `i + nx (j + ny k)` with `n = axes[a].len()` (x fastest).
    pub axes: [Vec<f64>; 3],
    /// Subdomain tag of each hex (0 = extracellular), x fastest.
    pub hex_tag: Vec<u16>,
    /// Number of cells.
    pub n_cells: usize,
    /// Tetrahedra as DOF indices; all four DOFs belong to the tet's subdomain.
    pub tets: Vec<[u32; 4]>,
    /// Subdomain tag of each tet.
    pub tet_tag: Vec<u16>,
    /// Grid vertex of each DOF.
    pub dof_vertex: Vec<u32>,
    /// Subdomain tag of each DOF.
    pub dof_tag: Vec<u16>,
    /// Extracellular DOF of each grid vertex, or [`NONE`].
    pub vertex_outer: Vec<u32>,
    /// Intracellular DOF of each grid vertex, or [`NONE`].
    pub vertex_inner: Vec<u32>,
    /// Lumped volume (row sum of the P1 mass matrix) of each DOF, µm³.
    pub lumped_volume: Vec<f64>,
    /// Membrane vertices.
    pub membrane: Vec<MembraneNode>,
}

/// Grid planes along one axis: domain ends and cell faces, refined to spacing ≤ `max_h`.
fn axis_planes(extent: f64, cells: &[Aabb], axis: usize, max_h: f64) -> Vec<f64> {
    let tol = 1e-9 * extent;
    let mut breaks = vec![0.0, extent];
    for c in cells {
        breaks.push(c.lo[axis]);
        breaks.push(c.hi[axis]);
    }
    breaks.sort_by(f64::total_cmp);
    breaks.dedup_by(|a, b| (*a - *b).abs() <= tol);
    let mut planes = vec![breaks[0]];
    for w in breaks.windows(2) {
        let n = ((w[1] - w[0]) / max_h - 1e-9).ceil().max(1.0) as usize;
        for s in 1..=n {
            planes.push(w[0] + (w[1] - w[0]) * s as f64 / n as f64);
        }
    }
    planes
}

/// The six Kuhn tetrahedra of a hex as local corner indices (bit 0 = x, 1 = y, 2 = z).
pub const KUHN_TETS: [[usize; 4]; 6] = {
    const PERMS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut tets = [[0usize; 4]; 6];
    let mut t = 0;
    while t < 6 {
        let p = PERMS[t];
        let a = 1 << p[0];
        let b = a | (1 << p[1]);
        tets[t] = [0, a, b, 7];
        t += 1;
    }
    tets
};

impl Mesh {
    /// Meshes `spec`. Panics with `u16::MAX` or more cells.
    pub fn build(spec: &MeshSpec) -> Result<Self, MeshError> {
        assert!(spec.cells.len() < u16::MAX as usize, "too many cells");
        for (c, cell) in spec.cells.iter().enumerate() {
            if (0..3).any(|a| {
                cell.lo[a] < 0.0 || cell.hi[a] > spec.extent[a] || cell.hi[a] <= cell.lo[a]
            }) {
                return Err(MeshError::CellOutsideDomain(c));
            }
        }
        let axes: [Vec<f64>; 3] =
            [0, 1, 2].map(|a| axis_planes(spec.extent[a], &spec.cells, a, spec.max_h[a]));
        let nv = [0, 1, 2].map(|a| axes[a].len());
        let nh = nv.map(|n| n - 1);
        let vid = |i: usize, j: usize, k: usize| i + nv[0] * (j + nv[1] * k);

        // Tag hexes by the cell containing their centre.
        let mut hex_tag = vec![0u16; nh[0] * nh[1] * nh[2]];
        for k in 0..nh[2] {
            for j in 0..nh[1] {
                for i in 0..nh[0] {
                    let centre = [
                        0.5 * (axes[0][i] + axes[0][i + 1]),
                        0.5 * (axes[1][j] + axes[1][j + 1]),
                        0.5 * (axes[2][k] + axes[2][k + 1]),
                    ];
                    if let Some(c) = spec.cells.iter().position(|c| c.contains(centre)) {
                        hex_tag[i + nh[0] * (j + nh[1] * k)] = c as u16 + 1;
                    }
                }
            }
        }

        // One DOF per (vertex, adjacent subdomain).
        let n_vertices = nv[0] * nv[1] * nv[2];
        let mut vertex_outer = vec![NONE; n_vertices];
        let mut vertex_inner = vec![NONE; n_vertices];
        let mut vertex_cell = vec![0u16; n_vertices];
        let mut dof_vertex = Vec::new();
        let mut dof_tag = Vec::new();
        for k in 0..nv[2] {
            for j in 0..nv[1] {
                for i in 0..nv[0] {
                    let v = vid(i, j, k);
                    for (di, dj, dk) in corner_offsets() {
                        let (Some(hi), Some(hj), Some(hk)) =
                            (i.checked_sub(di), j.checked_sub(dj), k.checked_sub(dk))
                        else {
                            continue;
                        };
                        if hi >= nh[0] || hj >= nh[1] || hk >= nh[2] {
                            continue;
                        }
                        let tag = hex_tag[hi + nh[0] * (hj + nh[1] * hk)];
                        let slot = if tag == 0 {
                            &mut vertex_outer[v]
                        } else {
                            let prev = vertex_cell[v];
                            if prev != 0 && prev != tag {
                                let (a, b) = (prev.min(tag), prev.max(tag));
                                return Err(MeshError::CellsTouch(a as usize - 1, b as usize - 1));
                            }
                            vertex_cell[v] = tag;
                            &mut vertex_inner[v]
                        };
                        if *slot == NONE {
                            *slot = dof_vertex.len() as u32;
                            dof_vertex.push(v as u32);
                            dof_tag.push(tag);
                        }
                    }
                }
            }
        }

        // Kuhn tetrahedra and lumped volumes.
        let n_dofs = dof_vertex.len();
        let mut tets = Vec::with_capacity(6 * hex_tag.len());
        let mut tet_tag = Vec::with_capacity(6 * hex_tag.len());
        let mut lumped_volume = vec![0.0; n_dofs];
        for k in 0..nh[2] {
            for j in 0..nh[1] {
                for i in 0..nh[0] {
                    let tag = hex_tag[i + nh[0] * (j + nh[1] * k)];
                    let map = if tag == 0 {
                        &vertex_outer
                    } else {
                        &vertex_inner
                    };
                    let corner = |c: usize| map[vid(i + (c & 1), j + (c >> 1 & 1), k + (c >> 2))];
                    let vol = (axes[0][i + 1] - axes[0][i])
                        * (axes[1][j + 1] - axes[1][j])
                        * (axes[2][k + 1] - axes[2][k])
                        / 6.0;
                    for t in KUHN_TETS {
                        let tet = t.map(corner);
                        for &d in &tet {
                            lumped_volume[d as usize] += 0.25 * vol;
                        }
                        tets.push(tet);
                        tet_tag.push(tag);
                    }
                }
            }
        }

        // Membrane facets: hex faces between a cell and the ECS. The Kuhn split cuts every
        // face along the diagonal from its lowest to its highest corner, so the lumped P1
        // interface mass gives those two corners area/3 and the other two area/6.
        let mut vertex_area = vec![0.0; n_vertices];
        for axis in 0..3 {
            let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
            for k in 0..nh[2] {
                for j in 0..nh[1] {
                    for i in 0..nh[0] {
                        let mut idx = [i, j, k];
                        idx[axis] += 1;
                        if idx[axis] >= nh[axis] {
                            continue;
                        }
                        let a = hex_tag[i + nh[0] * (j + nh[1] * k)];
                        let b = hex_tag[idx[0] + nh[0] * (idx[1] + nh[1] * idx[2])];
                        if (a == 0) == (b == 0) {
                            continue;
                        }
                        let mut lo = [i, j, k];
                        lo[axis] += 1;
                        let area = (axes[u][lo[u] + 1] - axes[u][lo[u]])
                            * (axes[w][lo[w] + 1] - axes[w][lo[w]]);
                        for (du, dw, share) in [(0, 0, 3.0), (1, 1, 3.0), (1, 0, 6.0), (0, 1, 6.0)]
                        {
                            let mut p = lo;
                            p[u] += du;
                            p[w] += dw;
                            vertex_area[vid(p[0], p[1], p[2])] += area / share;
                        }
                    }
                }
            }
        }
        let membrane = (0..n_vertices)
            .filter(|&v| vertex_area[v] > 0.0)
            .map(|v| MembraneNode {
                inner: vertex_inner[v],
                outer: vertex_outer[v],
                area: vertex_area[v],
                cell: vertex_cell[v],
            })
            .collect();

        Ok(Self {
            axes,
            hex_tag,
            n_cells: spec.cells.len(),
            tets,
            tet_tag,
            dof_vertex,
            dof_tag,
            vertex_outer,
            vertex_inner,
            lumped_volume,
            membrane,
        })
    }

    /// Number of DOFs per scalar field (all subdomains together).
    pub fn n_dofs(&self) -> usize {
        self.dof_vertex.len()
    }

    /// Grid vertex counts `[nx, ny, nz]`.
    pub fn vertex_dims(&self) -> [usize; 3] {
        [0, 1, 2].map(|a| self.axes[a].len())
    }

    /// Coordinates of grid vertex `v`, µm.
    pub fn vertex_position(&self, v: usize) -> [f64; 3] {
        let [nx, ny, _] = self.vertex_dims();
        [
            self.axes[0][v % nx],
            self.axes[1][(v / nx) % ny],
            self.axes[2][v / (nx * ny)],
        ]
    }

    /// Coordinates of the four vertices of tet `t`, µm.
    pub fn tet_points(&self, t: usize) -> [[f64; 3]; 4] {
        self.tets[t].map(|d| self.vertex_position(self.dof_vertex[d as usize] as usize))
    }
}

/// Offsets from a vertex to the low corners of its (up to) eight adjacent hexes.
fn corner_offsets() -> impl Iterator<Item = (usize, usize, usize)> {
    (0..8).map(|c| (c & 1, c >> 1 & 1, c >> 2))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_cell() -> MeshSpec {
        MeshSpec {
            extent: [4.0, 2.0, 2.0],
            cells: vec![Aabb::new([1.0, 0.5, 0.5], [3.0, 1.5, 1.5])],
            max_h: [0.5, 0.25, 0.25],
        }
    }

    fn domain_volume(mesh: &Mesh, tag: u16) -> f64 {
        (0..mesh.n_dofs())
            .filter(|&d| mesh.dof_tag[d] == tag)
            .map(|d| mesh.lumped_volume[d])
            .sum()
    }

    #[test]
    fn volumes_and_membrane_area_are_exact() {
        let spec = MeshSpec::paper_bundle();
        let mesh = Mesh::build(&spec).unwrap();
        let total: f64 = spec.extent.iter().product();
        let cells: f64 = spec.cells.iter().map(Aabb::volume).sum();
        assert!((domain_volume(&mesh, 0) - (total - cells)).abs() < 1e-9 * total);
        for (c, cell) in spec.cells.iter().enumerate() {
            let v = domain_volume(&mesh, c as u16 + 1);
            assert!((v - cell.volume()).abs() < 1e-9 * cell.volume());
            let area: f64 = mesh
                .membrane
                .iter()
                .filter(|m| m.cell == c as u16 + 1)
                .map(|m| m.area)
                .sum();
            assert!((area - cell.surface_area()).abs() < 1e-9 * area, "cell {c}");
        }
    }

    #[test]
    fn membrane_vertices_have_one_dof_per_side() {
        let mesh = Mesh::build(&single_cell()).unwrap();
        for m in &mesh.membrane {
            assert_eq!(mesh.dof_tag[m.inner as usize], 1);
            assert_eq!(mesh.dof_tag[m.outer as usize], 0);
            assert_eq!(
                mesh.dof_vertex[m.inner as usize],
                mesh.dof_vertex[m.outer as usize]
            );
        }
        // Every vertex on the cell boundary is a membrane vertex: (5×5×5) − (3×3×3).
        assert_eq!(mesh.membrane.len(), 125 - 27);
        let duplicated = (0..mesh.vertex_outer.len())
            .filter(|&v| mesh.vertex_outer[v] != NONE && mesh.vertex_inner[v] != NONE)
            .count();
        assert_eq!(duplicated, mesh.membrane.len());
    }

    #[test]
    fn tets_have_positive_volume_and_stay_in_their_subdomain() {
        let mesh = Mesh::build(&single_cell()).unwrap();
        for (t, tet) in mesh.tets.iter().enumerate() {
            assert!(tet
                .iter()
                .all(|&d| mesh.dof_tag[d as usize] == mesh.tet_tag[t]));
            let p = mesh.tet_points(t);
            let e = [1, 2, 3].map(|i| [0, 1, 2].map(|a| p[i][a] - p[0][a]));
            let det = e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1])
                - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0])
                + e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
            assert!(det.abs() > 1e-12);
        }
    }

    #[test]
    fn grid_planes_include_cell_faces() {
        let mesh = Mesh::build(&MeshSpec::paper_bundle()).unwrap();
        for y in [0.3, 0.5, 0.6, 0.8, 0.9, 1.1] {
            assert!(
                mesh.axes[1].iter().any(|&p| (p - y).abs() < 1e-12),
                "y = {y}"
            );
        }
        assert!(mesh.axes[1].windows(2).all(|w| w[1] - w[0] <= 0.1 + 1e-12));
        assert_eq!(mesh.n_cells, 9);
    }

    #[test]
    fn touching_or_outside_cells_are_rejected() {
        let mut spec = single_cell();
        spec.cells.push(Aabb::new([3.0, 0.5, 0.5], [3.5, 1.5, 1.5]));
        assert_eq!(Mesh::build(&spec).unwrap_err(), MeshError::CellsTouch(0, 1));
        spec.cells[1] = Aabb::new([3.5, 0.5, 0.5], [4.5, 1.5, 1.5]);
        assert_eq!(
            Mesh::build(&spec).unwrap_err(),
            MeshError::CellOutsideDomain(1)
        );
    }
}
