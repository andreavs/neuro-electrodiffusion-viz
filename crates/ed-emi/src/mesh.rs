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

use std::collections::HashMap;
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

    /// The app's default: the paper's cross-section with 200 µm axons on a coarse grid
    /// (5 µm along the axons, ≤ 0.3 µm across, which only splits the cell faces).
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
    /// `extent` or `max_h` is not positive and finite.
    BadSpacing,
    /// A cell is empty or sticks out of the domain.
    CellOutsideDomain(usize),
    /// A cell got no hexes: it lies inside another cell or is thinner than the snapping tolerance.
    EmptyCell(usize),
    /// Two cells share a vertex or overlap; KNP-EMI needs extracellular space between cells.
    CellsTouch(usize, usize),
    /// A tet has a vertex index out of range, a tag above the cell count, or no volume.
    BadTet(usize),
    /// The number of tet tags differs from the number of tets.
    TagCount,
    /// A face of this tet on a membrane is shared by more than two tets.
    NonConforming(usize),
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadSpacing => write!(f, "extent and max_h must be positive and finite"),
            Self::CellOutsideDomain(c) => write!(f, "cell {c} is empty or outside the domain"),
            Self::EmptyCell(c) => write!(f, "cell {c} has no hexes"),
            Self::CellsTouch(a, b) => write!(f, "cells {a} and {b} touch"),
            Self::BadTet(t) => write!(f, "tet {t} is invalid or flat"),
            Self::TagCount => write!(f, "need one tag per tet"),
            Self::NonConforming(t) => write!(f, "a membrane face of tet {t} has over two tets"),
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

/// The rectilinear grid a [`Mesh`] was built on by [`Mesh::build`].
#[derive(Debug, Clone)]
pub struct Grid {
    /// Plane coordinates along each axis, µm. Vertex `(i, j, k)` has index
    /// `i + nx (j + ny k)` with `n = axes[a].len()` (x fastest).
    pub axes: [Vec<f64>; 3],
    /// Subdomain tag of each hex (0 = extracellular), x fastest.
    pub hex_tag: Vec<u16>,
}

impl Grid {
    /// Vertex counts `[nx, ny, nz]`.
    pub fn vertex_dims(&self) -> [usize; 3] {
        [0, 1, 2].map(|a| self.axes[a].len())
    }
}

/// A tetrahedral mesh with per-subdomain P1 DOFs.
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Vertex coordinates, µm.
    pub points: Vec<[f64; 3]>,
    /// The grid, for meshes built from a [`MeshSpec`]; its vertices are `points`.
    pub grid: Option<Grid>,
    /// Number of cells.
    pub n_cells: usize,
    /// Tetrahedra as DOF indices; all four DOFs belong to the tet's subdomain.
    pub tets: Vec<[u32; 4]>,
    /// Subdomain tag of each tet.
    pub tet_tag: Vec<u16>,
    /// Vertex of each DOF.
    pub dof_vertex: Vec<u32>,
    /// Subdomain tag of each DOF.
    pub dof_tag: Vec<u16>,
    /// Extracellular DOF of each vertex, or [`NONE`]. Cells may touch the domain
    /// boundary, so outer boundary conditions must go through this map, not vertex position.
    pub vertex_outer: Vec<u32>,
    /// Intracellular DOF of each vertex, or [`NONE`].
    pub vertex_inner: Vec<u32>,
    /// Lumped volume (row sum of the P1 mass matrix) of each DOF, µm³.
    pub lumped_volume: Vec<f64>,
    /// Membrane vertices.
    pub membrane: Vec<MembraneNode>,
    /// Membrane triangles as indices into `membrane`, counter-clockwise seen from outside
    /// the cell.
    pub membrane_facets: Vec<[u32; 3]>,
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
    *breaks.last_mut().unwrap() = extent;
    let mut planes = vec![breaks[0]];
    for w in breaks.windows(2) {
        let n = ((w[1] - w[0]) / max_h - 1e-9).ceil().max(1.0) as usize;
        for s in 1..n {
            planes.push(w[0] + (w[1] - w[0]) * s as f64 / n as f64);
        }
        planes.push(w[1]);
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

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl Mesh {
    /// Meshes `spec` on a rectilinear grid. Panics with `u16::MAX` or more cells.
    pub fn build(spec: &MeshSpec) -> Result<Self, MeshError> {
        assert!(spec.cells.len() < u16::MAX as usize, "too many cells");
        let positive = |x: f64| x.is_finite() && x > 0.0;
        if !spec.extent.into_iter().chain(spec.max_h).all(positive) {
            return Err(MeshError::BadSpacing);
        }
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
        let vid = |i: usize, j: usize, k: usize| (i + nv[0] * (j + nv[1] * k)) as u32;

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
        let mut has_hex = vec![false; spec.cells.len() + 1];
        hex_tag.iter().for_each(|&t| has_hex[t as usize] = true);
        if let Some(c) = (1..has_hex.len()).find(|&c| !has_hex[c]) {
            return Err(MeshError::EmptyCell(c - 1));
        }

        let mut points = Vec::with_capacity(nv[0] * nv[1] * nv[2]);
        for k in 0..nv[2] {
            for j in 0..nv[1] {
                for i in 0..nv[0] {
                    points.push([axes[0][i], axes[1][j], axes[2][k]]);
                }
            }
        }
        let mut tets = Vec::with_capacity(6 * hex_tag.len());
        let mut tet_tag = Vec::with_capacity(6 * hex_tag.len());
        for k in 0..nh[2] {
            for j in 0..nh[1] {
                for i in 0..nh[0] {
                    let corner = |c: usize| vid(i + (c & 1), j + (c >> 1 & 1), k + (c >> 2));
                    for t in KUHN_TETS {
                        tets.push(t.map(corner));
                        tet_tag.push(hex_tag[i + nh[0] * (j + nh[1] * k)]);
                    }
                }
            }
        }
        let mut mesh = Self::from_tets(points, &tets, tet_tag, spec.cells.len())?;
        mesh.grid = Some(Grid { axes, hex_tag });
        Ok(mesh)
    }

    /// Builds the per-subdomain DOFs and the membrane of a conforming tetrahedral mesh.
    /// `tets` index into `points` (µm); `tet_tag` is 0 for the extracellular space and
    /// `1..=n_cells` for the cells. Membrane facets are the triangles shared by a cell tet
    /// and an extracellular tet; cell faces on the outside of the mesh are sealed. The mesh
    /// must be conforming; a hanging node on a membrane silently seals that part of it.
    pub fn from_tets(
        points: Vec<[f64; 3]>,
        tets: &[[u32; 4]],
        tet_tag: Vec<u16>,
        n_cells: usize,
    ) -> Result<Self, MeshError> {
        assert!(n_cells < u16::MAX as usize, "too many cells");
        let n_vertices = points.len();
        if tet_tag.len() != tets.len() {
            return Err(MeshError::TagCount);
        }
        // Reject out-of-range indices, bad tags and flat tets.
        let mut has_tet = vec![false; n_cells + 1];
        for (t, (tet, &tag)) in tets.iter().zip(&tet_tag).enumerate() {
            if tag as usize > n_cells || tet.iter().any(|&v| v as usize >= n_vertices) {
                return Err(MeshError::BadTet(t));
            }
            let p = tet.map(|v| points[v as usize]);
            let e = [1, 2, 3].map(|i| sub(p[i], p[0]));
            let vol = dot(e[0], cross(e[1], e[2])).abs();
            let len2 = e.iter().map(|x| dot(*x, *x)).fold(0.0, f64::max);
            if vol.is_nan() || vol <= 1e-12 * len2.powf(1.5) {
                return Err(MeshError::BadTet(t));
            }
            has_tet[tag as usize] = true;
        }
        if let Some(c) = (1..has_tet.len()).find(|&c| !has_tet[c]) {
            return Err(MeshError::EmptyCell(c - 1));
        }

        // One DOF per (vertex, adjacent subdomain), numbered vertex by vertex.
        let mut uses_outer = vec![false; n_vertices];
        let mut vertex_cell = vec![0u16; n_vertices];
        for (tet, &tag) in tets.iter().zip(&tet_tag) {
            for &v in tet {
                let v = v as usize;
                if tag == 0 {
                    uses_outer[v] = true;
                    continue;
                }
                let prev = vertex_cell[v];
                if prev != 0 && prev != tag {
                    let (a, b) = (prev.min(tag), prev.max(tag));
                    return Err(MeshError::CellsTouch(a as usize - 1, b as usize - 1));
                }
                vertex_cell[v] = tag;
            }
        }
        let mut vertex_outer = vec![NONE; n_vertices];
        let mut vertex_inner = vec![NONE; n_vertices];
        let mut dof_vertex = Vec::new();
        let mut dof_tag = Vec::new();
        for v in cuthill_mckee(n_vertices, tets) {
            if uses_outer[v] {
                vertex_outer[v] = dof_vertex.len() as u32;
                dof_vertex.push(v as u32);
                dof_tag.push(0);
            }
            if vertex_cell[v] != 0 {
                vertex_inner[v] = dof_vertex.len() as u32;
                dof_vertex.push(v as u32);
                dof_tag.push(vertex_cell[v]);
            }
        }

        // Tets on DOFs, and lumped volumes.
        let mut lumped_volume = vec![0.0; dof_vertex.len()];
        let mut dof_tets = Vec::with_capacity(tets.len());
        for (tet, &tag) in tets.iter().zip(&tet_tag) {
            let map = if tag == 0 {
                &vertex_outer
            } else {
                &vertex_inner
            };
            let p = tet.map(|v| points[v as usize]);
            let e = [1, 2, 3].map(|i| sub(p[i], p[0]));
            let vol = dot(e[0], cross(e[1], e[2])).abs() / 6.0;
            let dofs = tet.map(|v| map[v as usize]);
            for &d in &dofs {
                lumped_volume[d as usize] += 0.25 * vol;
            }
            dof_tets.push(dofs);
        }

        // Membrane facets: triangles shared by a cell tet and an extracellular tet. The
        // lumped P1 interface mass gives each corner a third of the area. Only faces whose
        // corners all carry both DOFs can be on a membrane, which skips almost every face.
        let dual = |v: u32| uses_outer[v as usize] && vertex_cell[v as usize] != 0;
        let mut faces: HashMap<[u32; 3], (usize, usize, bool)> = HashMap::new();
        let mut facets = Vec::new();
        for (t, tet) in tets.iter().enumerate() {
            for skip in 0..4 {
                let mut f = [0, 1, 2].map(|i| tet[i + usize::from(i >= skip)]);
                if !f.iter().all(|&v| dual(v)) {
                    continue;
                }
                f.sort_unstable();
                match faces.get_mut(&f) {
                    None => {
                        faces.insert(f, (t, skip, false));
                    }
                    Some((_, _, true)) => return Err(MeshError::NonConforming(t)),
                    Some((u, uskip, paired)) => {
                        *paired = true;
                        if (tet_tag[t] == 0) != (tet_tag[*u] == 0) {
                            let (cell_tet, cell_skip) = if tet_tag[t] == 0 {
                                (*u, *uskip)
                            } else {
                                (t, skip)
                            };
                            facets.push((f, tets[cell_tet][cell_skip]));
                        }
                    }
                }
            }
        }
        let mut vertex_area = vec![0.0; n_vertices];
        let mut oriented = Vec::with_capacity(facets.len());
        for (mut f, apex) in facets {
            let p = f.map(|v| points[v as usize]);
            let mut n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
            // Counter-clockwise seen from outside: the normal points away from the cell.
            if dot(n, sub(points[apex as usize], p[0])) > 0.0 {
                f.swap(1, 2);
                n = n.map(|x| -x);
            }
            let area = 0.5 * dot(n, n).sqrt();
            for &v in &f {
                vertex_area[v as usize] += area / 3.0;
            }
            oriented.push(f);
        }
        let mut node_of_vertex = vec![NONE; n_vertices];
        let mut membrane = Vec::new();
        for v in 0..n_vertices {
            if vertex_area[v] > 0.0 {
                node_of_vertex[v] = membrane.len() as u32;
                membrane.push(MembraneNode {
                    inner: vertex_inner[v],
                    outer: vertex_outer[v],
                    area: vertex_area[v],
                    cell: vertex_cell[v],
                });
            }
        }
        let membrane_facets = oriented
            .into_iter()
            .map(|f| f.map(|v| node_of_vertex[v as usize]))
            .collect();

        Ok(Self {
            points,
            grid: None,
            n_cells,
            tets: dof_tets,
            tet_tag,
            dof_vertex,
            dof_tag,
            vertex_outer,
            vertex_inner,
            lumped_volume,
            membrane,
            membrane_facets,
        })
    }

    /// Number of DOFs per scalar field (all subdomains together).
    pub fn n_dofs(&self) -> usize {
        self.dof_vertex.len()
    }

    /// Coordinates of vertex `v`, µm.
    pub fn vertex_position(&self, v: usize) -> [f64; 3] {
        self.points[v]
    }

    /// Coordinates of the DOF `d`'s vertex, µm.
    pub fn dof_position(&self, d: usize) -> [f64; 3] {
        self.points[self.dof_vertex[d] as usize]
    }

    /// Coordinates of the four vertices of tet `t`, µm.
    pub fn tet_points(&self, t: usize) -> [[f64; 3]; 4] {
        self.tets[t].map(|d| self.dof_position(d as usize))
    }

    /// Lower and upper corners of the bounding box of all vertices, µm.
    pub fn bounds(&self) -> ([f64; 3], [f64; 3]) {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in &self.points {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        (lo, hi)
    }
}

/// Reverse Cuthill-McKee order of the vertices of `tets` (each component from a
/// minimum-degree vertex), so neighbouring DOFs sit close in memory.
fn cuthill_mckee(n: usize, tets: &[[u32; 4]]) -> Vec<usize> {
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for tet in tets {
        for a in 0..4 {
            for b in 0..4 {
                if a != b {
                    adj[tet[a] as usize].push(tet[b]);
                }
            }
        }
    }
    for l in &mut adj {
        l.sort_unstable();
        l.dedup();
    }
    let mut by_degree: Vec<usize> = (0..n).collect();
    by_degree.sort_by_key(|&v| adj[v].len());
    let mut seen = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for &start in &by_degree {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut head = order.len();
        order.push(start);
        while head < order.len() {
            let v = order[head];
            head += 1;
            let mut next: Vec<usize> = adj[v]
                .iter()
                .map(|&u| u as usize)
                .filter(|&u| !seen[u])
                .collect();
            next.sort_by_key(|&u| adj[u].len());
            for u in next {
                seen[u] = true;
                order.push(u);
            }
        }
    }
    order.reverse();
    order
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
    fn grid_planes_include_cell_faces() {
        let mesh = Mesh::build(&MeshSpec::paper_bundle()).unwrap();
        for y in [0.3, 0.5, 0.6, 0.8, 0.9, 1.1] {
            assert!(
                mesh.grid.as_ref().unwrap().axes[1]
                    .iter()
                    .any(|&p| (p - y).abs() < 1e-12),
                "y = {y}"
            );
        }
        assert!(mesh.grid.as_ref().unwrap().axes[1]
            .windows(2)
            .all(|w| w[1] - w[0] <= 0.1 + 1e-12));
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
        spec.cells[1] = Aabb::new([1.5, 0.8, 0.8], [2.5, 1.2, 1.2]);
        assert_eq!(Mesh::build(&spec).unwrap_err(), MeshError::EmptyCell(1));
        spec.cells.pop();
        spec.max_h[2] = 0.0;
        assert_eq!(Mesh::build(&spec).unwrap_err(), MeshError::BadSpacing);
    }
}
