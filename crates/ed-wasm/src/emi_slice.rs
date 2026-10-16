//! Geometry-independent views of a KNP-EMI mesh: a plane cut through the tetrahedra and
//! sets of membrane nodes for probes.

use ed_emi::mesh::Mesh;

/// The triangles where the plane `x[axis] = at` cuts the mesh. Each corner lies on a tet
/// edge, and a field is sampled there by interpolating its two DOFs, so the values come
/// from the triangle's own subdomain.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlaneCut {
    /// Corner positions, xyz per corner, 3 corners per triangle, µm.
    pub positions: Vec<f32>,
    /// The two DOFs of each corner's edge.
    pub dofs: Vec<[u32; 2]>,
    /// Weight of the second DOF at each corner.
    pub weights: Vec<f32>,
    /// Subdomain tag of each triangle (0 = extracellular).
    pub tags: Vec<f32>,
}

impl PlaneCut {
    pub fn new(mesh: &Mesh, axis: usize, at: f64) -> Self {
        let mut cut = Self::default();
        for (t, tet) in mesh.tets.iter().enumerate() {
            let p = mesh.tet_points(t);
            // Vertices on the plane count as above it, so every crossing is a strict one.
            let above = p.map(|q| q[axis] >= at);
            let n_above = above.iter().filter(|&&a| a).count();
            if n_above == 0 || n_above == 4 {
                continue;
            }
            let (lo, hi): (Vec<usize>, Vec<usize>) = (0..4).partition(|&i| !above[i]);
            let corner = |i: usize, j: usize| {
                let w = (at - p[i][axis]) / (p[j][axis] - p[i][axis]);
                let mut pos = [0, 1, 2].map(|k| p[i][k] + w * (p[j][k] - p[i][k]));
                pos[axis] = at;
                (pos, [tet[i], tet[j]], w)
            };
            // One vertex alone on a side gives a triangle; two and two give a quad whose
            // corners go round the edges lo0-hi0, lo0-hi1, lo1-hi1, lo1-hi0.
            let quad = if lo.len() == 1 || hi.len() == 1 {
                let (a, rest) = if lo.len() == 1 {
                    (lo[0], &hi)
                } else {
                    (hi[0], &lo)
                };
                vec![corner(a, rest[0]), corner(a, rest[1]), corner(a, rest[2])]
            } else {
                vec![
                    corner(lo[0], hi[0]),
                    corner(lo[0], hi[1]),
                    corner(lo[1], hi[1]),
                    corner(lo[1], hi[0]),
                ]
            };
            let tris: &[[usize; 3]] = if quad.len() == 3 {
                &[[0, 1, 2]]
            } else {
                &[[0, 1, 2], [0, 2, 3]]
            };
            for tri in tris {
                // A tet that only touches the plane at a vertex or an edge (the plane passes
                // through mesh vertices) gives degenerate triangles; leave them out.
                let [a, b, c] = tri.map(|c| quad[c].0);
                let (u, v) = (sub(b, a), sub(c, a));
                let n = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                let scale = dot(u, u).max(dot(v, v));
                if dot(n, n) <= 1e-12 * scale * scale {
                    continue;
                }
                for &c in tri {
                    let (pos, dofs, w) = quad[c];
                    cut.positions.extend(pos.map(|x| x as f32));
                    cut.dofs.push(dofs);
                    cut.weights.push(w as f32);
                }
                cut.tags.push(f32::from(mesh.tet_tag[t]));
            }
        }
        cut
    }

    /// `field` (per DOF) at every corner.
    pub fn values(&self, field: &[f64]) -> Vec<f32> {
        self.dofs
            .iter()
            .zip(&self.weights)
            .map(|(&[a, b], &w)| {
                let (fa, fb) = (field[a as usize], field[b as usize]);
                (fa + f64::from(w) * (fb - fa)) as f32
            })
            .collect()
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Membrane nodes of `cell` (any cell if `None`) within `radius` µm of `centre`.
pub fn nodes_near(mesh: &Mesh, cell: Option<u16>, centre: [f64; 3], radius: f64) -> Vec<usize> {
    (0..mesh.membrane.len())
        .filter(|&p| {
            let m = mesh.membrane[p];
            let q = mesh.dof_position(m.inner as usize);
            let d2: f64 = (0..3).map(|k| (q[k] - centre[k]).powi(2)).sum();
            cell.map_or(true, |c| c == m.cell) && d2 <= radius * radius
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed_emi::mesh::{Aabb, MeshSpec};

    fn mesh() -> Mesh {
        Mesh::build(&MeshSpec {
            extent: [4.0, 2.0, 2.0],
            cells: vec![Aabb::new([1.0, 0.5, 0.5], [3.0, 1.5, 1.5])],
            max_h: [0.5, 0.25, 0.25],
        })
        .unwrap()
    }

    #[test]
    fn cut_covers_the_plane_and_interpolates_linear_fields() {
        let m = mesh();
        // 1.0 and 2.0 are planes of mesh vertices.
        for (axis, at) in [(0, 1.7), (1, 0.9), (2, 1.0), (0, 2.0), (1, 1.0)] {
            let cut = PlaneCut::new(&m, axis, at);
            let n = cut.tags.len();
            assert_eq!(cut.positions.len(), 9 * n);
            assert_eq!(cut.weights.len(), 3 * n);
            // Triangle areas add up to the box cross-section.
            let mut area = 0.0;
            for t in 0..n {
                let p = |c: usize| [0, 1, 2].map(|k| f64::from(cut.positions[9 * t + 3 * c + k]));
                let (a, b, c) = (p(0), p(1), p(2));
                assert!([a, b, c].iter().all(|q| (q[axis] - at).abs() < 1e-5));
                let u = [0, 1, 2].map(|k| b[k] - a[k]);
                let w = [0, 1, 2].map(|k| c[k] - a[k]);
                let x = [
                    u[1] * w[2] - u[2] * w[1],
                    u[2] * w[0] - u[0] * w[2],
                    u[0] * w[1] - u[1] * w[0],
                ];
                let tri = 0.5 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
                assert!(tri > 1e-6, "axis {axis} at {at}: degenerate triangle {t}");
                area += tri;
            }
            let face: f64 = (0..3)
                .filter(|&k| k != axis)
                .map(|k| [4.0, 2.0, 2.0][k])
                .product();
            assert!(
                (area - face).abs() < 1e-4 * face,
                "axis {axis}: {area} vs {face}"
            );
            // A linear field is reproduced exactly at the corners.
            let f: Vec<f64> = (0..m.n_dofs())
                .map(|d| {
                    let q = m.dof_position(d);
                    q[0] - 2.0 * q[1] + 0.5 * q[2]
                })
                .collect();
            for (c, v) in cut.values(&f).iter().enumerate() {
                let q = [0, 1, 2].map(|k| f64::from(cut.positions[3 * c + k]));
                assert!((f64::from(*v) - (q[0] - 2.0 * q[1] + 0.5 * q[2])).abs() < 1e-4);
            }
            // The cell shows up wherever the plane passes through it.
            let cell = [(1.0, 3.0), (0.5, 1.5), (0.5, 1.5)][axis];
            assert_eq!(cut.tags.contains(&1.0), cell.0 < at && at < cell.1);
        }
    }

    #[test]
    fn probe_nodes_are_within_the_sphere() {
        let m = mesh();
        let all = nodes_near(&m, None, [2.0, 1.0, 1.0], 10.0);
        assert_eq!(all.len(), m.membrane.len());
        let end = nodes_near(&m, Some(1), [1.0, 1.0, 1.0], 0.3);
        assert!(!end.is_empty() && end.len() < all.len());
        assert!(nodes_near(&m, Some(2), [1.0, 1.0, 1.0], 0.3).is_empty());
    }
}
