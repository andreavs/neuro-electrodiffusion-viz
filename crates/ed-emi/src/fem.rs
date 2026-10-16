//! P1 (linear Lagrange) finite elements on tetrahedra.
//!
//! The stiffness matrix of `⟨w ∇u, ∇v⟩` with a weight `w` that is constant on each element is
//! assembled from precomputed element edge coefficients. A P1 element stiffness matrix has
//! zero row sums, so it is fully described by its off-diagonal entries `s_ij`:
//! `(K u)_i = Σ_e w_e Σ_j s_ij (u_j − u_i)`. Only nonzero edges are stored; on the Kuhn
//! tetrahedra of a rectilinear grid that is three of the six edges.

use ed_core::linalg::{CgOptions, CgStats, CsrMatrix, Pcg, Preconditioner};

use crate::mesh::Mesh;

/// Gradients of the four barycentric (hat) functions and the volume of a tetrahedron.
pub fn tet_gradients(p: [[f64; 3]; 4]) -> ([[f64; 3]; 4], f64) {
    let e = [1, 2, 3].map(|i| [0, 1, 2].map(|a| p[i][a] - p[0][a]));
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    // Rows of the inverse Jacobian are the gradients of λ1, λ2, λ3.
    let c = [cross(e[1], e[2]), cross(e[2], e[0]), cross(e[0], e[1])];
    let det: f64 = (0..3).map(|a| e[0][a] * c[0][a]).sum();
    let g1 = c[0].map(|x| x / det);
    let g2 = c[1].map(|x| x / det);
    let g3 = c[2].map(|x| x / det);
    let g0 = [0, 1, 2].map(|a| -(g1[a] + g2[a] + g3[a]));
    ([g0, g1, g2, g3], det.abs() / 6.0)
}

/// Element stiffness matrix `∫ ∇λ_i · ∇λ_j dx`.
pub fn tet_stiffness(p: [[f64; 3]; 4]) -> [[f64; 4]; 4] {
    let (g, vol) = tet_gradients(p);
    let mut s = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            s[i][j] = vol * (0..3).map(|a| g[i][a] * g[j][a]).sum::<f64>();
        }
    }
    s
}

/// One nonzero element edge coefficient `s_ij` (i < j) of element `elem`.
#[derive(Debug, Clone, Copy)]
struct Edge {
    elem: u32,
    i: u32,
    j: u32,
    s: f64,
    /// CSR positions of (i, j), (j, i), (i, i), (j, j).
    pos: [u32; 4],
}

/// Weighted P1 stiffness assembly on a [`Mesh`], plus a matrix pattern that also holds the
/// membrane couplings between the two DOFs of each membrane vertex.
#[derive(Debug, Clone)]
pub struct Stiffness {
    edges: Vec<Edge>,
    pattern: CsrMatrix,
    diag_pos: Vec<u32>,
    /// CSR positions of (inner, outer), (outer, inner), (inner, inner), (outer, outer) for
    /// each membrane node.
    membrane_pos: Vec<[u32; 4]>,
}

impl Stiffness {
    pub fn new(mesh: &Mesh) -> Self {
        let n = mesh.n_dofs();
        let mut raw = Vec::new();
        for (t, tet) in mesh.tets.iter().enumerate() {
            let s = tet_stiffness(mesh.tet_points(t));
            let scale = (0..4).map(|i| s[i][i]).fold(0.0, f64::max);
            for a in 0..4 {
                for b in a + 1..4 {
                    if s[a][b].abs() > 1e-12 * scale {
                        raw.push((t as u32, tet[a], tet[b], s[a][b]));
                    }
                }
            }
        }
        let mut triplets: Vec<(usize, usize, f64)> = (0..n).map(|i| (i, i, 0.0)).collect();
        for &(_, i, j, _) in &raw {
            triplets.push((i as usize, j as usize, 0.0));
            triplets.push((j as usize, i as usize, 0.0));
        }
        for m in &mesh.membrane {
            triplets.push((m.inner as usize, m.outer as usize, 0.0));
            triplets.push((m.outer as usize, m.inner as usize, 0.0));
        }
        let pattern = CsrMatrix::from_triplets(n, n, &triplets);
        let pos = |r: u32, c: u32| {
            pattern
                .position(r as usize, c as usize)
                .expect("in pattern") as u32
        };
        let edges = raw
            .iter()
            .map(|&(elem, i, j, s)| Edge {
                elem,
                i,
                j,
                s,
                pos: [pos(i, j), pos(j, i), pos(i, i), pos(j, j)],
            })
            .collect();
        let diag_pos = (0..n as u32).map(|i| pos(i, i)).collect();
        let membrane_pos = mesh
            .membrane
            .iter()
            .map(|m| {
                [
                    pos(m.inner, m.outer),
                    pos(m.outer, m.inner),
                    pos(m.inner, m.inner),
                    pos(m.outer, m.outer),
                ]
            })
            .collect();
        Self {
            edges,
            pattern,
            diag_pos,
            membrane_pos,
        }
    }

    /// A zero matrix with the full sparsity pattern, to pass to [`assemble`](Self::assemble).
    pub fn new_matrix(&self) -> CsrMatrix {
        self.pattern.clone()
    }

    /// Overwrites `out` with `K_w + diag(shift) + membrane coupling`, where `K_w` is the
    /// stiffness with element weights `weight`, and each membrane node `m` with coupling
    /// `c = coupling[m]` adds `c [[1, −1], [−1, 1]]` on its (inner, outer) DOFs.
    pub fn assemble(&self, weight: &[f64], shift: &[f64], coupling: &[f64], out: &mut CsrMatrix) {
        debug_assert_eq!(shift.len(), self.diag_pos.len());
        debug_assert_eq!(coupling.len(), self.membrane_pos.len());
        let v = out.values_mut();
        v.fill(0.0);
        for e in &self.edges {
            let s = weight[e.elem as usize] * e.s;
            v[e.pos[0] as usize] += s;
            v[e.pos[1] as usize] += s;
            v[e.pos[2] as usize] -= s;
            v[e.pos[3] as usize] -= s;
        }
        for (d, &p) in self.diag_pos.iter().enumerate() {
            v[p as usize] += shift[d];
        }
        for (m, p) in self.membrane_pos.iter().enumerate() {
            let c = coupling[m];
            v[p[0] as usize] -= c;
            v[p[1] as usize] -= c;
            v[p[2] as usize] += c;
            v[p[3] as usize] += c;
        }
    }

    /// Adds `K_w u` to `out`, without assembling (element weights `weight`).
    pub fn apply_add(&self, weight: &[f64], u: &[f64], out: &mut [f64]) {
        for e in &self.edges {
            let (i, j) = (e.i as usize, e.j as usize);
            let f = weight[e.elem as usize] * e.s * (u[j] - u[i]);
            out[i] += f;
            out[j] -= f;
        }
    }
}

/// Aggregates for [`ed_core::linalg::TwoLevel`]: one per (x-plane, subdomain), numbered
/// plane by plane. Needs a grid mesh.
pub fn plane_aggregates(mesh: &Mesh) -> Vec<u32> {
    let nx = mesh.grid.as_ref().expect("a grid mesh").axes[0].len();
    let stride = mesh.n_cells as u32 + 1;
    let mut agg: Vec<u32> = (0..mesh.n_dofs())
        .map(|d| (mesh.dof_vertex[d] % nx as u32) * stride + mesh.dof_tag[d] as u32)
        .collect();
    // Renumber densely (a plane may not meet every cell), keeping the order.
    let mut map = vec![u32::MAX; nx * stride as usize];
    let mut used: Vec<u32> = agg.clone();
    used.sort_unstable();
    used.dedup();
    for (i, &g) in used.iter().enumerate() {
        map[g as usize] = i as u32;
    }
    agg.iter_mut().for_each(|g| *g = map[*g as usize]);
    agg
}

/// Solves `A x = b` as a correction `x += δ`, `A δ = b − A x`, so the tolerance is
/// relative to the change in this step rather than to the (much larger) solution.
pub fn solve_increment(
    cg: &mut Pcg,
    a: &CsrMatrix,
    pc: &mut dyn Preconditioner,
    b: &[f64],
    x: &mut [f64],
    work: &mut [Vec<f64>; 2],
    opts: &CgOptions,
) -> CgStats {
    let [r, delta] = work;
    a.matvec(x, r);
    r.iter_mut().zip(b).for_each(|(ri, bi)| *ri = bi - *ri);
    delta.fill(0.0);
    let stats = cg.solve(a, pc, r, delta, opts);
    x.iter_mut()
        .zip(delta.iter())
        .for_each(|(xi, di)| *xi += di);
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Aabb, MeshSpec};
    use ed_core::linalg::{CgOptions, Jacobi, Pcg};
    use std::f64::consts::PI;

    #[test]
    fn hat_gradients_reproduce_linear_functions() {
        let p = [
            [0.1, 0.0, 0.2],
            [1.0, 0.3, 0.0],
            [0.2, 1.1, 0.1],
            [0.0, 0.4, 0.9],
        ];
        let (g, vol) = tet_gradients(p);
        let f = |x: [f64; 3]| 2.0 * x[0] - 3.0 * x[1] + 0.5 * x[2] + 1.0;
        for a in 0..3 {
            let grad: f64 = (0..4).map(|i| f(p[i]) * g[i][a]).sum();
            assert!((grad - [2.0, -3.0, 0.5][a]).abs() < 1e-12);
        }
        assert!(vol > 0.0);
        let s = tet_stiffness(p);
        for row in s {
            assert!(row.iter().sum::<f64>().abs() < 1e-12);
        }
    }

    #[test]
    fn kuhn_tets_couple_only_along_grid_edges() {
        let mesh = Mesh::build(&MeshSpec {
            extent: [1.0, 2.0, 3.0],
            cells: vec![],
            max_h: [1.0, 2.0, 3.0],
        })
        .unwrap();
        let k = Stiffness::new(&mesh);
        assert_eq!(k.edges.len(), 6 * 3);
        assert!(k.edges.iter().all(|e| e.s < 0.0));
    }

    /// `−∇·∇u + u = f` with homogeneous Neumann conditions on the unit cube, for
    /// `u = cos πx cos πy cos πz`. Returns the discrete L2 error (lumped-mass weighted).
    fn neumann_error(n: usize) -> f64 {
        let h = 1.0 / n as f64;
        let mesh = Mesh::build(&MeshSpec {
            extent: [1.0; 3],
            cells: vec![],
            max_h: [h; 3],
        })
        .unwrap();
        let k = Stiffness::new(&mesh);
        let mut a = k.new_matrix();
        let ones = vec![1.0; mesh.tets.len()];
        k.assemble(&ones, &mesh.lumped_volume, &[], &mut a);
        let u = |p: [f64; 3]| (PI * p[0]).cos() * (PI * p[1]).cos() * (PI * p[2]).cos();
        let pos = |d: usize| mesh.dof_position(d);
        let b: Vec<f64> = (0..mesh.n_dofs())
            .map(|d| mesh.lumped_volume[d] * (3.0 * PI * PI + 1.0) * u(pos(d)))
            .collect();
        let mut x = vec![0.0; mesh.n_dofs()];
        let mut m = Jacobi::new(&a);
        let opts = CgOptions {
            rtol: 1e-12,
            ..Default::default()
        };
        assert!(
            Pcg::new(x.len())
                .solve(&a, &mut m, &b, &mut x, &opts)
                .converged
        );
        (0..mesh.n_dofs())
            .map(|d| mesh.lumped_volume[d] * (x[d] - u(pos(d))).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn neumann_problem_converges_at_second_order() {
        let e1 = neumann_error(8);
        let e2 = neumann_error(16);
        let rate = (e1 / e2).log2();
        assert!(rate > 1.9, "errors {e1} {e2}, rate {rate}");
    }

    #[test]
    fn assembled_operator_matches_matrix_free_and_has_membrane_coupling() {
        let mesh = Mesh::build(&MeshSpec {
            extent: [2.0, 1.0, 1.0],
            cells: vec![Aabb::new([0.5, 0.25, 0.25], [1.5, 0.75, 0.75])],
            max_h: [0.25; 3],
        })
        .unwrap();
        let k = Stiffness::new(&mesh);
        let n = mesh.n_dofs();
        let w: Vec<f64> = (0..mesh.tets.len()).map(|t| 1.0 + (t % 7) as f64).collect();
        let coupling: Vec<f64> = (0..mesh.membrane.len()).map(|m| 0.1 * m as f64).collect();
        let mut a = k.new_matrix();
        k.assemble(&w, &vec![0.0; n], &coupling, &mut a);
        let u: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let mut y = vec![0.0; n];
        a.matvec(&u, &mut y);
        let mut z = vec![0.0; n];
        k.apply_add(&w, &u, &mut z);
        for (m, node) in mesh.membrane.iter().enumerate() {
            let jump = coupling[m] * (u[node.inner as usize] - u[node.outer as usize]);
            z[node.inner as usize] += jump;
            z[node.outer as usize] -= jump;
        }
        for i in 0..n {
            assert!((y[i] - z[i]).abs() < 1e-10, "row {i}");
        }
        // Constants are in the null space: no flux leaves the closed box.
        a.matvec(&vec![1.0; n], &mut y);
        assert!(y.iter().all(|v| v.abs() < 1e-10));
    }
}
