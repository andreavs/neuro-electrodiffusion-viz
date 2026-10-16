//! Mesh invariants: conformity and tet orientation.

use ed_emi::mesh::{Aabb, Mesh, MeshSpec, NONE};

fn single_cell() -> MeshSpec {
    MeshSpec {
        extent: [4.0, 2.0, 2.0],
        cells: vec![Aabb::new([1.0, 0.5, 0.5], [3.0, 1.5, 1.5])],
        max_h: [0.5, 0.25, 0.25],
    }
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
fn mesh_is_conforming() {
    // Cell 1 touches the x = 0 face of the box.
    let mut spec = single_cell();
    spec.cells
        .push(Aabb::new([0.0, 0.0, 0.0], [0.5, 0.25, 0.25]));
    let mesh = Mesh::build(&spec).unwrap();
    let mut facets = std::collections::HashMap::new();
    for tet in &mesh.tets {
        for skip in 0..4 {
            let mut f = [0, 1, 2].map(|i| tet[i + usize::from(i >= skip)]);
            f.sort_unstable();
            *facets.entry(f).or_insert(0) += 1;
        }
    }
    for (f, n) in facets {
        assert!(n <= 2);
        if n == 1 {
            // Exposed facets lie on the box or on a membrane.
            let p = f.map(|d| mesh.vertex_position(mesh.dof_vertex[d as usize] as usize));
            let on_box = (0..3)
                .any(|a| p.iter().all(|q| q[a] == 0.0) || p.iter().all(|q| q[a] == spec.extent[a]));
            let on_membrane = f.iter().all(|&d| {
                let v = mesh.dof_vertex[d as usize] as usize;
                mesh.vertex_inner[v] != NONE && mesh.vertex_outer[v] != NONE
            });
            assert!(on_box || on_membrane);
        }
    }
}
