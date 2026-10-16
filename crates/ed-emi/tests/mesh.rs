//! Mesh invariants: conformity and tet orientation.

use ed_emi::mesh::{Aabb, Mesh, MeshError, MeshSpec, NONE};

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

/// The single-cell grid mesh rebuilt by `from_tets` with every vertex off the membrane and
/// the box moved a little, so nothing is aligned with a grid any more.
fn jittered() -> (Mesh, Mesh) {
    let grid = Mesh::build(&single_cell()).unwrap();
    let fixed: Vec<bool> = (0..grid.points.len())
        .map(|v| {
            let p = grid.points[v];
            let on_box = (0..3).any(|a| p[a] == 0.0 || p[a] == single_cell().extent[a]);
            on_box || (grid.vertex_inner[v] != NONE && grid.vertex_outer[v] != NONE)
        })
        .collect();
    let points = grid
        .points
        .iter()
        .enumerate()
        .map(|(v, p)| {
            if fixed[v] {
                *p
            } else {
                let s = 0.05 * ((v * 7919) % 13) as f64 / 13.0;
                [p[0] + s, p[1] - 0.5 * s, p[2] + 0.3 * s]
            }
        })
        .collect();
    let tets: Vec<[u32; 4]> = grid
        .tets
        .iter()
        .map(|t| t.map(|d| grid.dof_vertex[d as usize]))
        .collect();
    let mesh = Mesh::from_tets(points, &tets, grid.tet_tag.clone(), 1).unwrap();
    (grid, mesh)
}

#[test]
fn unstructured_mesh_keeps_volumes_and_membrane() {
    let (grid, mesh) = jittered();
    assert!(mesh.grid.is_none());
    let vol = |m: &Mesh, tag: u16| -> f64 {
        (0..m.n_dofs())
            .filter(|&d| m.dof_tag[d] == tag)
            .map(|d| m.lumped_volume[d])
            .sum()
    };
    for tag in [0, 1] {
        assert!((vol(&mesh, tag) - vol(&grid, tag)).abs() < 1e-9);
    }
    assert_eq!(mesh.membrane.len(), grid.membrane.len());
    let area = |m: &Mesh| m.membrane.iter().map(|n| n.area).sum::<f64>();
    assert!((area(&mesh) - 2.0 * (2.0 + 2.0 + 1.0)).abs() < 1e-9);
    assert!((area(&grid) - area(&mesh)).abs() < 1e-9);
    // Every facet is counter-clockwise seen from outside the cell (centre (2, 1, 1)).
    assert_eq!(mesh.membrane_facets.len(), grid.membrane_facets.len());
    for f in &mesh.membrane_facets {
        let p = f.map(|n| mesh.dof_position(mesh.membrane[n as usize].inner as usize));
        let (u, w) = (
            [0, 1, 2].map(|a| p[1][a] - p[0][a]),
            [0, 1, 2].map(|a| p[2][a] - p[0][a]),
        );
        let n = [
            u[1] * w[2] - u[2] * w[1],
            u[2] * w[0] - u[0] * w[2],
            u[0] * w[1] - u[1] * w[0],
        ];
        let out: f64 = (0..3).map(|a| n[a] * (p[0][a] - [2.0, 1.0, 1.0][a])).sum();
        assert!(out > 0.0);
    }
}

#[test]
fn from_tets_rejects_bad_input() {
    let p = vec![
        [0.0; 3],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ];
    let ok = [[0, 1, 2, 3]];
    assert!(Mesh::from_tets(p.clone(), &ok, vec![0], 0).is_ok());
    let err = |tets: &[[u32; 4]], tags: Vec<u16>, n| Mesh::from_tets(p.clone(), tets, tags, n);
    assert_eq!(
        err(&[[0, 1, 2, 9]], vec![0], 0).unwrap_err(),
        MeshError::BadTet(0)
    );
    assert_eq!(err(&ok, vec![2], 1).unwrap_err(), MeshError::BadTet(0));
    // Flat: all four points in z = 0.
    assert_eq!(
        err(&[[0, 1, 2, 4]], vec![0], 0).unwrap_err(),
        MeshError::BadTet(0)
    );
    assert_eq!(err(&ok, vec![0], 1).unwrap_err(), MeshError::EmptyCell(0));
    assert_eq!(err(&ok, vec![], 0).unwrap_err(), MeshError::BadTet(0));
}
