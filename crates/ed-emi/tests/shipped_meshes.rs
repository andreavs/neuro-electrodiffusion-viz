//! The meshes in web/public/meshes (made by tools/meshes/generate.py) load, and every
//! stimulus site and probe in their JSON lies on the membrane.

use std::path::Path;

const GEOMETRIES: [&str; 3] = ["neuron", "spiny-dendrite", "hay-neuron"];

/// The `"centre": [x, y, z], "radius": r` spheres in a geometry's JSON, one per line.
fn spheres(json: &str) -> Vec<([f64; 3], f64)> {
    let num = |s: &str| s.trim().parse::<f64>().unwrap();
    json.lines()
        .filter_map(|line| {
            let c = line.split("\"centre\": [").nth(1)?;
            let (xyz, rest) = c.split_once(']')?;
            let v: Vec<f64> = xyz.split(',').map(num).collect();
            let r = rest.split("\"radius\": ").nth(1)?;
            let r = num(r.split([',', '}']).next()?);
            Some(([v[0], v[1], v[2]], r))
        })
        .collect()
}

#[test]
fn shipped_meshes_load_and_their_sites_touch_the_membrane() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/public/meshes");
    for name in GEOMETRIES {
        let bytes = std::fs::read(dir.join(format!("{name}.edmesh"))).unwrap();
        let mesh = ed_emi::meshfile::read(&bytes).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_eq!(mesh.n_cells, 1, "{name}");
        assert!(!mesh.membrane_facets.is_empty(), "{name}");
        let json = std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap();
        let sites = spheres(&json);
        // Every site parsed (one per line, as generate.py writes them).
        assert_eq!(sites.len(), json.matches("\"centre\"").count(), "{name}");
        for (c, r) in sites {
            let hits = mesh
                .membrane
                .iter()
                .filter(|m| {
                    let p = mesh.dof_position(m.inner as usize);
                    (0..3).map(|k| (p[k] - c[k]).powi(2)).sum::<f64>() < r * r
                })
                .count();
            assert!(
                hits >= 5,
                "{name}: site {c:?} r {r} touches {hits} membrane nodes"
            );
        }
    }
}
