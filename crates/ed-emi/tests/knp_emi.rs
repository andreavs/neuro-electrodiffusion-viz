//! Physics checks for the KNP-EMI solver (issue #8 acceptance criteria).

use ed_core::membrane::MembraneParams;
use ed_emi::mesh::{Aabb, Mesh, MeshSpec};
use ed_emi::solver::{EmiParams, EmiSolver, SPECIES};

/// One 200 µm axon of 0.2 × 0.2 µm along x in a 210 × 0.8 × 0.8 µm box.
fn single_axon(params: EmiParams) -> EmiSolver {
    let spec = MeshSpec {
        extent: [210.0, 0.8, 0.8],
        cells: vec![Aabb::new([5.0, 0.3, 0.3], [205.0, 0.5, 0.5])],
        max_h: [5.0, 0.3, 0.3],
    };
    EmiSolver::new(Mesh::build(&spec).unwrap(), params)
}

fn x_of(s: &EmiSolver, p: usize) -> f64 {
    let m = s.mesh().membrane[p];
    s.mesh()
        .vertex_position(s.mesh().dof_vertex[m.inner as usize] as usize)[0]
}

/// Mean membrane potential of `cell` over membrane nodes at `x`.
fn phi_m_at(s: &EmiSolver, cell: u16, x: f64) -> f64 {
    let nodes: Vec<usize> = (0..s.mesh().membrane.len())
        .filter(|&p| s.mesh().membrane[p].cell == cell && (x_of(s, p) - x).abs() < 1e-9)
        .collect();
    assert!(!nodes.is_empty(), "no membrane node at x = {x}");
    nodes.iter().map(|&p| s.phi_m()[p]).sum::<f64>() / nodes.len() as f64
}

fn amount(s: &EmiSolver, k: usize) -> f64 {
    (0..=s.mesh().n_cells as u16)
        .map(|tag| s.amount(k, tag))
        .sum()
}

/// Asserts that each species amount changed by at most the charge on the membrane capacitor
/// (from the end-point change in φ_M; ions only cross the membrane with the capacitive split).
fn assert_within_capacitor_bound(s: &EmiSolver, before: &[f64], phi_m0: &[f64]) {
    let cap_charge: f64 = s
        .mesh()
        .membrane
        .iter()
        .enumerate()
        .map(|(p, m)| m.area * s.params().membrane.c_m * (s.phi_m()[p] - phi_m0[p]).abs())
        .sum::<f64>()
        / ed_emi::solver::F_U;
    for k in 0..3 {
        let change = (amount(s, k) - before[k]).abs();
        assert!(
            change <= cap_charge + 1e-12 * before[k],
            "{}: amount changed by {change:e}, bound {cap_charge:e}",
            SPECIES[k].name
        );
    }
}

/// Line resistance (Ω/µm) of a region with cross-section `area` µm² and concentrations
/// (Na⁺, K⁺, Cl⁻) mM: σ = F/ψ Σ D z² c, with D µm²/ms = 1e-9 m²/s and S/m = 1e-6 S/µm.
fn line_resistance(params: &EmiParams, c: [f64; 3], area: f64) -> f64 {
    let psi_v = 1e-3 * ed_core::constants::psi_mv(params.membrane.temperature);
    let sum: f64 = (0..3).map(|k| SPECIES[k].d * c[k]).sum();
    1.0 / (ed_core::constants::F / psi_v * sum * 1e-9 * 1e-6 * area)
}

/// Conduction velocity (µm/ms, between x = 80 and 160 µm of the axon frame used by
/// [`single_axon`]) of a converged 1-D HH cable with the same membrane, axial resistance
/// r_i + r_e and stimulus: Δx = 1 µm, Δt = 0.005 ms, implicit diffusion.
fn cable_conduction_velocity(params: &EmiParams) -> f64 {
    use ed_core::membrane::MembranePatches;
    let c = params.initial;
    let r = line_resistance(params, [c.na_i, c.k_i, c.cl_i], 0.04)
        + line_resistance(params, [c.na_e, c.k_e, c.cl_e], 0.64 - 0.04);
    // c_m per unit length: µF/cm² × 0.8 µm perimeter, in F/µm; D_V in µm²/ms.
    let c_line = params.membrane.c_m * 1e-6 * 1e-8 * 0.8;
    let d_v = 1e-3 / (r * c_line);
    let (dx, dt, n) = (1.0, 0.005, 201);
    let mut patches = MembranePatches::new(n, params.membrane, c);
    for i in 0..=20 {
        patches.set_synapse(i, 4.0, 0.0); // x = 5..25 µm in the 3-D model
    }
    let k = d_v * dt / (dx * dx);
    let (mut cp, mut dp) = (vec![0.0; n], vec![0.0; n]);
    let probe = [75, 155]; // x = 80 and 160 µm in the 3-D frame
    let mut prev = probe.map(|i| patches.phi_m[i]);
    let mut cross = [f64::NAN; 2];
    let mut t = 0.0;
    while t < 3.0 {
        patches.advance(t, dt, 1, None);
        // (1 − k Δ) V = V* with sealed ends (Thomas algorithm).
        let v = &mut patches.phi_m;
        for i in 0..n {
            let diag = 1.0 + k * (if i == 0 || i == n - 1 { 1.0 } else { 2.0 });
            let (lo, hi) = (if i > 0 { -k } else { 0.0 }, -k);
            let den = diag - lo * if i > 0 { cp[i - 1] } else { 0.0 };
            cp[i] = hi / den;
            dp[i] = (v[i] - lo * if i > 0 { dp[i - 1] } else { 0.0 }) / den;
        }
        v[n - 1] = dp[n - 1];
        for i in (0..n - 1).rev() {
            v[i] = dp[i] - cp[i] * v[i + 1];
        }
        t += dt;
        for (j, &i) in probe.iter().enumerate() {
            let vi = v[i];
            if cross[j].is_nan() && vi > -20.0 && prev[j] <= -20.0 {
                cross[j] = t - dt * (vi + 20.0) / (vi - prev[j]);
            }
            prev[j] = vi;
        }
    }
    80.0 / (cross[1] - cross[0])
}

#[test]
fn axon_at_rest_stays_at_rest() {
    let mut s = single_axon(EmiParams::default());
    let rest = s.phi_m()[0];
    for _ in 0..200 {
        s.step().unwrap();
    }
    let drift = s
        .phi_m()
        .iter()
        .map(|v| (v - rest).abs())
        .fold(0.0, f64::max);
    assert!(
        drift < 0.5,
        "φ_M drifted {drift} mV from rest {rest} mV in 20 ms"
    );
    let phi_e_max = (0..s.mesh().n_dofs())
        .filter(|&d| s.mesh().dof_tag[d] == 0)
        .map(|d| s.phi()[d].abs())
        .fold(0.0, f64::max);
    assert!(phi_e_max < 0.1, "|φ_e| = {phi_e_max} mV at rest");
}

#[test]
fn action_potential_propagates_and_conserves_ions_and_charge() {
    // Δt = 0.05 ms: backward Euler is first order, and at the default 0.1 ms the conduction
    // velocity is ~5% below its converged value.
    let mut s = single_axon(EmiParams {
        dt: 0.05,
        ..Default::default()
    });
    s.stimulate(1, 0.0, 25.0, 4.0, 0.0);
    let before: Vec<f64> = (0..3).map(|k| amount(&s, k)).collect();
    let phi_m0 = s.phi_m().to_vec();
    let charge =
        |s: &EmiSolver| -> f64 { (0..3).map(|k| SPECIES[k].z as f64 * amount(s, k)).sum() };
    let charge0 = charge(&s);
    let k_e0 = s.conc(1)[s.mesh().membrane[0].outer as usize];

    let mut max_imbalance: f64 = 0.0;
    let mut cross = [f64::NAN; 2];
    let mut prev = [phi_m_at(&s, 1, 80.0), phi_m_at(&s, 1, 160.0)];
    while s.t() < 5.0 {
        s.step().unwrap();
        max_imbalance = max_imbalance.max(s.max_charge_imbalance());
        for (i, x) in [80.0, 160.0].into_iter().enumerate() {
            let v = phi_m_at(&s, 1, x);
            if cross[i].is_nan() && v > -20.0 && prev[i] <= -20.0 {
                cross[i] = s.t() - s.params().dt * (v + 20.0) / (v - prev[i]);
            }
            prev[i] = v;
        }
    }
    let cv = 80.0 / (cross[1] - cross[0]);
    let cv_cable = cable_conduction_velocity(s.params());
    println!("conduction velocity {cv:.0} µm/ms, 1-D cable reference {cv_cable:.0} µm/ms");
    assert!(
        (cv / cv_cable - 1.0).abs() < 0.05,
        "{cv} vs {cv_cable} µm/ms"
    );
    assert!(
        phi_m_at(&s, 1, 160.0) < 0.0,
        "the action potential should repolarise"
    );

    // Each species: the only exchange is across the membrane, where the capacitive part of
    // the flux is split differently on the two sides (α_i ≠ α_e, eq. 13). The channel part
    // moves the same ions on both sides and the capacitive part integrates to C_M Δφ_M, so
    // with |α_e − α_i| ≤ 1 each amount changes by at most Σ A C_M |Δφ_M| / (F |z|).
    assert_within_capacitor_bound(&s, &before, &phi_m0);
    let dq = (charge(&s) - charge0).abs();
    assert!(dq < 1e-9 * before[0], "net charge changed by {dq}");
    assert!(
        max_imbalance < 1e-2,
        "bulk charge imbalance {max_imbalance} mM"
    );

    // Extracellular K⁺ rises next to the membrane that fired.
    let k_e = (0..s.mesh().membrane.len())
        .filter(|&p| (x_of(&s, p) - 100.0).abs() < 1e-9)
        .map(|p| s.conc(1)[s.mesh().membrane[p].outer as usize])
        .fold(0.0, f64::max);
    assert!(
        k_e > k_e0 + 0.01,
        "[K]_e near the membrane {k_e} mM (from {k_e0} mM)"
    );
}

#[test]
fn passive_axon_matches_cable_theory() {
    // A sustained synaptic conductance at one end of a passive axon gives a steady
    // depolarisation V(x) ∝ cosh((L − x)/λ) for a sealed end at L, with
    // λ = sqrt(r_m / (r_i + r_e)).
    let mut params = EmiParams {
        membrane: MembraneParams {
            syn_tau: 1e12,
            ..MembraneParams::PASSIVE
        },
        ..Default::default()
    };
    params.dt = 0.2;
    let mut s = single_axon(params);
    s.stimulate(1, 0.0, 15.0, 2.0, 0.0);
    let rest = s.phi_m()[0];
    let before: Vec<f64> = (0..3).map(|k| amount(&s, k)).collect();
    while s.t() < 8.0 {
        s.step().unwrap();
    }
    let c = s.params().initial;
    let r_i = line_resistance(s.params(), [c.na_i, c.k_i, c.cl_i], 0.04);
    let r_e = line_resistance(s.params(), [c.na_e, c.k_e, c.cl_e], 0.64 - 0.04);
    let g_m = s.params().membrane.g_leak_na + s.params().membrane.g_leak_k; // mS/cm²
    let r_m = 1.0 / (g_m * 1e-3 * 1e-8 * 0.8); // Ω µm
    let lambda = (r_m / (r_i + r_e)).sqrt();
    let (l_end, x1, x2) = (205.0, 40.0, 140.0);
    let v1 = phi_m_at(&s, 1, x1) - rest;
    let v2 = phi_m_at(&s, 1, x2) - rest;
    let model = (((l_end - x2) / lambda).cosh()) / (((l_end - x1) / lambda).cosh());
    println!(
        "λ = {lambda:.1} µm, V2/V1 = {:.4}, cable {model:.4}",
        v2 / v1
    );
    assert!(v1 > 1.0, "depolarisation {v1} mV");
    assert!(((v2 / v1) / model - 1.0).abs() < 0.02);

    let phi_m0 = vec![rest; s.mesh().membrane.len()];
    assert_within_capacitor_bound(&s, &before, &phi_m0);
}

/// A mesh without a grid (as from gmsh) runs: same membrane, a stable resting state.
#[test]
fn unstructured_mesh_runs() {
    let grid = single_axon(EmiParams::default());
    let m = grid.mesh();
    let tets: Vec<[u32; 4]> = m
        .tets
        .iter()
        .map(|t| t.map(|d| m.dof_vertex[d as usize]))
        .collect();
    let mesh = Mesh::from_tets(m.points.clone(), &tets, m.tet_tag.clone(), 1).unwrap();
    let mut s = EmiSolver::new(mesh, EmiParams::default());
    let rest = s.phi_m()[0];
    for _ in 0..10 {
        s.step().unwrap();
    }
    assert!(s.phi_m().iter().all(|v| (v - rest).abs() < 1.0));
}
