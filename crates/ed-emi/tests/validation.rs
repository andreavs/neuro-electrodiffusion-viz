//! Physical validation of the KNP-EMI solver.

use ed_core::constants::psi_mv;
use ed_core::membrane::{IonConcentrations, MembraneParams};
use ed_core::Simulation;
use ed_emi::{AxonBundle, EmiParams, EmiSolver, VoxelGeometry, ECS, SPECIES, UA_PER_CM2_TO_FLUX};

/// One axon of square cross-section `side` (µm) and `length` (µm) at 1 µm voxels across
/// and 2 µm along the axon, in a tight extracellular box. Returns the solver and the x
/// coordinate where the axon starts.
fn single_axon(length: f64, params: EmiParams) -> (EmiSolver, f64) {
    let bundle = AxonBundle {
        count: 1,
        length,
        ..AxonBundle::default()
    };
    let (geo, axons) = bundle.build();
    let x0 = axons[0].lo[0] as f64 * bundle.hx;
    (EmiSolver::new(geo, params), x0)
}

/// Lateral (+y side) membrane patches of cell `cell`, ordered by x.
fn lateral_patches(sim: &EmiSolver, cell: u32) -> Vec<(f64, usize)> {
    let mut out: Vec<(f64, usize)> = sim
        .faces()
        .iter()
        .enumerate()
        .filter(|(_, f)| f.cell == cell && f.face.axis == 1 && f.outside > f.inside)
        .map(|(i, f)| (f.center(&sim.geometry().grid)[0], i))
        .collect();
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

#[test]
fn axon_at_rest_stays_at_rest() {
    let (mut sim, _) = single_axon(60.0, EmiParams::default());
    let rest = sim.membrane_potential()[0];
    let mut worst = 0.0f64;
    for _ in 0..200 {
        sim.step(1);
        for v in sim.membrane_potential() {
            worst = worst.max((v - rest).abs());
        }
    }
    println!("rest {rest:.3} mV, max deviation over 20 ms {worst:.2e} mV");
    assert!(worst < 0.5, "membrane potential drifted by {worst} mV");
}

#[test]
fn ions_are_conserved_and_bulk_stays_electroneutral() {
    let bundle = AxonBundle {
        count: 4,
        length: 60.0,
        ..AxonBundle::default()
    };
    let (geo, axons) = bundle.build();
    let mut sim = EmiSolver::new(geo, EmiParams::default());
    let x0 = axons[0].lo[0] as f64 * bundle.hx;
    let p = sim.patches_near_start(1, x0 + 10.0);
    sim.add_synapse(0.5, 4.0, &p);
    let before = sim.total_ions();
    let (mut peak, mut rho_max) = (f64::MIN, 0.0f64);
    for _ in 0..80 {
        sim.step(1);
        peak = sim.membrane_potential().iter().fold(peak, |m, &v| m.max(v));
        rho_max = sim
            .charge_density()
            .iter()
            .fold(rho_max, |m, v| m.max(v.abs()));
    }
    let after = sim.total_ions();
    for k in 0..3 {
        let rel = (after[k] - before[k]).abs() / before[k];
        println!("{}: relative change {rel:.2e}", SPECIES[k].name);
        assert!(rel < 1e-9, "{} not conserved: {rel}", SPECIES[k].name);
    }
    println!("peak {peak:.1} mV, max |Σ z c| {rho_max:.2e} mM");
    assert!(peak > 20.0, "no action potential (peak {peak} mV)");
    assert!(rho_max < 1e-3, "bulk charge {rho_max} mM");
}

#[test]
fn action_potential_propagates_and_raises_extracellular_potassium() {
    let length = 200.0;
    let params = EmiParams::default();
    let (mut sim, x0) = single_axon(length, params);
    let (mut quiet, _) = single_axon(length, params);
    let p = sim.patches_near_start(1, x0 + 10.0);
    sim.add_synapse(0.5, 4.0, &p);
    let lat = lateral_patches(&sim, 1);
    let probe = |x: f64| {
        lat.iter()
            .min_by(|a, b| (a.0 - x).abs().total_cmp(&(b.0 - x).abs()))
            .unwrap()
    };
    let (a, b) = (*probe(x0 + 50.0), *probe(x0 + 150.0));
    let mut cross = [None, None];
    let mut prev = [sim.membrane_potential()[a.1], sim.membrane_potential()[b.1]];
    for _ in 0..(6.0 / params.dt) as usize {
        sim.step(1);
        quiet.step(1);
        for (n, &(_, i)) in [a, b].iter().enumerate() {
            let v = sim.membrane_potential()[i];
            if cross[n].is_none() && prev[n] < 0.0 && v >= 0.0 {
                let dt = sim.params().dt;
                cross[n] = Some(sim.time_ms() - dt * v / (v - prev[n]));
            }
            prev[n] = v;
        }
    }
    let (ta, tb) = (
        cross[0].expect("AP at x=50"),
        cross[1].expect("AP at x=150"),
    );
    let cv = (b.0 - a.0) / (tb - ta);
    println!(
        "AP crosses 0 mV at {ta:.3} and {tb:.3} ms: conduction velocity {cv:.0} µm/ms ({:.2} m/s)",
        cv / 1e3
    );
    assert!(tb > ta, "AP did not travel away from the synapse");
    assert!(
        (100.0..5000.0).contains(&cv),
        "implausible conduction velocity {cv} µm/ms"
    );

    // Extracellular K⁺ next to the axon is higher than without the action potential.
    let k = sim.concentration(1);
    let k0 = quiet.concentration(1);
    let rise = (0..lat.len())
        .map(|n| sim.faces()[lat[n].1].outside)
        .map(|v| k[v] - k0[v])
        .fold(f64::MAX, f64::min);
    println!("minimum extracellular K⁺ rise along the axon: {rise:.4} mM");
    assert!(rise > 1e-3, "K⁺ rise {rise} mM");
}

/// Steady state of a passive axon with a constant synapse at one end, compared with a 1-D
/// cable whose axial conductance combines the intra- and extracellular paths in series
/// (closed box: the axial currents are equal and opposite).
#[test]
fn passive_axon_matches_cable_equation() {
    let params = EmiParams {
        membrane: MembraneParams {
            syn_tau: 1e12, // constant synaptic conductance
            ..MembraneParams::PASSIVE
        },
        ..EmiParams::default()
    };
    let g_syn: f64 = std::env::var("GSYN")
        .map(|s| s.parse().unwrap())
        .unwrap_or(4.0);
    let steps: usize = std::env::var("STEPS")
        .map(|s| s.parse().unwrap())
        .unwrap_or(150);
    let (syn_len, length) = (10.0, 200.0);
    let (mut sim, x0) = single_axon(length, params);
    let p = sim.patches_near_start(1, x0 + syn_len);
    sim.add_synapse(0.0, g_syn, &p);
    sim.step(steps);
    let na = sim.concentration(0);
    let f0 = sim.faces()[p[1]];
    println!(
        "Na_i at synapse {:.3}, Na_e {:.3}",
        na[f0.inside], na[f0.outside]
    );

    let geo: &VoxelGeometry = sim.geometry();
    let (h, hx) = (geo.grid.h[1], geo.grid.h[0]);
    let side = h; // one voxel across
    let n_ecs = geo.tags.iter().filter(|&&t| t == ECS).count() as f64;
    let box_area = (geo.grid.n[1] * geo.grid.n[2]) as f64 * h * h;
    assert_eq!(n_ecs as usize, geo.grid.len() - (length / hx) as usize);
    let c = IonConcentrations::KNP_EMI;
    let psi = psi_mv(params.membrane.temperature);
    let sigma =
        |na: f64, k: f64, cl: f64| (SPECIES[0].d * na + SPECIES[1].d * k + SPECIES[2].d * cl) / psi;
    let g_i = sigma(c.na_i, c.k_i, c.cl_i) * side * side;
    let g_e = sigma(c.na_e, c.k_e, c.cl_e) * (box_area - side * side);
    let g_ax = 1.0 / (1.0 / g_i + 1.0 / g_e);

    let m = params.membrane;
    let e_na = psi * (c.na_e / c.na_i).ln();
    let e_k = psi * (c.k_e / c.k_i).ln();
    let g_m = m.g_leak_na + m.g_leak_k;
    let v_rest = (m.g_leak_na * e_na + m.g_leak_k * e_k) / g_m;

    // Finite-volume cable for u = φ_M − V_rest on the axon's x-segments.
    let n = (length / hx) as usize;
    let (mut diag, mut rhs) = (vec![0.0; n], vec![0.0; n]);
    let off = g_ax / hx;
    for i in 0..n {
        let x = x0 + (i as f64 + 0.5) * hx;
        let mut area = 4.0 * side * hx;
        if i == 0 || i == n - 1 {
            area += side * side; // end cap
        }
        let gs = if x < x0 + syn_len { g_syn } else { 0.0 };
        diag[i] = UA_PER_CM2_TO_FLUX * (g_m + gs) * area;
        rhs[i] = UA_PER_CM2_TO_FLUX * gs * area * (e_na - v_rest);
        if i > 0 {
            diag[i] += off;
        }
        if i + 1 < n {
            diag[i] += off;
        }
    }
    // Thomas algorithm with constant off-diagonal -off.
    let mut cp = vec![0.0; n];
    cp[0] = -off / diag[0];
    rhs[0] /= diag[0];
    for i in 1..n {
        let den = diag[i] + off * cp[i - 1];
        cp[i] = -off / den;
        rhs[i] = (rhs[i] + off * rhs[i - 1]) / den;
    }
    for i in (0..n - 1).rev() {
        rhs[i] -= cp[i] * rhs[i + 1];
    }

    let lat = lateral_patches(&sim, 1);
    assert_eq!(lat.len(), n);
    let peak = rhs.iter().fold(0.0f64, |a, &b| a.max(b));
    let mut worst = 0.0f64;
    for (i, &(_, p)) in lat.iter().enumerate() {
        let u = sim.membrane_potential()[p] - v_rest;
        worst = worst.max((u - rhs[i]).abs());
        if i % 20 == 0 || i == n - 1 {
            println!(
                "x = {:5.1}: EMI {u:7.3} mV, cable {:7.3} mV",
                lat[i].0 - x0,
                rhs[i]
            );
        }
    }
    println!(
        "max difference {worst:.3} mV = {:.1}% of the peak depolarisation",
        100.0 * worst / peak
    );
    assert!(
        rhs[n - 1] < 0.8 * rhs[0],
        "the reference should decay along the axon"
    );
    assert!(
        worst < 0.03 * peak,
        "EMI deviates from the cable by {worst} mV"
    );
}
