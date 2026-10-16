use super::*;

/// Run `patches` for `t_end` ms in steps of `dt` with `substeps` each, calling
/// `ext(t)` for a uniform injected current. Returns φ_M of patch 0 per step.
fn run(
    patches: &mut MembranePatches,
    t_end: f64,
    dt: f64,
    substeps: usize,
    ext: impl Fn(f64) -> f64,
) -> Vec<f64> {
    let steps = (t_end / dt).round() as usize;
    let mut i_ext = vec![0.0; patches.len()];
    let mut trace = Vec::with_capacity(steps);
    for k in 0..steps {
        let t = k as f64 * dt;
        i_ext.fill(ext(t));
        patches.advance(t, dt, substeps, Some(&i_ext));
        trace.push(patches.phi_m[0]);
    }
    trace
}

fn peak(trace: &[f64]) -> f64 {
    trace.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn hh_patch() -> MembranePatches {
    MembranePatches::new(
        1,
        MembraneParams::HODGKIN_HUXLEY,
        IonConcentrations::KNP_EMI,
    )
}

fn synaptic_ap_peak(dt: f64) -> f64 {
    let mut p = hh_patch();
    p.set_synapse(0, 4.0, 1.0);
    peak(&run(&mut p, 20.0, dt, 1, |_| 0.0))
}

#[test]
fn rest_state_is_near_thesis_initial_state() {
    let rest = rest_state(&MembraneParams::HODGKIN_HUXLEY, &IonConcentrations::KNP_EMI).unwrap();
    assert!(
        (rest.phi_m - PatchState::KNP_EMI.phi_m).abs() < 2.5,
        "{rest:?}"
    );
}

#[test]
fn resting_patch_stays_at_rest_for_100_ms() {
    let mut p = hh_patch();
    let phi0 = p.phi_m[0];
    // PDE step 0.1 ms with 25 ODE substeps, as in KNP-EMI.
    let trace = run(&mut p, 100.0, 0.1, 25, |_| 0.0);
    let drift = trace.iter().map(|v| (v - phi0).abs()).fold(0.0, f64::max);
    assert!(drift < 0.1, "drift {drift} mV");
}

#[test]
fn thesis_initial_state_relaxes_to_rest() {
    let mut p = MembranePatches::with_state(
        1,
        MembraneParams::HODGKIN_HUXLEY,
        IonConcentrations::KNP_EMI,
        PatchState::KNP_EMI,
    );
    let rest = rest_state(&p.params, &IonConcentrations::KNP_EMI).unwrap();
    let trace = run(&mut p, 100.0, 0.1, 25, |_| 0.0);
    assert!(peak(&trace) < -60.0, "no spurious spike");
    assert!((trace.last().unwrap() - rest.phi_m).abs() < 0.1);
}

#[test]
fn synaptic_input_fires_action_potential_and_returns_to_rest() {
    let mut p = hh_patch();
    let rest = p.phi_m[0];
    p.set_synapse(0, 4.0, 1.0);
    let trace = run(&mut p, 100.0, 0.1, 25, |_| 0.0);
    assert!(peak(&trace) > 20.0, "peak {}", peak(&trace));
    assert!(
        (trace.last().unwrap() - rest).abs() < 1.0,
        "end {}",
        trace.last().unwrap()
    );
}

#[test]
fn current_pulse_fires_but_subthreshold_pulse_does_not() {
    let pulse = |amp: f64| move |t: f64| if (1.0..2.0).contains(&t) { amp } else { 0.0 };
    let mut p = hh_patch();
    let strong = run(&mut p, 30.0, 0.01, 1, pulse(20.0));
    assert!(
        peak(&strong) > 20.0,
        "suprathreshold peak {}",
        peak(&strong)
    );

    let mut p = hh_patch();
    // Threshold for a 1 ms pulse is ≈ 15.7 µA/cm²; 12 is just below it.
    let weak = run(&mut p, 30.0, 0.01, 1, pulse(12.0));
    assert!(peak(&weak) < -55.0, "subthreshold peak {}", peak(&weak));
}

#[test]
fn weak_synapse_does_not_fire() {
    let mut p = hh_patch();
    p.set_synapse(0, 0.05, 1.0);
    let trace = run(&mut p, 30.0, 0.1, 25, |_| 0.0);
    assert!(peak(&trace) < -55.0, "peak {}", peak(&trace));
}

#[test]
fn currents_at_rest_balance_with_correct_signs() {
    let p = hh_patch();
    assert!(p.i_ch[0].abs() < 1e-9, "I_ch = {}", p.i_ch[0]);
    assert!(p.i_na[0] < 0.0, "Na⁺ leaks inward at rest");
    assert!(p.i_k[0] > 0.0, "K⁺ leaks outward at rest");
    // g_ch is the slope of I_ch with frozen gates.
    let mut q = p.clone();
    q.phi_m[0] += 1e-3;
    q.compute_currents(0.0);
    let slope = (q.i_ch[0] - p.i_ch[0]) / 1e-3;
    assert!((slope - p.g_ch[0]).abs() < 1e-9, "{slope} vs {}", p.g_ch[0]);
}

#[test]
fn per_ion_currents_sum_to_total() {
    let mut p = MembranePatches::new(
        3,
        MembraneParams::HODGKIN_HUXLEY,
        IonConcentrations::KNP_EMI,
    );
    p.params.g_leak_cl = 0.1;
    p.set_synapse(1, 4.0, 0.0);
    for k in 0..50 {
        let t = k as f64 * 0.1;
        p.advance(t, 0.1, 25, None);
        for i in 0..p.len() {
            let sum = p.i_na[i] + p.i_k[i] + p.i_cl[i];
            assert!((sum - p.i_ch[i]).abs() <= 1e-12 * sum.abs().max(1.0));
        }
    }
    assert!(p.i_cl[0] != 0.0);
}

#[test]
fn peak_is_insensitive_to_substep_size() {
    let a = synaptic_ap_peak(0.01);
    let b = synaptic_ap_peak(0.004);
    assert!((a - b).abs() < 1.0, "peaks {a} vs {b}");
}

#[test]
fn substepping_matches_fine_steps() {
    let mut coarse = hh_patch();
    coarse.set_synapse(0, 4.0, 1.0);
    let mut fine = coarse.clone();
    let a = run(&mut coarse, 20.0, 0.1, 25, |_| 0.0);
    let b = run(&mut fine, 20.0, 0.004, 1, |_| 0.0);
    // Same substep length and sample times, so identical up to rounding.
    let worst = a
        .iter()
        .enumerate()
        .map(|(k, v)| (v - b[25 * k + 24]).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-9, "max diff {worst} mV");
}

#[test]
fn large_steps_remain_bounded() {
    let mut p = hh_patch();
    p.set_synapse(0, 4.0, 1.0);
    let trace = run(&mut p, 50.0, 0.5, 1, |_| 0.0);
    assert!(trace
        .iter()
        .all(|v| v.is_finite() && (-100.0..60.0).contains(v)));
    for x in [&p.m, &p.h, &p.n] {
        assert!((0.0..=1.0).contains(&x[0]));
    }
}

#[test]
fn passive_patch_relaxes_to_weighted_reversal_with_membrane_time_constant() {
    let params = MembraneParams::PASSIVE;
    let conc = IonConcentrations::KNP_EMI;
    let psi = psi_mv(params.temperature);
    let (e_na, e_k, _) = nernst_all(&conc, psi);
    let g = params.g_leak_na + params.g_leak_k;
    let e_rest = (params.g_leak_na * e_na + params.g_leak_k * e_k) / g;

    let rest = rest_state(&params, &conc).unwrap();
    assert!((rest.phi_m - e_rest).abs() < 1e-9);

    let phi0 = -40.0;
    let mut p = MembranePatches::with_state(
        1,
        params,
        conc,
        PatchState {
            phi_m: phi0,
            ..rest
        },
    );
    let tau = params.c_m / g;
    p.advance(0.0, tau, 7, None);
    let expected = e_rest + (phi0 - e_rest) * (-1.0f64).exp();
    assert!(
        (p.phi_m[0] - expected).abs() < 1e-9,
        "{} vs {expected}",
        p.phi_m[0]
    );
    // Passive patches never touch the gates.
    assert_eq!(p.m[0], rest.m);
}

#[test]
fn patches_are_independent() {
    let mut p = MembranePatches::new(
        4,
        MembraneParams::HODGKIN_HUXLEY,
        IonConcentrations::KNP_EMI,
    );
    let rest = p.phi_m[0];
    p.set_synapse(2, 4.0, 1.0);
    let mut max = [f64::NEG_INFINITY; 4];
    for k in 0..100 {
        p.advance(k as f64 * 0.1, 0.1, 25, None);
        for (m, v) in max.iter_mut().zip(&p.phi_m) {
            *m = m.max(*v);
        }
    }
    assert!(max[2] > 20.0);
    for i in [0, 1, 3] {
        assert!((max[i] - rest).abs() < 1e-9, "patch {i} moved");
    }
}

#[test]
fn nernst_tracks_concentration_changes() {
    let mut p = MembranePatches::new(1, MembraneParams::PASSIVE, IonConcentrations::KNP_EMI);
    let before = p.i_k[0];
    // Raising extracellular K⁺ depolarises E_K, so the outward K⁺ current drops.
    p.k_e[0] = 20.0;
    p.compute_currents(0.0);
    assert!(p.i_k[0] < before);
}

#[test]
fn synaptic_conductance_decays_exponentially() {
    let mut p = hh_patch();
    p.set_synapse(0, 4.0, 1.0);
    assert_eq!(p.synaptic_conductance(0, 0.5), 0.0);
    assert!((p.synaptic_conductance(0, 1.0) - 4.0).abs() < 1e-12);
    assert!((p.synaptic_conductance(0, 3.0) - 4.0 * (-1.0f64).exp()).abs() < 1e-12);
}
