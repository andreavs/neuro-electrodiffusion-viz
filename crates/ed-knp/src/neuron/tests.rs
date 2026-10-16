use super::*;

/// Somatic spike times and peak potential over the next `ms`.
fn spike_times(n: &mut Neuron, ms: f64) -> (Vec<f64>, f64) {
    let steps = (ms / n.params().dt).round() as usize;
    let (mut times, mut peak) = (Vec::new(), f64::MIN);
    let mut above = n.v()[0] > 0.0;
    for _ in 0..steps {
        n.step(1);
        peak = peak.max(n.v()[0]);
        let now = n.v()[0] > 0.0;
        if now && !above {
            times.push(n.time_ms());
        }
        above = now;
    }
    (times, peak)
}

/// |Σ_j (I_Na + I_K + I_X + I_cap) + I_return| relative to the current magnitudes.
fn conservation_error(n: &Neuron) -> f64 {
    let mut sum = n.electrode_return();
    let mut mag = 1e-12;
    for j in 0..n.len() {
        sum += n.i_na()[j] + n.i_k()[j] + n.i_x()[j] + n.i_cap()[j];
        mag += n.i_na()[j].abs() + n.i_k()[j].abs() + n.i_cap()[j].abs();
    }
    (sum / mag).abs()
}

#[test]
fn stays_at_rest_without_input() {
    let mut n = Neuron::pyramidal();
    let v0 = n.v().to_vec();
    assert!((-80.0..-55.0).contains(&v0[0]), "rest {}", v0[0]);
    n.step(4000); // 100 ms
    for (a, b) in v0.iter().zip(n.v()) {
        assert!((a - b).abs() < 0.1, "{a} -> {b}");
    }
    assert!(conservation_error(&n) < 1e-10);
}

#[test]
fn somatic_step_fires_repeatedly() {
    let mut n = Neuron::pyramidal();
    n.i_stim = 2.0;
    let (spikes, peak) = spike_times(&mut n, 200.0);
    assert!(spikes.len() >= 4, "{spikes:?}");
    assert!(peak > 20.0, "peak {peak}");
}

#[test]
fn currents_are_conserved_every_step() {
    let mut n = Neuron::pyramidal();
    n.i_stim = 2.0;
    n.syn_rate_hz = 5000.0;
    let mut worst: f64 = 0.0;
    for _ in 0..4000 {
        n.step(1);
        worst = worst.max(conservation_error(&n));
    }
    assert!(worst < 1e-10, "relative imbalance {worst}");
    assert!((n.electrode_return() + 2.0).abs() < 1e-9);
}

#[test]
fn each_compartment_balances_its_axial_currents() {
    // Backward-Euler rows: membrane current out = axial current in (+ stim).
    let mut n = Neuron::pyramidal();
    n.i_stim = 3.0;
    n.syn_rate_hz = 20_000.0;
    for _ in 0..200 {
        n.step(1);
        let mut net_in = vec![0.0; n.len()];
        net_in[0] = n.i_stim;
        for i in 1..n.len() {
            let f = n.g_axial[i] * (n.v[n.parent[i]] - n.v[i]);
            net_in[i] += f;
            net_in[n.parent[i]] -= f;
        }
        for (i, inflow) in net_in.iter().enumerate() {
            let out = n.i_na[i] + n.i_k[i] + n.i_x[i] + n.i_cap[i];
            assert!(
                (out - inflow).abs() < 1e-9 * (1.0 + inflow.abs()),
                "row {i}: {out} vs {inflow}"
            );
        }
    }
}

#[test]
fn outputs_average_over_a_multi_step_call() {
    let mut n = Neuron::pyramidal();
    n.i_stim = 2.0;
    let mut m = n.clone();
    let mut mean = vec![0.0; n.len()];
    for _ in 0..4 {
        m.step(1);
        mean.iter_mut()
            .zip(m.i_k())
            .for_each(|(a, b)| *a += b / 4.0);
    }
    n.step(4);
    assert_eq!(n.v(), m.v());
    for (a, b) in n.i_k().iter().zip(&mean) {
        assert!((a - b).abs() < 1e-12 * (1.0 + b.abs()));
    }
    assert!(conservation_error(&n) < 1e-10);
}

#[test]
fn poisson_input_rate_controls_firing() {
    let rate = |hz: f64| {
        let mut n = Neuron::pyramidal();
        n.syn_rate_hz = hz;
        spike_times(&mut n, 1000.0).0.len()
    };
    let (none, low, high) = (rate(0.0), rate(2000.0), rate(10_000.0));
    assert_eq!(none, 0);
    assert!(low < high && high >= 20, "{low} vs {high} spikes/s");
}

#[test]
fn synaptic_events_follow_rate_and_area() {
    let mut n = Neuron::pyramidal();
    n.syn_rate_hz = 20_000.0;
    let (mut events, mut hits) = (0usize, vec![0usize; n.len()]);
    for _ in 0..40_000 {
        let before: Vec<f64> = (0..n.passive.len()).map(|j| n.passive.syn_g[j]).collect();
        n.deliver_synapses(n.time);
        for (hit, &(act, j)) in hits.iter_mut().zip(&n.slot) {
            assert!(!act || n.active.syn_g[j] == 0.0, "event on a non-dendrite");
            if !act && n.passive.syn_g[j] != before[j] {
                events += 1;
                *hit += 1;
            }
        }
        n.time += n.params().dt;
    }
    // 1 s at 20 kHz: 20000 ± 141 events (at most one counted per compartment
    // and step, which loses ≈ 0.1 %).
    assert!((events as f64 - 20_000.0).abs() < 600.0, "{events}");
    let comps = &n.morphology().compartments;
    let dend = |c: &&Compartment| c.kind == Kind::Dendrite;
    let total: f64 = comps.iter().filter(dend).map(|c| c.area()).sum();
    // Basal dendrites and the apical tuft each get their area share.
    for region in [|z: f64| z < 0.0, |z: f64| z > 765.0] {
        let (area, count) = comps
            .iter()
            .zip(&hits)
            .filter(|(c, _)| dend(c) && region(c.centre()[2]))
            .fold((0.0, 0), |(a, k), (c, h)| (a + c.area(), k + h));
        let expected = events as f64 * area / total;
        let err = (count as f64 - expected).abs();
        assert!(err < 5.0 * expected.sqrt(), "{count} vs {expected}");
    }
}

#[test]
#[ignore = "timing; run with --release -- --ignored --nocapture"]
fn step_time() {
    let mut n = Neuron::pyramidal();
    n.i_stim = 0.5;
    n.syn_rate_hz = 5000.0;
    let steps = 40_000;
    let t0 = std::time::Instant::now();
    n.step(steps);
    let us = t0.elapsed().as_secs_f64() * 1e6 / steps as f64;
    println!("{} compartments: {us:.2} µs/step", n.len());
    assert!(us < 1000.0);
}
