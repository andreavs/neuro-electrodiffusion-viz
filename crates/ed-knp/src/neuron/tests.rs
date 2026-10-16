use super::*;

/// Somatic spike times and peak potential over the next `ms`.
fn spike_times(n: &mut Neuron, ms: f64) -> (Vec<f64>, f64) {
    let steps = (ms / n.params.dt).round() as usize;
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

/// Largest |Σ_j (I_Na + I_K + I_X + I_cap)| relative to Σ_j |I_cap|.
fn conservation_error(n: &Neuron) -> f64 {
    let mut sum = 0.0;
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
    n.stim_na = 2.0;
    let (spikes, peak) = spike_times(&mut n, 200.0);
    assert!(spikes.len() >= 4, "{spikes:?}");
    assert!(peak > 20.0, "peak {peak}");
}

#[test]
fn currents_are_conserved_every_step() {
    let mut n = Neuron::pyramidal();
    n.stim_na = 2.0;
    n.syn_rate_hz = 5000.0;
    let mut worst: f64 = 0.0;
    for _ in 0..4000 {
        n.step(1);
        worst = worst.max(conservation_error(&n));
    }
    assert!(worst < 1e-10, "relative imbalance {worst}");
    // The electrode return is booked as X⁻ at the soma.
    assert!((n.i_x()[0] + 2.0).abs() < 1e-9);
}

#[test]
fn poisson_input_rate_controls_firing() {
    let rate = |hz: f64| {
        let mut n = Neuron::pyramidal();
        n.syn_rate_hz = hz;
        spike_times(&mut n, 1000.0).0.len()
    };
    let (low, high) = (rate(0.0), rate(10_000.0));
    assert_eq!(low, 0);
    assert!(high >= 3, "{high} spikes/s");
}

#[test]
fn same_seed_is_deterministic() {
    let run = || {
        let mut n = Neuron::pyramidal();
        n.syn_rate_hz = 5000.0;
        n.step(2000);
        n.v().to_vec()
    };
    assert_eq!(run(), run());
}

#[test]
#[ignore = "timing; run with --release -- --ignored --nocapture"]
fn step_time() {
    let mut n = Neuron::pyramidal();
    n.syn_rate_hz = 5000.0;
    n.stim_na = 0.5;
    let steps = 40_000;
    let t0 = std::time::Instant::now();
    n.step(steps);
    let us = t0.elapsed().as_secs_f64() * 1e6 / steps as f64;
    println!("{} compartments: {us:.2} µs/step", n.len());
    assert!(us < 1000.0);
}
