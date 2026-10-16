use super::*;
use ed_core::constants::F;

fn run(pop: &mut Population, dt: f64, ms: f64) {
    for _ in 0..(ms / dt).round() as usize {
        pop.step(dt);
    }
}

#[test]
fn baseline_is_a_steady_state() {
    for g_nap in [0.0, 0.05] {
        let mut pop = Population::new(
            3,
            PopulationParams {
                g_nap,
                ..PopulationParams::default()
            },
        );
        run(&mut pop, 2.0, 2000.0);
        for c in 0..6 {
            assert!((pop.v()[c] + 65.0).abs() < 1e-6, "V {}", pop.v()[c]);
            assert!((pop.na_i()[c] - 12.0).abs() < 1e-9);
            assert!((pop.k_i()[c] - 125.0).abs() < 1e-9);
            let net = pop.i_na()[c] + pop.i_k()[c] + pop.i_x()[c] + pop.i_cap()[c];
            assert!(pop.i_na()[c].abs() < 1e-9 && net.abs() < 1e-9);
        }
        // A quiet neuron takes one internal step per step.
        assert_eq!(pop.substeps(), 3);
    }
}

#[test]
fn driven_neuron_fires_and_ions_are_conserved() {
    let p = PopulationParams::default();
    let mut pop = Population::new(2, p.clone());
    pop.set_drive(0, 5.0);
    let amount = |pop: &Population, c: usize| {
        [pop.na_i()[c], pop.k_i()[c], pop.cl_i()[c]].map(|x| x * p.volume[c % 2])
    };
    let start: Vec<_> = (0..2).map(|c| amount(&pop, c)).collect();
    let mut released = [[0.0; 3]; 2];
    let (mut spikes, mut above) = (0, false);
    for _ in 0..2000 {
        pop.step(0.1);
        for (s, r) in released.iter_mut().enumerate() {
            for (k, cur) in [pop.i_na(), pop.i_k(), pop.i_x()].iter().enumerate() {
                r[k] += cur[s] * 0.1;
            }
            let net: f64 = [pop.i_na(), pop.i_k(), pop.i_x(), pop.i_cap()]
                .iter()
                .map(|a| a[0] + a[1])
                .sum();
            assert!(net.abs() < 1e-9, "net current {net}");
        }
        let up = pop.v()[0] > 0.0;
        spikes += usize::from(up && !above);
        above = up;
    }
    assert!(spikes > 5, "{spikes} spikes in 200 ms");
    assert!(pop.v()[2] < -64.9, "the undriven neuron stays at rest");
    let z = [1.0, 1.0, -1.0];
    for s in 0..2 {
        let now = amount(&pop, s);
        for k in 0..3 {
            let lost = (start[s][k] - now[k]) * F * 1e-6;
            assert!((lost - released[s][k] / z[k]).abs() < 1e-6 * released[s][k].abs().max(1.0));
        }
    }
    assert!(pop.na_i()[0] > 12.0 && pop.k_i()[0] < 125.0);
}

#[test]
fn adaptive_steps_match_a_fine_fixed_step() {
    let spikes_with = |h_max: f64, v_tol: f64| {
        let mut pop = Population::new(
            1,
            PopulationParams {
                h_max,
                v_tol,
                ..PopulationParams::default()
            },
        );
        pop.set_drive(0, 5.0);
        let (mut spikes, mut above, mut steps) = (0, false, 0);
        for _ in 0..500 {
            pop.step(1.0);
            steps += pop.substeps();
            let up = pop.v()[0] > -20.0;
            spikes += usize::from(up && !above);
            above = up;
        }
        (spikes, steps, pop.k_i()[0])
    };
    let (fine, fine_steps, k_fine) = spikes_with(0.025, 1e9);
    let (adaptive, steps, k) = spikes_with(2.0, 2.0);
    assert!(fine > 10);
    assert!(
        (adaptive as f64 - fine as f64).abs() <= 0.1 * fine as f64,
        "{adaptive} vs {fine} spikes"
    );
    assert!(
        (k - k_fine).abs() < 0.1 * (125.0 - k_fine),
        "{k} vs {k_fine}"
    );
    assert!(
        steps < fine_steps / 2,
        "{steps} vs {fine_steps} internal steps"
    );
}

#[test]
fn raised_potassium_depolarises() {
    let mut pop = Population::new(1, PopulationParams::default());
    pop.k_e.fill(12.0);
    run(&mut pop, 1.0, 50.0);
    assert!(pop.v()[1] > -58.0, "dendrite at {}", pop.v()[1]);
}
