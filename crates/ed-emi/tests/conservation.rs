//! Conservation properties of the KNP-EMI solver during activity.

use ed_core::Simulation;
use ed_emi::{AxonBundle, EmiParams, KnpEmi};

#[test]
fn conserves_ions_and_electroneutrality_during_activity() {
    let mut sim = KnpEmi::new(AxonBundle::default().build(), EmiParams::default());
    sim.stimulate(5, 0.5);
    sim.stimulate(1, 1.0);
    let before: Vec<f64> = (0..3).map(|s| sim.total_amount(s)).collect();
    let mut peak = f64::MIN;
    for _ in 0..60 {
        sim.step(1);
        peak = sim
            .membrane_potential()
            .iter()
            .copied()
            .fold(peak, f64::max);
    }
    assert!(peak > 20.0, "no action potential (peak {peak} mV)");
    for (s, b) in before.iter().enumerate() {
        let rel = (sim.total_amount(s) - b) / b;
        assert!(rel.abs() < 1e-12, "species {s}: relative change {rel:e}");
    }
    // Initial concentrations are neutral (Na + K − Cl = 0), and must stay so in every voxel.
    let (na, k, cl) = (
        sim.concentration(0),
        sim.concentration(1),
        sim.concentration(2),
    );
    let q = (0..na.len())
        .map(|v| (na[v] + k[v] - cl[v]).abs())
        .fold(0.0, f64::max);
    assert!(q < 1e-6, "max |Σ z c| = {q:e} mM");
}
