//! Bindings for `ed_knp::spreading`: thousands of reduced neurons coupled both ways to the
//! tissue, for the spreading-depression scenario.
//!
//! [`SdSim`] owns an [`SdTissue`]. Positions are in grid coordinates (µm, the box spans
//! `[0, size]` on each axis); fields are flat, x fastest.

use ed_core::Simulation;
use ed_knp::spreading::{SdConfig, SdTissue};
use ed_knp::tissue::{K, NA, X};
use wasm_bindgen::prelude::*;

/// Accepted ranges of the creation and run-time parameters.
const NEURONS: (usize, usize) = (1, 20_000);
const G_NAP: (f64, f64) = (0.0, 0.3);
const PUMP: (f64, f64) = (0.0, 3.0);
const BUFFERING: (f64, f64) = (0.0, 0.01);
const DRIVE_NS: (f64, f64) = (0.0, 100.0);

/// `Err` with a message unless `lo <= v <= hi` (kept free of `JsError`, which only exists
/// in a browser, so native tests can call it).
fn in_range(name: &str, v: f64, (lo, hi): (f64, f64)) -> Result<(), String> {
    if (lo..=hi).contains(&v) {
        Ok(())
    } else {
        Err(format!("{name} must be within {lo}–{hi}"))
    }
}

fn check(name: &str, v: f64, range: (f64, f64)) -> Result<(), JsError> {
    in_range(name, v, range).map_err(|e| JsError::new(&e))
}

#[wasm_bindgen]
pub struct SdSim {
    tissue: SdTissue,
}

#[wasm_bindgen]
impl SdSim {
    /// `neurons` cells placed with `seed`, with persistent Na⁺ conductance `g_nap` (mS/cm²).
    /// These fix the calibrated resting state, so changing them needs a new simulation.
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u32, neurons: usize, g_nap: f64) -> Result<SdSim, JsError> {
        if !(NEURONS.0..=NEURONS.1).contains(&neurons) {
            return Err(JsError::new(&format!(
                "neuron count must be within {}–{}",
                NEURONS.0, NEURONS.1
            )));
        }
        check("persistent Na⁺ conductance", g_nap, G_NAP)?;
        let mut cfg = SdConfig {
            neurons,
            seed: u64::from(seed),
            ..SdConfig::default()
        };
        cfg.population.g_nap = g_nap;
        Ok(Self {
            tissue: SdTissue::new(&cfg),
        })
    }

    /// Drives the `count` neurons nearest the centre of the soma layer with `g_ns` nS each.
    pub fn set_drive(&mut self, count: usize, g_ns: f64) -> Result<(), JsError> {
        check("drive", g_ns, DRIVE_NS)?;
        let s = self.tissue.knp.grid().extent();
        let count = count.min(self.tissue.pop.len());
        self.tissue
            .set_drive([s[0] / 2.0, s[1] / 2.0, s[2] / 4.0], count, g_ns);
        Ok(())
    }
    /// Pump capacity relative to the calibrated one.
    pub fn set_pump(&mut self, scale: f64) -> Result<(), JsError> {
        check("pump strength", scale, PUMP)?;
        self.tissue.pop.set_pump_scale(scale);
        Ok(())
    }
    /// Glial KCl uptake rate, 1/ms.
    pub fn set_buffering(&mut self, rate: f64) -> Result<(), JsError> {
        check("buffering", rate, BUFFERING)?;
        self.tissue.set_buffering(rate);
        Ok(())
    }

    pub fn step(&mut self, n: usize) {
        self.tissue.step(n);
    }
    pub fn dt(&self) -> f64 {
        self.tissue.knp.dt()
    }
    pub fn time_ms(&self) -> f64 {
        self.tissue.time_ms()
    }
    /// Internal neuron steps per neuron in the last step (cost indicator).
    pub fn substeps_per_neuron(&self) -> f64 {
        self.tissue.pop.substeps() as f64 / self.tissue.pop.len() as f64
    }
    /// Fraction of neurons in sustained depolarisation.
    pub fn depolarised(&self) -> f64 {
        self.tissue.depolarised_fraction()
    }
    /// Cells per axis `[nx, ny, nz]`.
    pub fn shape(&self) -> Vec<u32> {
        self.tissue.knp.grid().n.iter().map(|&n| n as u32).collect()
    }
    /// Box size per axis, µm.
    pub fn size(&self) -> Vec<f64> {
        self.tissue.knp.grid().extent().to_vec()
    }
    /// A whole field: "k", "na", "x" (mM) or "phi" (mV).
    pub fn field(&self, name: &str) -> Result<Vec<f32>, JsError> {
        let knp = &self.tissue.knp;
        let f = match name {
            "k" => knp.concentration(K),
            "na" => knp.concentration(NA),
            "x" => knp.concentration(X),
            "phi" => knp.potential(),
            _ => return Err(JsError::new(&format!("unknown field {name:?}"))),
        };
        Ok(f.iter().map(|&v| v as f32).collect())
    }
    /// Soma positions, xyz per neuron, µm.
    pub fn somata(&self) -> Vec<f32> {
        let p = self.tissue.positions();
        p.iter().step_by(2).flatten().map(|&x| x as f32).collect()
    }
    /// Somatic membrane potential per neuron, mV.
    pub fn v_soma(&self) -> Vec<f32> {
        let v = self.tissue.pop.v();
        v.iter().step_by(2).map(|&x| x as f32).collect()
    }
    /// Indices of the driven neurons.
    pub fn driven(&self) -> Vec<u32> {
        self.tissue.driven().iter().map(|&i| i as u32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_and_neurons_have_consistent_sizes() {
        let mut s = SdSim::new(1, 200, 0.25).unwrap();
        s.set_drive(3, 20.0).unwrap();
        s.step(5);
        assert!((s.time_ms() - 5.0 * s.dt()).abs() < 1e-9);
        let n = s.shape();
        let len = (n[0] * n[1] * n[2]) as usize;
        for f in ["k", "na", "x", "phi"] {
            assert_eq!(s.field(f).unwrap().len(), len);
        }
        assert_eq!(s.somata().len(), 3 * 200);
        assert_eq!(s.v_soma().len(), 200);
        assert_eq!(s.driven().len(), 3);
        // Driven cells sit closest to the centre of the soma layer.
        let size = s.size();
        let centre = [size[0] / 2.0, size[1] / 2.0, size[2] / 4.0];
        let p = s.somata();
        let d = |i: usize| {
            (0..3)
                .map(|a| (p[3 * i + a] as f64 - centre[a]).powi(2))
                .sum::<f64>()
        };
        let far = s
            .driven()
            .iter()
            .map(|&i| d(i as usize))
            .fold(0.0, f64::max);
        assert_eq!((0..200).filter(|&i| d(i) <= far).count(), 3);
        assert!(in_range("pump", -1.0, PUMP).is_err() && in_range("g", 0.3, G_NAP).is_ok());
    }
}
