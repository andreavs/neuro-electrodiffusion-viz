//! Bindings for `ed_knp::spreading`: thousands of reduced neurons coupled both ways to the
//! tissue, for the spreading-depression scenario.
//!
//! [`SdSim`] owns an [`SdTissue`]. Positions are in grid coordinates (µm, the box spans
//! `[0, size]` on each axis); fields are flat, x fastest.

use ed_core::Simulation;
use ed_knp::glia::GliaConfig;
use ed_knp::spreading::{SdConfig, SdTissue};
use ed_knp::tissue::{K, NA, X};
use wasm_bindgen::prelude::*;

/// Accepted ranges of the creation and run-time parameters.
const NEURONS: (usize, usize) = (1, 20_000);
const G_NAP: (f64, f64) = (0.0, 0.3);
const PUMP: (f64, f64) = (0.0, 3.0);
const BUFFERING: (f64, f64) = (0.0, 0.01);
const DRIVE_NS: (f64, f64) = (0.0, 100.0);
const GLIA: (f64, f64) = (0.0, 8.0);
const COUPLING: (f64, f64) = (0.01, 1.0);

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
    /// `neurons` cells placed with `seed`, with persistent Na⁺ conductance `g_nap` (mS/cm²),
    /// and astrocytes with `glia` µm² of membrane per µm³ of tissue (0 for none) and
    /// gap-junction `coupling` (1 = Halnes et al.'s syncytium). These fix the calibrated
    /// resting state, so changing them needs a new simulation.
    #[wasm_bindgen(constructor)]
    pub fn new(
        seed: u32,
        neurons: usize,
        g_nap: f64,
        glia: f64,
        coupling: f64,
    ) -> Result<SdSim, JsError> {
        if !(NEURONS.0..=NEURONS.1).contains(&neurons) {
            return Err(JsError::new(&format!(
                "neuron count must be within {}–{}",
                NEURONS.0, NEURONS.1
            )));
        }
        check("persistent Na⁺ conductance", g_nap, G_NAP)?;
        check("astrocyte membrane area", glia, GLIA)?;
        check("gap-junction coupling", coupling, COUPLING)?;
        let mut cfg = SdConfig {
            neurons,
            seed: u64::from(seed),
            glia: (glia > 0.0).then(|| GliaConfig {
                area_per_volume: glia,
                coupling,
                ..GliaConfig::default()
            }),
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
    /// A whole field: "k", "na", "x" (mM) or "phi" (mV) of the extracellular space, or, with
    /// astrocytes, their membrane potential "vg" (mV) or K⁺ "kg" (mM).
    pub fn field(&self, name: &str) -> Result<Vec<f32>, JsError> {
        self.field_values(name)
            .map(|f| f.iter().map(|&v| v as f32).collect())
            .map_err(|e| JsError::new(&e))
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

impl SdSim {
    fn field_values(&self, name: &str) -> Result<&[f64], String> {
        let knp = &self.tissue.knp;
        let glia = || {
            self.tissue
                .glia
                .as_ref()
                .ok_or_else(|| format!("field {name:?} needs astrocytes"))
        };
        Ok(match name {
            "k" => knp.concentration(K),
            "na" => knp.concentration(NA),
            "x" => knp.concentration(X),
            "phi" => knp.potential(),
            "vg" => glia()?.potential(),
            "kg" => glia()?.interior().concentration(K),
            _ => return Err(format!("unknown field {name:?}")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_and_neurons_have_consistent_sizes() {
        let mut s = SdSim::new(1, 200, 0.25, 0.0, 1.0).unwrap();
        s.set_drive(3, 20.0).unwrap();
        s.step(5);
        assert!((s.time_ms() - 5.0 * s.dt()).abs() < 1e-9);
        let n = s.shape();
        let len = (n[0] * n[1] * n[2]) as usize;
        for f in ["k", "na", "x", "phi"] {
            assert_eq!(s.field_values(f).unwrap().len(), len);
        }
        assert!(s.field_values("vg").is_err());
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

    #[test]
    fn astrocytes_add_their_fields() {
        let mut s = SdSim::new(1, 200, 0.25, 1.0, 0.5).unwrap();
        s.set_drive(3, 20.0).unwrap();
        s.step(5);
        let len = s.field_values("k").unwrap().len();
        let v = s.field_values("vg").unwrap();
        assert_eq!(v.len(), len);
        assert!(v.iter().all(|v| (-90.0..-40.0).contains(v)), "{v:?}");
        assert!(s.field_values("kg").unwrap().iter().all(|&k| k > 90.0));
        assert!(in_range("coupling", 0.0, COUPLING).is_err());
    }
}
