//! Bindings for `ed-particles`: Ca²⁺ ions escaping a dendritic spine (Publication 3).
//!
//! [`ParticlesSim`] runs the population with the chosen field and, when a field is on, a
//! smaller zero-field reference population in lockstep so the view can overlay both.

use ed_core::Simulation;
use ed_particles::{
    ElectricField, Junction, NeckField, NoField, ParticleParams, ParticleSim, SpineGeometry,
    UniformField,
};
use wasm_bindgen::prelude::*;

/// Where the potential drop sits: nowhere, across the neck only, or along the whole spine.
#[derive(Debug, Clone, Copy, PartialEq)]
enum FieldKind {
    Off,
    Neck,
    Uniform,
}

impl FieldKind {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "off" => Ok(Self::Off),
            "neck" => Ok(Self::Neck),
            "uniform" => Ok(Self::Uniform),
            _ => Err(format!("unknown field kind {s:?}")),
        }
    }
}

#[wasm_bindgen]
pub struct ParticlesSim {
    geom: SpineGeometry,
    sim: ParticleSim<SpineGeometry>,
    /// Zero-field population for comparison; `None` while the field is off.
    reference: Option<ParticleSim<SpineGeometry>>,
    field: FieldKind,
    /// Head-minus-base potential drop, mV; positive pushes cations out.
    delta_v: f64,
    n_reference: usize,
}

#[wasm_bindgen]
impl ParticlesSim {
    /// `geometry` is "sharp" or "funnel"; `dt` in ms.
    #[wasm_bindgen(constructor)]
    pub fn new(geometry: &str, dt: f64, seed: u32) -> Result<ParticlesSim, JsError> {
        let geom = parse_geometry(geometry).map_err(js)?;
        let params = checked_params(dt).map_err(js)?;
        Ok(Self {
            geom,
            sim: ParticleSim::new(geom, params, u64::from(seed)),
            reference: None,
            field: FieldKind::Off,
            delta_v: 0.0,
            n_reference: 0,
        })
    }

    /// Sets the field as a potential drop `delta_v_mv` (head minus base) applied across
    /// the neck ("neck") or along the whole spine height ("uniform"). Takes effect from the
    /// next step; call [`ParticlesSim::release`] to restart the statistics.
    pub fn set_field(&mut self, kind: &str, delta_v_mv: f64) -> Result<(), JsError> {
        if !delta_v_mv.is_finite() {
            return Err(js("field strength must be finite".into()));
        }
        self.field = FieldKind::parse(kind).map_err(js)?;
        self.delta_v = delta_v_mv;
        let field = self.make_field();
        self.sim.set_field(BoxedField(field));
        Ok(())
    }

    pub fn set_dt(&mut self, dt: f64) -> Result<(), JsError> {
        let params = checked_params(dt).map_err(js)?;
        self.sim.set_params(params);
        if let Some(r) = &mut self.reference {
            r.set_params(params);
        }
        Ok(())
    }

    /// Restarts the experiment: `n` ions at the head centre, plus `n_reference` zero-field
    /// ions if a field is on.
    pub fn release(&mut self, n: usize, n_reference: usize) -> Result<(), JsError> {
        if n > ed_particles::sim::MAX_PARTICLES || n_reference > ed_particles::sim::MAX_PARTICLES {
            return Err(js("too many particles".into()));
        }
        self.sim.release(n);
        self.n_reference = n_reference;
        self.reference = (self.has_field() && n_reference > 0).then(|| {
            // Fixed seed: an independent stream from the main population's.
            let seed = 0x9E37_79B9_7F4A_7C15;
            let mut r = ParticleSim::new(self.geom, self.sim.params(), seed);
            r.release(n_reference);
            r
        });
        Ok(())
    }

    /// Advances both populations by up to `n` steps; stops early once every main-population
    /// ion has escaped. Returns the steps taken.
    pub fn step(&mut self, n: usize) -> usize {
        let mut taken = 0;
        while taken < n && self.sim.n_alive() > 0 {
            self.sim.step_once();
            if let Some(r) = &mut self.reference {
                r.step(1);
            }
            taken += 1;
        }
        taken
    }

    pub fn time_ms(&self) -> f64 {
        self.sim.time_ms()
    }
    pub fn n_alive(&self) -> usize {
        self.sim.n_alive()
    }
    pub fn n_released(&self) -> usize {
        self.sim.n_released()
    }
    pub fn has_reference(&self) -> bool {
        self.reference.is_some()
    }

    /// Interleaved xyz (µm) of the live ions. A fresh array each call.
    pub fn positions(&mut self) -> Vec<f32> {
        self.sim.positions_xyz().to_vec()
    }

    /// Per live ion: 1 in the head, 0 in the neck (same order as `positions`).
    pub fn regions(&self) -> Vec<f32> {
        let top = self.geom.neck_top();
        let mut out = Vec::with_capacity(self.sim.n_alive());
        self.sim
            .for_each_position(|p| out.push(if p[2] >= top { 1.0 } else { 0.0 }));
        out
    }

    /// Survival curve as interleaved (t ms, S) pairs.
    pub fn survival(&self, reference: bool) -> Vec<f32> {
        self.population(reference)
            .map(|s| {
                s.survival_curve()
                    .iter()
                    .flat_map(|&(t, f)| [t as f32, f as f32])
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Exit-time histogram over [0, t_max) ms as fractions of the released ions.
    pub fn histogram(&self, reference: bool, bins: usize, t_max: f64) -> Vec<f32> {
        let Some(s) = self.population(reference) else {
            return vec![0.0; bins];
        };
        let n = s.n_released().max(1) as f32;
        s.exit_time_histogram(bins, t_max)
            .into_iter()
            .map(|c| c as f32 / n)
            .collect()
    }

    /// `[mean, std_err, n_escaped, censored_mean]` of the exit times (ms); NaN where undefined.
    /// The censored mean also counts the ions still inside (exponential-tail estimate).
    pub fn mfpt(&self, reference: bool) -> Vec<f64> {
        let Some(s) = self.population(reference) else {
            return vec![f64::NAN, f64::NAN, 0.0, f64::NAN];
        };
        match (s.mfpt(), s.mfpt_censored()) {
            (Some(m), Some(c)) => vec![m.mean, m.std_err, m.n as f64, c.mean],
            _ => vec![f64::NAN, f64::NAN, 0.0, f64::NAN],
        }
    }

    /// The spine wall in the (r, z) half-plane, from the neck base (r = neck radius, z = 0)
    /// to the top of the head (r = 0), as interleaved pairs in µm. Revolve it about z to
    /// draw the surface.
    pub fn profile(&self, segments: usize) -> Vec<f32> {
        spine_profile(&self.geom, segments.max(4))
            .into_iter()
            .flat_map(|(r, z)| [r as f32, z as f32])
            .collect()
    }

    /// Height where the neck meets the head (µm).
    pub fn neck_top(&self) -> f64 {
        self.geom.neck_top()
    }
    pub fn neck_radius(&self) -> f64 {
        self.geom.neck_radius()
    }
}

impl ParticlesSim {
    fn has_field(&self) -> bool {
        self.field != FieldKind::Off && self.delta_v != 0.0
    }

    fn population(&self, reference: bool) -> Option<&ParticleSim<SpineGeometry>> {
        if reference {
            self.reference.as_ref()
        } else {
            Some(&self.sim)
        }
    }

    fn make_field(&self) -> Box<dyn ElectricField> {
        match self.field {
            FieldKind::Off => Box::new(NoField),
            FieldKind::Neck => Box::new(NeckField::for_spine(&self.geom, self.delta_v)),
            FieldKind::Uniform => {
                let height = self.geom.head_centre()[2] + self.geom.head_radius();
                Box::new(UniformField::axial_out(self.delta_v / height))
            }
        }
    }
}

/// Forwards to a boxed field so `set_field` can take a runtime choice.
struct BoxedField(Box<dyn ElectricField>);

impl ElectricField for BoxedField {
    fn field(&self, p: [f64; 3]) -> [f64; 3] {
        self.0.field(p)
    }
    fn field_batch(&self, x: &[f64], y: &[f64], z: &[f64], out: &mut [[f64; 3]]) {
        self.0.field_batch(x, y, z, out)
    }
    fn uniform(&self) -> Option<[f64; 3]> {
        self.0.uniform()
    }
}

fn js(msg: String) -> JsError {
    JsError::new(&msg)
}

fn parse_geometry(name: &str) -> Result<SpineGeometry, String> {
    match name {
        "sharp" => Ok(SpineGeometry::sharp()),
        "funnel" => Ok(SpineGeometry::funneled()),
        _ => Err(format!("unknown geometry {name:?}")),
    }
}

fn checked_params(dt: f64) -> Result<ParticleParams, String> {
    if !(dt.is_finite() && dt > 0.0 && dt <= 0.01) {
        return Err("time step must be in (0, 0.01] ms".into());
    }
    Ok(ParticleParams {
        dt,
        ..ParticleParams::default()
    })
}

/// Wall of `g` in (r, z): neck line, funnel arc (if any), then the head arc to the apex.
fn spine_profile(g: &SpineGeometry, segments: usize) -> Vec<(f64, f64)> {
    let a = g.neck_radius();
    let l = g.neck_length();
    let rh = g.head_radius();
    let zc = g.head_centre()[2];
    let mut pts = vec![(a, 0.0), (a, l)];
    if let Junction::Funnel { radius: rho } = g.junction() {
        // Tube circle centred at (a + ρ, l); the arc runs from (a, l) up to the head tangent.
        let ring = a + rho;
        let zt = g.neck_top();
        let rt = (rh * rh - (zt - zc).powi(2)).sqrt();
        let (start, end) = (std::f64::consts::PI, (zt - l).atan2(rt - ring));
        for i in 1..=segments / 4 {
            let th = start + (end - start) * i as f64 / (segments / 4) as f64;
            pts.push((ring + rho * th.cos(), l + rho * th.sin()));
        }
    }
    // Head arc by polar angle from +z, from the junction point to the apex.
    let &(r0, z0) = pts.last().expect("profile starts with two points");
    let phi0 = r0.atan2(z0 - zc);
    for i in 1..=segments {
        let phi = phi0 * (1.0 - i as f64 / segments as f64);
        pts.push((rh * phi.sin(), zc + rh * phi.cos()));
    }
    pts
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed_particles::Geometry;

    #[test]
    fn profile_lies_on_the_wall() {
        for g in [SpineGeometry::sharp(), SpineGeometry::funneled()] {
            let pts = spine_profile(&g, 64);
            assert!(pts.last().unwrap().0.abs() < 1e-12);
            for &(r, z) in &pts[1..] {
                let d = g.sdf([r, 0.0, z]);
                assert!(d.abs() < 1e-9, "sdf({r}, {z}) = {d} for {:?}", g.junction());
            }
            // Heights never decrease along the wall.
            assert!(pts.windows(2).all(|w| w[1].1 >= w[0].1 - 1e-12));
        }
    }

    #[test]
    fn statistics_and_frames_have_matching_sizes() {
        let mut s = ParticlesSim::new("sharp", 1e-3, 3).unwrap();
        s.set_field("neck", 20.0).unwrap();
        s.release(500, 200).unwrap();
        assert!(s.has_reference());
        assert_eq!(s.step(200), 200);
        let n = s.n_alive();
        assert_eq!(s.positions().len(), 3 * n);
        assert_eq!(s.regions().len(), n);
        assert!(s.regions().iter().all(|&v| v == 0.0 || v == 1.0));
        let surv = s.survival(false);
        assert_eq!(surv.len() % 2, 0);
        assert_eq!(&surv[..2], &[0.0, 1.0]);
        assert!(!s.survival(true).is_empty());
        let h = s.histogram(false, 10, s.time_ms());
        let escaped: f32 = h.iter().sum();
        assert!((escaped - (500 - n) as f32 / 500.0).abs() < 1e-6);
        let m = s.mfpt(false);
        assert_eq!(m[2] as usize, 500 - n);
    }

    #[test]
    fn field_off_has_no_reference_and_bad_input_is_rejected() {
        let mut s = ParticlesSim::new("funnel", 1e-4, 1).unwrap();
        s.release(10, 10).unwrap();
        assert!(!s.has_reference());
        assert!(s.survival(true).is_empty());
        assert!(s.mfpt(true)[0].is_nan());
        assert!(parse_geometry("cube").is_err());
        assert!(checked_params(0.0).is_err());
        assert!(FieldKind::parse("sideways").is_err());
    }

    #[test]
    fn outward_field_empties_the_spine_faster() {
        let run = |dv: f64| {
            let mut s = ParticlesSim::new("sharp", 1e-3, 5).unwrap();
            s.set_field("uniform", dv).unwrap();
            s.release(400, 0).unwrap();
            s.step(3000);
            s.n_alive()
        };
        assert!(run(40.0) < run(-40.0));
    }
}
