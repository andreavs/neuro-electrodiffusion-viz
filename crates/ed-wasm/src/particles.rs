//! Bindings for `ed-particles`: Ca²⁺ ions escaping a dendritic spine (Publication 3).
//!
//! [`ParticlesSim`] runs the population with the chosen field and, when a field is on, a
//! smaller zero-field reference population in lockstep so the view can overlay both.
//! The neck either ends in an absorbing base or opens into a parent dendrite, where escaped
//! ions keep diffusing; either way the first passage through the neck base is what is timed.

use ed_core::Simulation;
use ed_particles::{
    ElectricField, Geometry, Junction, NeckField, NoField, ParticleParams, ParticleSim,
    SegmentExit, SpineGeometry, SpineOnDendrite, UniformField, Vec3,
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
    geom: Domain,
    sim: ParticleSim<Domain>,
    /// Zero-field population for comparison; `None` while the field is off.
    reference: Option<ParticleSim<Domain>>,
    field: FieldKind,
    /// Head-minus-base potential drop, mV; positive pushes cations out.
    delta_v: f64,
    seed: u64,
    /// Releases so far; varies the reference population's random stream.
    releases: u64,
}

#[wasm_bindgen]
impl ParticlesSim {
    /// `geometry` is "sharp" or "funnel"; with `dendrite` the neck opens into a parent
    /// dendrite instead of an absorbing base. `dt` in ms.
    #[wasm_bindgen(constructor)]
    pub fn new(
        geometry: &str,
        dendrite: bool,
        dt: f64,
        seed: u32,
    ) -> Result<ParticlesSim, JsError> {
        let spine = parse_geometry(geometry).map_err(js)?;
        let geom = if dendrite {
            Domain::Dendrite(SpineOnDendrite::with_default_dendrite(spine))
        } else {
            Domain::Spine(spine)
        };
        let params = checked_params(dt).map_err(js)?;
        Ok(Self {
            geom,
            sim: ParticleSim::new(geom, params, u64::from(seed)),
            reference: None,
            field: FieldKind::Off,
            delta_v: 0.0,
            seed: u64::from(seed),
            releases: 0,
        })
    }

    /// Sets the field as a potential drop `delta_v_mv` (head minus base) applied across
    /// the neck ("neck") or along the whole spine height ("uniform"). Takes effect from the
    /// next step; call [`ParticlesSim::release`] to restart the statistics.
    pub fn set_field(&mut self, kind: &str, delta_v_mv: f64) -> Result<(), JsError> {
        if delta_v_mv.is_nan() || delta_v_mv.abs() > MAX_DROP_MV {
            return Err(js(format!(
                "potential drop must be within ±{MAX_DROP_MV} mV"
            )));
        }
        self.field = FieldKind::parse(kind).map_err(js)?;
        self.delta_v = delta_v_mv;
        let field = self.make_field();
        self.sim.set_field(BoxedField(field));
        if !self.has_field() {
            self.reference = None;
        }
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
    /// ions if a field is on (switching the field on later adds no reference until the
    /// next release).
    pub fn release(&mut self, n: usize, n_reference: usize) -> Result<(), JsError> {
        if n > ed_particles::sim::MAX_PARTICLES || n_reference > ed_particles::sim::MAX_PARTICLES {
            return Err(js("too many particles".into()));
        }
        self.sim.release(n);
        self.releases += 1;
        self.reference = (self.has_field() && n_reference > 0).then(|| {
            // A stream independent of the main population's, different for every release.
            let seed = self.seed ^ 0x9E37_79B9_7F4A_7C15 ^ (self.releases << 32);
            // Only its exit times are shown, so it needs no dendrite: the absorbing base
            // records the same first passage without moving the escaped ions on.
            let geom = Domain::Spine(*self.spine());
            let mut r = ParticleSim::new(geom, self.sim.params(), seed);
            r.release(n_reference);
            r
        });
        Ok(())
    }

    /// Advances both populations by up to `n` steps. The reference stops once all its ions
    /// have escaped; the main population runs on while escaped ions are still in the
    /// dendrite. Returns the steps taken (0 when nothing is left to move).
    pub fn step(&mut self, n: usize) -> usize {
        let mut taken = 0;
        while taken < n
            && (self.sim.n_alive() + self.sim.n_escaped_moving() > 0 || self.reference_alive() > 0)
        {
            // `step` skips an empty population, so its clock stops at its last exit.
            self.sim.step(1);
            if let Some(r) = self.reference.as_mut().filter(|r| r.n_alive() > 0) {
                r.step(1);
            }
            taken += 1;
        }
        taken
    }

    /// Simulated time of whichever population is still running, ms.
    pub fn time_ms(&self) -> f64 {
        let r = self.reference.as_ref().map_or(0.0, |r| r.time_ms());
        self.sim.time_ms().max(r)
    }
    /// Time until the last ion of either population escaped (the current time while some
    /// are inside), ms: the end of the survival curves.
    pub fn passage_time_ms(&self) -> f64 {
        let r = self.reference.as_ref().map_or(0.0, |r| r.passage_time());
        self.sim.passage_time().max(r)
    }
    /// Ions still inside the spine.
    pub fn n_alive(&self) -> usize {
        self.sim.n_alive()
    }
    /// Escaped ions still diffusing in the dendrite (0 with an absorbing base).
    pub fn n_in_dendrite(&self) -> usize {
        self.sim.n_escaped_moving()
    }
    pub fn n_released(&self) -> usize {
        self.sim.n_released()
    }
    pub fn has_reference(&self) -> bool {
        self.reference.is_some()
    }

    /// Interleaved xyz (µm) of the ions in the spine, then of those in the dendrite.
    /// A fresh array each call.
    pub fn positions(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(3 * (self.sim.n_alive() + self.sim.n_escaped_moving()));
        let mut push = |p: Vec3| out.extend([p[0] as f32, p[1] as f32, p[2] as f32]);
        self.sim.for_each_position(&mut push);
        self.sim.for_each_escaped_position(&mut push);
        out
    }

    /// Per ion, in the order of `positions`: 0 in the neck, 1 in the head, 2 escaped and in
    /// the dendrite.
    pub fn regions(&self) -> Vec<f32> {
        let top = self.spine().neck_top();
        let mut out = Vec::with_capacity(self.sim.n_alive() + self.sim.n_escaped_moving());
        let region = |p: Vec3| if p[2] >= top { 1.0 } else { 0.0 };
        self.sim.for_each_position(|p| out.push(region(p)));
        // Escaped ions that wander back up the neck take the spine's colours again.
        self.sim
            .for_each_escaped_position(|p| out.push(if p[2] < 0.0 { 2.0 } else { region(p) }));
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

    /// Exit-time histogram with `bins` equal bins over [0, t_max) ms (later exits land in the
    /// last bin), as the fraction of released ions per bin.
    pub fn histogram(&self, reference: bool, bins: usize, t_max: f64) -> Vec<f32> {
        let bins = bins.min(MAX_BINS);
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
        spine_profile(self.spine(), segments.clamp(4, MAX_BINS))
            .into_iter()
            .flat_map(|(r, z)| [r as f32, z as f32])
            .collect()
    }

    /// Height where the neck meets the head (µm).
    pub fn neck_top(&self) -> f64 {
        self.spine().neck_top()
    }
    pub fn neck_radius(&self) -> f64 {
        self.spine().neck_radius()
    }
    /// Dendrite radius and half-length along x (µm); 0 with an absorbing base.
    pub fn dendrite_radius(&self) -> f64 {
        self.dendrite().map_or(0.0, |d| d.radius())
    }
    pub fn dendrite_half_length(&self) -> f64 {
        self.dendrite().map_or(0.0, |d| d.half_length())
    }
}

impl ParticlesSim {
    fn spine(&self) -> &SpineGeometry {
        match &self.geom {
            Domain::Spine(s) => s,
            Domain::Dendrite(d) => d.spine(),
        }
    }

    fn dendrite(&self) -> Option<&SpineOnDendrite> {
        match &self.geom {
            Domain::Spine(_) => None,
            Domain::Dendrite(d) => Some(d),
        }
    }

    fn reference_alive(&self) -> usize {
        self.reference.as_ref().map_or(0, |r| r.n_alive())
    }

    fn has_field(&self) -> bool {
        self.field != FieldKind::Off && self.delta_v != 0.0
    }

    fn population(&self, reference: bool) -> Option<&ParticleSim<Domain>> {
        if reference {
            self.reference.as_ref()
        } else {
            Some(&self.sim)
        }
    }

    fn make_field(&self) -> Box<dyn ElectricField> {
        match self.field {
            FieldKind::Off => Box::new(NoField),
            FieldKind::Neck => Box::new(NeckField::for_spine(self.spine(), self.delta_v)),
            FieldKind::Uniform => {
                let spine = self.spine();
                let height = spine.head_centre()[2] + spine.head_radius();
                let e_out = self.delta_v / height;
                if self.dendrite().is_some() {
                    // The drop is across the spine only; the dendrite stays field-free.
                    Box::new(NeckField {
                        e_out,
                        z_top: height,
                    })
                } else {
                    Box::new(UniformField::axial_out(e_out))
                }
            }
        }
    }
}

/// The spine alone or on its dendrite, chosen at run time.
#[derive(Debug, Clone, Copy)]
enum Domain {
    Spine(SpineGeometry),
    Dendrite(SpineOnDendrite),
}

macro_rules! delegate {
    ($self:ident, $g:ident => $e:expr) => {
        match $self {
            Domain::Spine($g) => $e,
            Domain::Dendrite($g) => $e,
        }
    };
}

impl Geometry for Domain {
    fn sdf(&self, p: Vec3) -> f64 {
        delegate!(self, g => g.sdf(p))
    }
    #[inline]
    fn contains(&self, p: Vec3) -> bool {
        delegate!(self, g => g.contains(p))
    }
    fn absorbs(&self, b: Vec3) -> bool {
        delegate!(self, g => g.absorbs(b))
    }
    #[inline]
    fn escaped(&self, p: Vec3) -> bool {
        delegate!(self, g => g.escaped(p))
    }
    fn normal(&self, p: Vec3) -> Vec3 {
        delegate!(self, g => g.normal(p))
    }
    fn segment_exit(&self, a: Vec3, b: Vec3) -> Option<SegmentExit> {
        delegate!(self, g => g.segment_exit(a, b))
    }
    fn release_point(&self) -> Vec3 {
        delegate!(self, g => g.release_point())
    }
    fn bounds(&self) -> (Vec3, Vec3) {
        delegate!(self, g => g.bounds())
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

/// Bounds on JS-supplied sizes and fields, so bad input errors instead of aborting the module.
const MAX_BINS: usize = 4096;
const MAX_DROP_MV: f64 = 1000.0;

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
        let mut s = ParticlesSim::new("sharp", false, 1e-3, 3).unwrap();
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
        let mut s = ParticlesSim::new("funnel", false, 1e-4, 1).unwrap();
        s.release(10, 10).unwrap();
        assert!(!s.has_reference());
        assert!(s.survival(true).is_empty());
        assert!(s.mfpt(true)[0].is_nan());
        assert!(parse_geometry("cube").is_err());
        assert!(checked_params(0.0).is_err());
        assert!(FieldKind::parse("sideways").is_err());
        assert_eq!(s.histogram(false, usize::MAX, 1.0).len(), MAX_BINS);
    }

    #[test]
    fn outward_field_empties_the_spine_faster() {
        for kind in ["uniform", "neck"] {
            let run = |dv: f64| {
                let mut s = ParticlesSim::new("sharp", false, 1e-3, 5).unwrap();
                s.set_field(kind, dv).unwrap();
                s.release(400, 0).unwrap();
                s.step(3000);
                s.n_alive()
            };
            assert!(run(40.0) < run(-40.0), "{kind}");
        }
    }

    #[test]
    fn reference_keeps_running_after_the_main_population_has_escaped() {
        let mut s = ParticlesSim::new("sharp", false, 2e-3, 7).unwrap();
        s.set_field("uniform", 200.0).unwrap();
        s.release(50, 50).unwrap();
        while s.step(1000) > 0 {}
        assert_eq!((s.n_alive(), s.reference_alive()), (0, 0));
        assert_eq!(s.mfpt(true)[2], 50.0);
        assert!(
            s.time_ms() > s.sim.time_ms(),
            "the zero-field ions take longer"
        );
        assert!(s.mfpt(true)[0] > s.mfpt(false)[0]);
        // Turning the field off drops the reference.
        s.set_field("off", 0.0).unwrap();
        assert!(!s.has_reference());
    }

    #[test]
    fn dendrite_mode_shows_escaped_ions_until_they_leave_its_ends() {
        let mut s = ParticlesSim::new("funnel", true, 2e-3, 9).unwrap();
        assert!(s.dendrite_radius() > s.neck_radius() && s.dendrite_half_length() > 1.0);
        s.set_field("uniform", 40.0).unwrap();
        s.release(300, 100).unwrap();
        let mut seen_in_dendrite = 0;
        while s.n_alive() > 0 || s.reference_alive() > 0 {
            s.step(500);
            let (n, d) = (s.n_alive(), s.n_in_dendrite());
            assert_eq!(s.positions().len(), 3 * (n + d));
            let regions = s.regions();
            assert_eq!(regions.len(), n + d);
            let pos = s.positions();
            for (i, &r) in regions.iter().enumerate().skip(n) {
                assert_eq!(r == 2.0, pos[3 * i + 2] < 0.0, "escaped ion {i}");
            }
            assert!(regions[..n].iter().all(|&v| v < 2.0));
            assert_eq!(s.mfpt(false)[2] as usize, 300 - n);
            seen_in_dendrite = seen_in_dendrite.max(d);
        }
        assert!(seen_in_dendrite > 50, "{seen_in_dendrite}");
        // The reference is timed against an absorbing base, so none of its ions linger.
        assert_eq!(s.reference.as_ref().unwrap().n_escaped_moving(), 0);
        let t = s.passage_time_ms();
        // The escaped ions keep moving after the spine has emptied, then the run ends.
        while s.step(1000) > 0 {}
        assert_eq!((s.n_in_dendrite(), s.passage_time_ms()), (0, t));
        assert!(s.time_ms() >= t);
        assert_eq!(
            ParticlesSim::new("sharp", false, 1e-3, 1)
                .unwrap()
                .dendrite_radius(),
            0.0
        );
    }
}
