//! Population of Brownian ions and its first-passage statistics.

use ed_core::constants::{psi_mv, T_DEFAULT};
use ed_core::ions::IonSpecies;
use ed_core::Simulation;

use crate::field::{ElectricField, NoField};
use crate::geometry::{Geometry, Vec3};
use crate::rng::Xoshiro256pp;

/// Ca²⁺ as used in Publication 3: z = 2, D = 0.4 µm²/ms (4·10⁻¹⁰ m²/s).
pub const CA_SPINE: IonSpecies = IonSpecies {
    name: "Ca2+",
    z: 2,
    d: 0.4,
};

/// Default time step, ms. Boundary effects converge for Δt ≲ 1e-4 ms.
pub const DT_DEFAULT: f64 = 1e-4;

/// Largest population [`ParticleSim::release`] accepts.
pub const MAX_PARTICLES: usize = 1_000_000;

/// Particles processed per batch in the stepper (noise and field are evaluated per batch).
const CHUNK: usize = 128;

/// Maximum number of stored survival-curve samples; older samples are thinned 2:1.
const SURVIVAL_CAPACITY: usize = 2048;

/// Physical and numerical parameters of the moving species.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleParams {
    /// Valence.
    pub z: i32,
    /// Diffusion coefficient, µm²/ms.
    pub d: f64,
    /// Time step, ms.
    pub dt: f64,
    /// Temperature, K.
    pub temperature: f64,
}

impl Default for ParticleParams {
    fn default() -> Self {
        Self {
            z: CA_SPINE.z,
            d: CA_SPINE.d,
            dt: DT_DEFAULT,
            temperature: T_DEFAULT,
        }
    }
}

impl ParticleParams {
    /// Standard deviation of the random displacement per axis, √(2DΔt), µm.
    pub fn sigma(&self) -> f64 {
        (2.0 * self.d * self.dt).sqrt()
    }

    /// Drift per unit field, z D Δt / ψ, in µm per (mV/µm).
    pub fn mobility_dt(&self) -> f64 {
        self.z as f64 * self.d * self.dt / psi_mv(self.temperature)
    }
}

/// Result of moving one particle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepOutcome {
    /// The particle is inside the domain at this position.
    Inside(Vec3),
    /// The particle crossed an absorbing boundary.
    Absorbed,
}

const MAX_ROOT_ITERS: usize = 60;
const MAX_REFLECTIONS: usize = 8;
/// A point whose |SDF| is below this (µm) is taken to lie on the boundary.
const BOUNDARY_TOL: f64 = 1e-9;
/// Bracket width (µm) at which the crossing search stops if the SDF test has not.
const BRACKET_TOL: f64 = 1e-7;

/// Moves a particle from `from` (inside) towards the proposed position `to`, resolving
/// walls: specular reflection off reflecting walls, absorption at absorbing ones.
///
/// The crossing point is found on the segment with a bracketed secant search on the SDF
/// (Illinois regula falsi), so any [`Geometry`] works. If repeated reflections cannot bring
/// the particle back inside (e.g. in a corner), it stays at the last inside point, so it
/// never leaves the domain.
#[inline]
pub fn resolve_step<G: Geometry + ?Sized>(geom: &G, from: Vec3, to: Vec3) -> StepOutcome {
    if geom.contains(to) {
        StepOutcome::Inside(to)
    } else {
        resolve_collision(geom, from, to)
    }
}

#[inline(never)]
fn resolve_collision<G: Geometry + ?Sized>(geom: &G, from: Vec3, to: Vec3) -> StepOutcome {
    let mut start = from;
    let mut end = to;
    for _ in 0..MAX_REFLECTIONS {
        let (inside, b, n) = match geom.segment_exit(start, end) {
            Some(e) => (e.inside, e.boundary, e.normal),
            None => {
                let (inside, b) = crossing(geom, start, end);
                (inside, b, geom.normal(b))
            }
        };
        if geom.absorbs(b) {
            return StepOutcome::Absorbed;
        }
        let excess = (end[0] - b[0]) * n[0] + (end[1] - b[1]) * n[1] + (end[2] - b[2]) * n[2];
        if excess <= 0.0 {
            // `end` is outside but behind the tangent plane: the wall curves into the domain
            // (the funnel fillet). Mirroring in the tangent plane would not bring it back.
            return mirror_in_nearest_wall(geom, end, inside);
        }
        let reflected = [
            end[0] - 2.0 * excess * n[0],
            end[1] - 2.0 * excess * n[1],
            end[2] - 2.0 * excess * n[2],
        ];
        if geom.contains(reflected) {
            return StepOutcome::Inside(reflected);
        }
        start = inside;
        end = reflected;
    }
    mirror_in_nearest_wall(geom, end, start)
}

/// Mirrors the outside point `p` through the nearest boundary point, p − 2·sdf(p)·n(p);
/// if that is not inside either, falls back to `fallback` (an inside point).
fn mirror_in_nearest_wall<G: Geometry + ?Sized>(geom: &G, p: Vec3, fallback: Vec3) -> StepOutcome {
    let d = geom.sdf(p);
    let n = geom.normal(p);
    let m = [
        p[0] - 2.0 * d * n[0],
        p[1] - 2.0 * d * n[1],
        p[2] - 2.0 * d * n[2],
    ];
    if d.is_finite() && geom.contains(m) {
        StepOutcome::Inside(m)
    } else {
        StepOutcome::Inside(fallback)
    }
}

/// Finds where the segment `a` (inside) → `b` (outside) crosses the boundary. Returns the
/// last inside point found on the segment and the boundary point.
///
/// Assumes the step is short compared with the wall's radius of curvature: on a long
/// segment that leaves and re-enters, it may find a later crossing instead of the first.
/// Secant steps on the SDF converge in two or three iterations for surfaces that are smooth
/// on the scale of a step; the Illinois rule and a bisection step every third iteration keep
/// the bracket shrinking at creases.
fn crossing<G: Geometry + ?Sized>(geom: &G, a: Vec3, b: Vec3) -> (Vec3, Vec3) {
    let (mut lo, mut hi) = (a, b);
    let mut f_lo = geom.sdf(lo).min(0.0);
    let mut f_hi = geom.sdf(hi).max(0.0);
    let mut side = 0i8;
    for it in 0..MAX_ROOT_ITERS {
        let denom = f_lo - f_hi;
        let t = if it % 3 == 2 || denom >= 0.0 || !denom.is_finite() {
            0.5
        } else {
            // In [0, 1] because f_lo ≤ 0 ≤ f_hi; keep strictly inside so the bracket moves.
            (f_lo / denom).clamp(1e-6, 1.0 - 1e-6)
        };
        let m = lerp(lo, hi, t);
        let f = geom.sdf(m);
        let inside = geom.contains(m);
        if inside {
            lo = m;
        } else {
            hi = m;
        }
        let d = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        if f.abs() < BOUNDARY_TOL
            || d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < BRACKET_TOL * BRACKET_TOL
        {
            return (lo, m);
        }
        if inside {
            f_lo = f.min(0.0);
            if side == -1 {
                f_hi *= 0.5;
            }
            side = -1;
        } else {
            f_hi = f.max(0.0);
            if side == 1 {
                f_lo *= 0.5;
            }
            side = 1;
        }
    }
    (lo, lerp(lo, hi, 0.5))
}

#[inline]
fn lerp(a: Vec3, b: Vec3, t: f64) -> Vec3 {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

/// Mean first passage time with its standard error, ms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MfptEstimate {
    pub mean: f64,
    pub std_err: f64,
    /// Number of particles the estimate is based on (absorbed so far).
    pub n: usize,
}

/// Brownian electrodiffusion of a population of identical ions.
///
/// Each step moves every live particle by r ← r + X + (z D Δt / ψ) E(r),
/// X ~ N(0, 2DΔt) per axis, which is exact in a constant field at any Δt.
/// Positions are stored as struct-of-arrays; absorbed particles are removed at the end of
/// the step, so live particles stay contiguous.
pub struct ParticleSim<G: Geometry> {
    geom: G,
    field: Box<dyn ElectricField>,
    params: ParticleParams,
    rng: Xoshiro256pp,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    n_released: usize,
    time: f64,
    steps: u64,
    exit_times: Vec<f64>,
    survival: Vec<(f64, f64)>,
    survival_every: u64,
    xyz: Vec<f32>,
    /// Escaped particles still moving (domains with an [`Geometry::escaped`] region).
    ex: Vec<f64>,
    ey: Vec<f64>,
    ez: Vec<f64>,
    /// Particles that escaped during the current step, moved to `ex/ey/ez` at its end.
    pending: Vec<Vec3>,
    /// Time of the last step that began with particles inside, ms.
    passage_time: f64,
}

impl<G: Geometry> ParticleSim<G> {
    /// Creates an empty simulation (no particles until [`ParticleSim::release`]).
    pub fn new(geom: G, params: ParticleParams, seed: u64) -> Self {
        assert!(
            params.d > 0.0 && params.dt > 0.0,
            "D and Δt must be positive"
        );
        Self {
            geom,
            field: Box::new(NoField),
            params,
            rng: Xoshiro256pp::new(seed),
            x: Vec::new(),
            y: Vec::new(),
            z: Vec::new(),
            n_released: 0,
            time: 0.0,
            steps: 0,
            exit_times: Vec::new(),
            survival: Vec::new(),
            survival_every: 1,
            xyz: Vec::new(),
            ex: Vec::new(),
            ey: Vec::new(),
            ez: Vec::new(),
            pending: Vec::new(),
            passage_time: 0.0,
        }
    }

    /// Builder-style field setter.
    pub fn with_field(mut self, field: impl ElectricField + 'static) -> Self {
        self.field = Box::new(field);
        self
    }

    /// Replaces the field; particles keep their positions.
    pub fn set_field(&mut self, field: impl ElectricField + 'static) {
        self.field = Box::new(field);
    }

    /// Replaces the parameters (e.g. Δt); particles keep their positions.
    pub fn set_params(&mut self, params: ParticleParams) {
        assert!(
            params.d > 0.0 && params.dt > 0.0,
            "D and Δt must be positive"
        );
        self.params = params;
    }

    pub fn params(&self) -> ParticleParams {
        self.params
    }

    pub fn geometry(&self) -> &G {
        &self.geom
    }

    /// Discards the current population and statistics, then releases `n` particles at
    /// the geometry's release point at t = 0.
    pub fn release(&mut self, n: usize) {
        assert!(n <= MAX_PARTICLES, "at most {MAX_PARTICLES} particles");
        let p = self.geom.release_point();
        for v in [
            &mut self.x,
            &mut self.y,
            &mut self.z,
            &mut self.ex,
            &mut self.ey,
            &mut self.ez,
        ] {
            v.clear();
        }
        self.x.resize(n, p[0]);
        self.y.resize(n, p[1]);
        self.z.resize(n, p[2]);
        self.n_released = n;
        self.time = 0.0;
        self.passage_time = 0.0;
        self.steps = 0;
        self.exit_times.clear();
        self.exit_times.reserve(n);
        self.survival.clear();
        self.survival.push((0.0, if n > 0 { 1.0 } else { 0.0 }));
        self.survival_every = 1;
    }

    /// Advances all live particles by one time step.
    pub fn step_once(&mut self) {
        let mover = Mover {
            sigma: self.params.sigma(),
            mob: self.params.mobility_dt(),
            field: self.field.as_ref(),
        };
        let t_end = self.time + self.params.dt;
        let in_spine = !self.x.is_empty();
        // Escaped particles move first, so the ones that escape during this step are not
        // moved twice.
        let pop = [&mut self.ex, &mut self.ey, &mut self.ez];
        let gone = mover.advance(&self.geom, &mut self.rng, pop, |outcome| match outcome {
            StepOutcome::Inside(r) => Some(r),
            StepOutcome::Absorbed => None,
        });
        if gone {
            compact([&mut self.ex, &mut self.ey, &mut self.ez]);
        }
        let (geom, exit_times, pending) = (&self.geom, &mut self.exit_times, &mut self.pending);
        let pop = [&mut self.x, &mut self.y, &mut self.z];
        let left = mover.advance(geom, &mut self.rng, pop, |outcome| match outcome {
            StepOutcome::Inside(r) if !geom.escaped(r) => Some(r),
            outcome => {
                exit_times.push(t_end);
                if let StepOutcome::Inside(r) = outcome {
                    pending.push(r);
                }
                None
            }
        });
        if left {
            compact([&mut self.x, &mut self.y, &mut self.z]);
        }
        for p in self.pending.drain(..) {
            self.ex.push(p[0]);
            self.ey.push(p[1]);
            self.ez.push(p[2]);
        }
        self.time = t_end;
        self.steps += 1;
        if in_spine {
            self.passage_time = t_end;
            // Always sample the moment the last particle escapes, so the curve ends at 0.
            if self.steps % self.survival_every == 0 || self.x.is_empty() {
                self.survival.push((self.time, self.survival_fraction()));
                if self.survival.len() >= SURVIVAL_CAPACITY {
                    // Keep t = 0 and every other sample; sample half as often from now on.
                    let mut k = 0;
                    self.survival.retain(|_| {
                        k += 1;
                        k % 2 == 1
                    });
                    self.survival_every *= 2;
                }
            }
        }
    }

    /// Live particles that have not escaped yet.
    pub fn n_alive(&self) -> usize {
        self.x.len()
    }

    /// Escaped particles that are still moving (always 0 when escaping means absorption).
    pub fn n_escaped_moving(&self) -> usize {
        self.ex.len()
    }

    /// Simulated time until the last particle escaped, or the current time while some
    /// are still inside, ms. Survival samples stop here.
    pub fn passage_time(&self) -> f64 {
        self.passage_time
    }

    /// Particles released by the last [`ParticleSim::release`].
    pub fn n_released(&self) -> usize {
        self.n_released
    }

    /// Fraction of released particles still inside, S(t).
    pub fn survival_fraction(&self) -> f64 {
        if self.n_released == 0 {
            0.0
        } else {
            self.n_alive() as f64 / self.n_released as f64
        }
    }

    /// Exit times (ms) of absorbed particles, in absorption order.
    pub fn exit_times(&self) -> &[f64] {
        &self.exit_times
    }

    /// Survival curve samples `(t, S(t))`, starting at (0, 1). At most ~2048 samples are
    /// kept; the sampling interval doubles as the run gets longer.
    pub fn survival_curve(&self) -> &[(f64, f64)] {
        &self.survival
    }

    /// Sample mean and standard error of the exit times observed so far. This is the MFPT
    /// once every particle has been absorbed; before that it is biased towards early exits.
    pub fn mfpt(&self) -> Option<MfptEstimate> {
        let n = self.exit_times.len();
        if n == 0 {
            return None;
        }
        let nf = n as f64;
        let mean = self.exit_times.iter().sum::<f64>() / nf;
        let std_err = if n > 1 {
            let var = self
                .exit_times
                .iter()
                .map(|t| (t - mean).powi(2))
                .sum::<f64>()
                / (nf - 1.0);
            (var / nf).sqrt()
        } else {
            f64::NAN
        };
        Some(MfptEstimate { mean, std_err, n })
    }

    /// Running MFPT estimate that accounts for particles still inside, assuming an
    /// exponential exit-time tail (a good model for narrow escape): total time spent
    /// inside by all particles divided by the number absorbed. Equals [`ParticleSim::mfpt`]
    /// once every particle has been absorbed.
    pub fn mfpt_censored(&self) -> Option<MfptEstimate> {
        let n = self.exit_times.len();
        if n == 0 {
            return None;
        }
        let total = self.exit_times.iter().sum::<f64>() + self.n_alive() as f64 * self.passage_time;
        let mean = total / n as f64;
        Some(MfptEstimate {
            mean,
            std_err: mean / (n as f64).sqrt(),
            n,
        })
    }

    /// Histogram of exit times over [0, t_max) with `bins` equal bins; exits at or after
    /// `t_max` go in the last bin.
    pub fn exit_time_histogram(&self, bins: usize, t_max: f64) -> Vec<u32> {
        let mut h = vec![0u32; bins];
        if bins == 0 || !t_max.is_finite() || t_max <= 0.0 {
            return h;
        }
        let scale = bins as f64 / t_max;
        for &t in &self.exit_times {
            let k = ((t * scale) as usize).min(bins - 1);
            h[k] += 1;
        }
        h
    }

    /// Interleaved xyz positions of live particles as `f32` (µm), for rendering.
    /// The buffer is reused between calls.
    pub fn positions_xyz(&mut self) -> &[f32] {
        let n = self.x.len();
        self.xyz.resize(3 * n, 0.0);
        for (i, out) in self.xyz.chunks_exact_mut(3).enumerate() {
            out[0] = self.x[i] as f32;
            out[1] = self.y[i] as f32;
            out[2] = self.z[i] as f32;
        }
        &self.xyz
    }

    /// Calls `f` with the position (f64, µm) of every live particle that has not escaped.
    pub fn for_each_position(&self, mut f: impl FnMut(Vec3)) {
        for i in 0..self.x.len() {
            f([self.x[i], self.y[i], self.z[i]]);
        }
    }

    /// Calls `f` with the position (f64, µm) of every escaped particle still moving.
    pub fn for_each_escaped_position(&self, mut f: impl FnMut(Vec3)) {
        for i in 0..self.ex.len() {
            f([self.ex[i], self.ey[i], self.ez[i]]);
        }
    }
}

/// Moves one population (struct-of-arrays) through a time step.
struct Mover<'a> {
    sigma: f64,
    mob: f64,
    field: &'a dyn ElectricField,
}

impl Mover<'_> {
    /// Moves every particle of `pop` and hands each outcome to `fate`, which returns the new
    /// position or `None` to remove the particle (marked NaN; see [`compact`]). Returns
    /// whether any particle was removed.
    fn advance<G: Geometry>(
        &self,
        geom: &G,
        rng: &mut Xoshiro256pp,
        [xs, ys, zs]: [&mut Vec<f64>; 3],
        mut fate: impl FnMut(StepOutcome) -> Option<Vec3>,
    ) -> bool {
        let uniform = self
            .field
            .uniform()
            .map(|e| [self.mob * e[0], self.mob * e[1], self.mob * e[2]]);
        let mut noise = [0.0; 3 * CHUNK];
        let mut drift = [[0.0; 3]; CHUNK];
        let n = xs.len();
        let mut removed = false;
        for start in (0..n).step_by(CHUNK) {
            let end = (start + CHUNK).min(n);
            let len = end - start;
            rng.fill_normal(&mut noise[..3 * len]);
            match uniform {
                Some(d) => drift[..len].fill(d),
                None => {
                    // One dynamic call per chunk; the loop inside is statically dispatched.
                    self.field.field_batch(
                        &xs[start..end],
                        &ys[start..end],
                        &zs[start..end],
                        &mut drift[..len],
                    );
                    for d in &mut drift[..len] {
                        *d = [self.mob * d[0], self.mob * d[1], self.mob * d[2]];
                    }
                }
            }
            debug_assert!(
                drift[..len].iter().flatten().all(|v| v.is_finite()),
                "electric field returned a non-finite value"
            );
            let sigma = self.sigma;
            for j in 0..len {
                let i = start + j;
                let p = [xs[i], ys[i], zs[i]];
                let q = [
                    p[0] + drift[j][0] + sigma * noise[3 * j],
                    p[1] + drift[j][1] + sigma * noise[3 * j + 1],
                    p[2] + drift[j][2] + sigma * noise[3 * j + 2],
                ];
                let r = fate(resolve_step(geom, p, q)).unwrap_or_else(|| {
                    removed = true;
                    [f64::NAN; 3]
                });
                xs[i] = r[0];
                ys[i] = r[1];
                zs[i] = r[2];
            }
        }
        removed
    }
}

/// Removes particles marked as removed (NaN x), keeping the others in order.
fn compact([xs, ys, zs]: [&mut Vec<f64>; 3]) {
    let mut w = 0;
    for r in 0..xs.len() {
        if !xs[r].is_nan() {
            xs[w] = xs[r];
            ys[w] = ys[r];
            zs[w] = zs[r];
            w += 1;
        }
    }
    xs.truncate(w);
    ys.truncate(w);
    zs.truncate(w);
}

impl<G: Geometry> Simulation for ParticleSim<G> {
    fn step(&mut self, n: usize) {
        for _ in 0..n {
            if self.x.is_empty() && self.ex.is_empty() {
                break;
            }
            self.step_once();
        }
    }

    fn time_ms(&self) -> f64 {
        self.time
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{NeckField, UniformField};
    use crate::geometry::{FreeSpace, SpineGeometry};

    #[test]
    fn free_motion_matches_drift_and_variance() {
        let params = ParticleParams {
            dt: 0.01,
            ..Default::default()
        };
        let e = 10.0; // mV/µm along x
        let mut sim =
            ParticleSim::new(FreeSpace, params, 11).with_field(UniformField([e, 0.0, 0.0]));
        let n = 200_000;
        let steps = 5;
        sim.release(n);
        sim.step(steps);
        let t = steps as f64 * params.dt;
        let mut sum = [0.0; 3];
        let mut sum2 = [0.0; 3];
        sim.for_each_position(|p| {
            for k in 0..3 {
                sum[k] += p[k];
                sum2[k] += p[k] * p[k];
            }
        });
        let nf = n as f64;
        let drift = params.z as f64 * params.d * e * t / psi_mv(params.temperature);
        let var_expect = 2.0 * params.d * t;
        let se_mean = (var_expect / nf).sqrt();
        for k in 0..3 {
            let mean = sum[k] / nf;
            let var = sum2[k] / nf - mean * mean;
            let expect = if k == 0 { drift } else { 0.0 };
            assert!(
                (mean - expect).abs() < 4.0 * se_mean,
                "axis {k}: mean {mean}, expected {expect} ± {se_mean}"
            );
            assert!(
                (var / var_expect - 1.0).abs() < 0.015,
                "axis {k}: var {var}, expected {var_expect}"
            );
        }
        // The drift itself is resolved well above the noise.
        assert!(drift > 10.0 * se_mean);
    }

    #[test]
    fn particles_never_leave_the_spine() {
        for geom in [SpineGeometry::sharp(), SpineGeometry::funneled()] {
            // Coarse Δt and a strong field stress the reflection code.
            let params = ParticleParams {
                dt: 2e-3,
                ..Default::default()
            };
            let mut sim =
                ParticleSim::new(geom, params, 5).with_field(UniformField([3.0, -2.0, 4.0]));
            sim.release(5000);
            for _ in 0..400 {
                sim.step_once();
                sim.for_each_position(|p| assert!(geom.contains(p), "{p:?} escaped {geom:?}"));
            }
            assert_eq!(sim.n_alive() + sim.exit_times().len(), 5000);
        }
    }

    #[test]
    fn reflection_is_specular_off_a_flat_wall() {
        let g = SpineGeometry::sharp();
        let from = [0.05, 0.0, 0.5];
        let to = [0.13, 0.0, 0.52];
        match resolve_step(&g, from, to) {
            StepOutcome::Inside(p) => {
                assert!((p[0] - (2.0 * NECK - 0.13)).abs() < 1e-9, "{p:?}");
                assert!((p[2] - 0.52).abs() < 1e-9);
            }
            StepOutcome::Absorbed => panic!("absorbed"),
        }
        assert_eq!(
            resolve_step(&g, [0.0, 0.0, 0.01], [0.0, 0.0, -0.01]),
            StepOutcome::Absorbed
        );
    }
    const NECK: f64 = crate::geometry::NECK_RADIUS;

    #[test]
    fn grazing_hit_on_the_funnel_reflects_instead_of_sticking() {
        // Found in review: reflecting in the tangent plane leaves this step inside the
        // torus, and the old loop then ping-ponged until it gave up on the wall.
        let g = SpineGeometry::funneled();
        let a = [-0.148_135_584, 0.043_695_023, 1.045_471_113];
        let b = [-0.151_889_974, 0.020_437_573, 1.042_373_450];
        assert!(g.contains(a) && !g.contains(b));
        match resolve_step(&g, a, b) {
            StepOutcome::Inside(p) => {
                assert!(g.contains(p));
                // The step overshot the wall by ~1.6e-4 µm, so it ends about that far inside.
                assert!(g.sdf(p) < -1e-4, "stuck on the wall: sdf {}", g.sdf(p));
            }
            StepOutcome::Absorbed => panic!("absorbed"),
        }
    }

    #[test]
    fn statistics_bookkeeping() {
        let params = ParticleParams {
            dt: 1e-3,
            ..Default::default()
        };
        let mut sim = ParticleSim::new(SpineGeometry::sharp(), params, 1);
        sim.release(100);
        assert!(sim.mfpt().is_none());
        for _ in 0..5000 {
            sim.step_once();
        }
        assert_eq!(sim.n_alive() + sim.exit_times().len(), 100);
        let h = sim.exit_time_histogram(10, sim.time_ms());
        assert_eq!(h.iter().sum::<u32>() as usize, sim.exit_times().len());
        let s = sim.survival_curve();
        assert_eq!(s[0], (0.0, 1.0));
        assert!(s.len() < SURVIVAL_CAPACITY);
        assert!(s.windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 >= w[1].1));
        assert_eq!(s.last().unwrap().1, sim.survival_fraction());
        assert_eq!(sim.positions_xyz().len(), 3 * sim.n_alive());
    }

    /// Free space where everything below z = 0 counts as escaped and the plane z = −1
    /// absorbs.
    #[derive(Clone, Copy)]
    struct Ledge;
    impl Geometry for Ledge {
        fn sdf(&self, p: Vec3) -> f64 {
            -1.0 - p[2]
        }
        fn absorbs(&self, _b: Vec3) -> bool {
            true
        }
        fn escaped(&self, p: Vec3) -> bool {
            p[2] < 0.0
        }
        fn release_point(&self) -> Vec3 {
            [0.0, 0.0, 0.05]
        }
        fn bounds(&self) -> (Vec3, Vec3) {
            (
                [f64::NEG_INFINITY, f64::NEG_INFINITY, -1.0],
                [f64::INFINITY; 3],
            )
        }
    }

    #[test]
    fn escaped_particles_are_timed_once_and_keep_moving() {
        let params = ParticleParams {
            dt: 1e-3,
            ..Default::default()
        };
        let mut sim = ParticleSim::new(Ledge, params, 2).with_field(UniformField([0.0, 0.0, -2.0]));
        let n = 1000;
        sim.release(n);
        let mut max_moving = 0;
        while sim.n_alive() > 0 {
            sim.step(20);
            sim.for_each_position(|p| assert!(p[2] >= 0.0));
            sim.for_each_escaped_position(|p| assert!(p[2] > -1.0));
            assert_eq!(sim.n_alive() + sim.exit_times().len(), n);
            assert!(sim.n_escaped_moving() <= sim.exit_times().len());
            max_moving = max_moving.max(sim.n_escaped_moving());
        }
        assert!(
            max_moving > n / 2,
            "only {max_moving} escaped particles moving"
        );
        let t = sim.passage_time();
        assert!(t <= sim.time_ms() && t > sim.time_ms() - 20.0 * params.dt);
        assert_eq!(sim.survival_curve().last().unwrap(), &(t, 0.0));
        // The escaped particles keep moving (and are absorbed at z = −1) without adding
        // survival samples or exits.
        let samples = sim.survival_curve().len();
        while sim.n_escaped_moving() > 0 {
            sim.step(100);
        }
        assert!(sim.time_ms() > t);
        assert_eq!(
            (sim.passage_time(), sim.survival_curve().len()),
            (t, samples)
        );
        assert_eq!(sim.exit_times().len(), n);
    }

    #[test]
    fn last_exit_is_always_sampled() {
        let mut sim = ParticleSim::new(Ledge, ParticleParams::default(), 4)
            .with_field(UniformField([0.0, 0.0, -50.0]));
        sim.release(3);
        while sim.n_alive() > 0 {
            sim.step_once();
        }
        assert_eq!(
            sim.survival_curve().last().unwrap(),
            &(sim.passage_time(), 0.0)
        );
    }

    // ---- First-passage tests. These run long simulations; CI runs tests in release mode. ----

    /// Work is always split into this many independently seeded threads, so results do not
    /// depend on the machine's core count.
    const WORK_UNITS: usize = 8;

    /// Runs `n` particles from the head centre until all are absorbed, split across threads.
    fn exit_times(
        geom: SpineGeometry,
        params: ParticleParams,
        field: Option<NeckField>,
        n: usize,
        seed: u64,
    ) -> Vec<f64> {
        let threads = WORK_UNITS;
        let chunk = n.div_ceil(threads);
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|k| {
                    s.spawn(move || {
                        let count = chunk.min(n.saturating_sub(k * chunk));
                        let mut sim =
                            ParticleSim::new(geom, params, seed.wrapping_add(k as u64 * 0x1000));
                        if let Some(f) = field {
                            sim.set_field(f);
                        }
                        sim.release(count);
                        while sim.n_alive() > 0 {
                            sim.step(1000);
                        }
                        sim.exit_times().to_vec()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        })
    }

    fn mean(v: &[f64]) -> f64 {
        v.iter().sum::<f64>() / v.len() as f64
    }

    /// Holcman & Schuss narrow-escape approximation for a spine:
    /// τ ≈ V/(4aD) + L²/(2D) + V L/(π a² D), with V the head volume.
    fn narrow_escape_mfpt(g: &SpineGeometry, d: f64) -> f64 {
        let v = 4.0 / 3.0 * std::f64::consts::PI * g.head_radius().powi(3);
        let a = g.neck_radius();
        let l = g.neck_length();
        v / (4.0 * a * d) + l * l / (2.0 * d) + v * l / (std::f64::consts::PI * a * a * d)
    }

    #[test]
    fn mfpt_without_field_matches_narrow_escape() {
        let g = SpineGeometry::sharp();
        let params = ParticleParams::default();
        let t = exit_times(g, params, None, 800, 2024);
        let m = mean(&t);
        let expect = narrow_escape_mfpt(&g, params.d);
        eprintln!("MFPT {m:.2} ms, narrow escape {expect:.2} ms");
        assert!((m / expect - 1.0).abs() < 0.10, "MFPT {m} vs {expect}");
    }

    #[test]
    fn field_out_of_the_spine_lowers_mfpt_and_reverse_raises_it() {
        let g = SpineGeometry::sharp();
        let params = ParticleParams {
            dt: 2e-4,
            ..Default::default()
        };
        // True ratios are about 0.73 (out) and 1.4 (in); with 600 particles the standard
        // error of each ratio is about 6%, so the thresholds sit several SE away.
        let n = 600;
        let none = mean(&exit_times(g, params, None, n, 9));
        let out = mean(&exit_times(
            g,
            params,
            Some(NeckField::for_spine(&g, 10.0)),
            n,
            9,
        ));
        let inward = mean(&exit_times(
            g,
            params,
            Some(NeckField::for_spine(&g, -10.0)),
            n,
            9,
        ));
        eprintln!("MFPT out {out:.2}, none {none:.2}, in {inward:.2} ms");
        assert!(out < 0.9 * none, "out {out} vs none {none}");
        assert!(inward > 1.15 * none, "in {inward} vs none {none}");
    }

    #[test]
    fn mfpt_converges_in_dt() {
        // Synchronously coupled paths: the coarse path (Δt) uses the sum of each pair of
        // fine increments (Δt/2), so both follow the same Brownian motion. The pair stays
        // close until one of them enters the neck before the other, which still reduces
        // the variance of the difference.
        let g = SpineGeometry::sharp();
        let coarse = ParticleParams {
            dt: 1e-4,
            ..Default::default()
        };
        let fine = ParticleParams { dt: 5e-5, ..coarse };
        let n: usize = 2400;
        let threads = WORK_UNITS;
        let per = n.div_ceil(threads);
        let pairs: Vec<(f64, f64)> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|k| {
                    s.spawn(move || {
                        let mut rng = Xoshiro256pp::new(77 + k as u64);
                        let sf = fine.sigma();
                        let sc = coarse.sigma();
                        let mut out = Vec::new();
                        for _ in 0..per {
                            let (mut pf, mut pc) = (Some(g.head_centre()), Some(g.head_centre()));
                            let (mut tf, mut tc) = (0.0, 0.0);
                            let mut t = 0.0;
                            while pf.is_some() || pc.is_some() {
                                let xi1 = [rng.normal(), rng.normal(), rng.normal()];
                                let xi2 = [rng.normal(), rng.normal(), rng.normal()];
                                for xi in [xi1, xi2] {
                                    t += fine.dt;
                                    if let Some(p) = pf {
                                        let q = [
                                            p[0] + sf * xi[0],
                                            p[1] + sf * xi[1],
                                            p[2] + sf * xi[2],
                                        ];
                                        pf = match resolve_step(&g, p, q) {
                                            StepOutcome::Inside(r) => Some(r),
                                            StepOutcome::Absorbed => {
                                                tf = t;
                                                None
                                            }
                                        };
                                    }
                                }
                                if let Some(p) = pc {
                                    let c = sc * std::f64::consts::FRAC_1_SQRT_2;
                                    let q = [
                                        p[0] + c * (xi1[0] + xi2[0]),
                                        p[1] + c * (xi1[1] + xi2[1]),
                                        p[2] + c * (xi1[2] + xi2[2]),
                                    ];
                                    pc = match resolve_step(&g, p, q) {
                                        StepOutcome::Inside(r) => Some(r),
                                        StepOutcome::Absorbed => {
                                            tc = t;
                                            None
                                        }
                                    };
                                }
                            }
                            out.push((tc, tf));
                        }
                        out
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        let mc = pairs.iter().map(|p| p.0).sum::<f64>() / pairs.len() as f64;
        let mf = pairs.iter().map(|p| p.1).sum::<f64>() / pairs.len() as f64;
        let diffs: Vec<f64> = pairs.iter().map(|p| p.0 - p.1).collect();
        let md = mean(&diffs);
        let sd =
            (diffs.iter().map(|d| (d - md).powi(2)).sum::<f64>() / (diffs.len() - 1) as f64).sqrt();
        let se_rel = sd / (diffs.len() as f64).sqrt() / mf;
        eprintln!(
            "MFPT Δt=1e-4: {mc:.2} ms, Δt=5e-5: {mf:.2} ms, rel. diff {:.3} ± {se_rel:.3}",
            mc / mf - 1.0
        );
        assert!((mc / mf - 1.0).abs() < 0.05, "coarse {mc} vs fine {mf}");
        // The coupling must actually resolve a 5% difference.
        assert!(se_rel < 0.02, "relative standard error {se_rel}");
    }
}
