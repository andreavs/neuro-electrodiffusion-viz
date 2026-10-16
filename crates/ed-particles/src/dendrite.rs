//! A spine on its parent dendrite: ions that leave through the neck keep diffusing.
//!
//! The dendrite is a cylinder along x whose top touches the plane z = 0 under the neck, so
//! the neck base of [`SpineGeometry`] opens into it. Crossing z = 0 is the first passage
//! (the same event the absorbing base records); the far ends of the dendrite absorb.

use crate::geometry::{box_grad, box_sdf, Geometry, SegmentExit, SpineGeometry, Vec3};

/// Default dendrite radius and half-length, µm.
pub const DENDRITE_RADIUS: f64 = 0.45;
pub const DENDRITE_HALF_LENGTH: f64 = 3.0;

/// Height (µm) up to which the neck stub used in the SDF overlaps the spine's own neck, so
/// the spine's base plane is never an internal surface of the union.
const STUB_TOP: f64 = 0.05;
/// Boundary points this close to the dendrite's end planes (µm) count as absorbed.
const END_TOL: f64 = 1e-6;

/// Union of a [`SpineGeometry`] (neck extended down into the dendrite) and a dendrite shaft.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpineOnDendrite {
    spine: SpineGeometry,
    radius: f64,
    half_length: f64,
    /// Below this height the neck stub lies wholly inside the shaft.
    sliver_z: f64,
}

impl SpineOnDendrite {
    pub fn new(spine: SpineGeometry, radius: f64, half_length: f64) -> Self {
        let a = spine.neck_radius();
        assert!(radius > a, "the dendrite must be wider than the neck");
        assert!(
            half_length > a,
            "the dendrite must be longer than the neck is wide"
        );
        Self {
            spine,
            radius,
            half_length,
            sliver_z: (radius * radius - a * a).sqrt() - radius,
        }
    }

    /// `spine` on a dendrite of the default size.
    pub fn with_default_dendrite(spine: SpineGeometry) -> Self {
        Self::new(spine, DENDRITE_RADIUS, DENDRITE_HALF_LENGTH)
    }

    pub fn spine(&self) -> &SpineGeometry {
        &self.spine
    }
    pub fn radius(&self) -> f64 {
        self.radius
    }
    pub fn half_length(&self) -> f64 {
        self.half_length
    }

    /// SDF terms: spine, neck stub, shaft.
    #[inline]
    fn terms(&self, p: Vec3) -> [f64; 3] {
        let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
        let rho = (p[1] * p[1] + (p[2] + self.radius).powi(2)).sqrt();
        [
            self.spine.sdf(p),
            box_sdf(r, p[2], self.spine.neck_radius(), -self.radius, STUB_TOP),
            box_sdf(rho, p[0], self.radius, -self.half_length, self.half_length),
        ]
    }

    /// Closed-form exit through the shaft's wall or ends, for segments wholly below the
    /// neck stub.
    fn shaft_exit(&self, a: Vec3, b: Vec3) -> Option<SegmentExit> {
        let (rr, l) = (self.radius, self.half_length);
        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let len2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if len2 == 0.0 {
            return None;
        }
        // Candidate crossings (t, surface): 0 wall, 1 the +x end, 2 the −x end.
        let mut cand = [(f64::INFINITY, 0u8); 4];
        let mut nc = 0;
        let mut push = |t: f64, s: u8| {
            if t > 0.0 && t <= 1.0 {
                cand[nc] = (t, s);
                nc += 1;
            }
        };
        let az = a[2] + rr;
        let ar = d[1] * d[1] + d[2] * d[2];
        if ar > 0.0 {
            let br = a[1] * d[1] + az * d[2];
            let cr = a[1] * a[1] + az * az - rr * rr;
            let disc = br * br - ar * cr;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                push((-br - sq) / ar, 0);
                push((-br + sq) / ar, 0);
            }
        }
        if d[0] != 0.0 {
            push((l - a[0]) / d[0], 1);
            push((-l - a[0]) / d[0], 2);
        }
        let cand = &mut cand[..nc];
        cand.sort_unstable_by(|x, y| x.0.total_cmp(&y.0));
        let eps = 1e-9 / len2.sqrt();
        let at = |t: f64| [a[0] + t * d[0], a[1] + t * d[1], a[2] + t * d[2]];
        for &(t, surface) in cand.iter() {
            if self.contains(at((t + eps).min(1.0))) {
                continue;
            }
            let inside = at((t - eps).max(0.0));
            if !self.contains(inside) {
                return None;
            }
            let p = at(t);
            let (boundary, normal) = match surface {
                0 => (p, [0.0, p[1] / rr, (p[2] + rr) / rr]),
                1 => ([l, p[1], p[2]], [1.0, 0.0, 0.0]),
                _ => ([-l, p[1], p[2]], [-1.0, 0.0, 0.0]),
            };
            return Some(SegmentExit {
                inside,
                boundary,
                normal,
            });
        }
        None
    }
}

impl Geometry for SpineOnDendrite {
    fn sdf(&self, p: Vec3) -> f64 {
        let [s, n, d] = self.terms(p);
        s.min(n).min(d)
    }

    #[inline]
    fn contains(&self, p: Vec3) -> bool {
        let a = self.spine.neck_radius();
        let rr = self.radius;
        let z = p[2];
        let in_stub = (p[0] * p[0] + p[1] * p[1] < a * a) & (z <= 0.0) & (z > -rr);
        let zc = z + rr;
        let in_shaft = (p[1] * p[1] + zc * zc < rr * rr) & (p[0].abs() < self.half_length);
        self.spine.contains(p) | in_stub | in_shaft
    }

    fn absorbs(&self, b: Vec3) -> bool {
        b[0].abs() > self.half_length - END_TOL
    }

    #[inline]
    fn escaped(&self, p: Vec3) -> bool {
        p[2] < 0.0
    }

    /// Gradient of whichever SDF term is smallest.
    fn normal(&self, p: Vec3) -> Vec3 {
        let [s, n, d] = self.terms(p);
        if s <= n && s <= d {
            return self.spine.normal(p);
        }
        let v = if n <= d {
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            let a = self.spine.neck_radius();
            let (gr, gz) = box_grad(r, p[2], a, -self.radius, STUB_TOP);
            let (cx, cy) = if r > 1e-12 {
                (p[0] / r, p[1] / r)
            } else {
                (0.0, 0.0)
            };
            [gr * cx, gr * cy, gz]
        } else {
            let zc = p[2] + self.radius;
            let rho = (p[1] * p[1] + zc * zc).sqrt();
            let (gr, gx) = box_grad(rho, p[0], self.radius, -self.half_length, self.half_length);
            let (cy, cz) = if rho > 1e-12 {
                (p[1] / rho, zc / rho)
            } else {
                (0.0, 0.0)
            };
            [gx, gr * cy, gr * cz]
        };
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len > 0.0 {
            [v[0] / len, v[1] / len, v[2] / len]
        } else {
            [0.0, 0.0, 1.0]
        }
    }

    fn segment_exit(&self, a: Vec3, b: Vec3) -> Option<SegmentExit> {
        if a[2].min(b[2]) > 0.0 {
            // Above the base the union is just the spine.
            self.spine.segment_exit(a, b)
        } else if a[2].max(b[2]) < self.sliver_z {
            self.shaft_exit(a, b)
        } else {
            None
        }
    }

    fn release_point(&self) -> Vec3 {
        self.spine.release_point()
    }

    fn bounds(&self) -> (Vec3, Vec3) {
        let (_, hi) = self.spine.bounds();
        let x = self.half_length.max(hi[0]);
        let y = self.radius.max(hi[1]);
        ([-x, -y, -2.0 * self.radius], [x, y, hi[2]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Xoshiro256pp;
    use crate::sim::{resolve_step, StepOutcome};

    fn domains() -> [SpineOnDendrite; 2] {
        [
            SpineOnDendrite::with_default_dendrite(SpineGeometry::sharp()),
            SpineOnDendrite::with_default_dendrite(SpineGeometry::funneled()),
        ]
    }

    fn sample(g: &SpineOnDendrite, rng: &mut Xoshiro256pp) -> Vec3 {
        // Half the points near the neck-dendrite junction, where the surfaces meet.
        if rng.uniform() < 0.5 {
            return [
                0.3 * rng.uniform() - 0.15,
                0.3 * rng.uniform() - 0.15,
                0.1 * rng.uniform() - 0.05,
            ];
        }
        let (lo, hi) = g.bounds();
        let pad = 0.05;
        [0, 1, 2].map(|k| lo[k] - pad + (hi[k] - lo[k] + 2.0 * pad) * rng.uniform())
    }

    #[test]
    fn contains_agrees_with_sdf_sign() {
        let mut rng = Xoshiro256pp::new(4);
        for g in domains() {
            for _ in 0..300_000 {
                let p = sample(&g, &mut rng);
                let d = g.sdf(p);
                if d.abs() > 1e-9 {
                    assert_eq!(g.contains(p), d < 0.0, "{p:?} sdf {d}");
                }
            }
        }
    }

    #[test]
    fn neck_opens_into_the_dendrite() {
        let g = domains()[0];
        // A straight walk down the neck axis into the shaft never meets a wall.
        let mut p = [0.0, 0.0, 0.5];
        for _ in 0..80 {
            let q = [p[0], p[1], p[2] - 0.01];
            assert_eq!(resolve_step(&g, p, q), StepOutcome::Inside(q));
            p = q;
        }
        assert!(g.escaped(p) && g.contains([2.9, 0.0, -0.45]));
        assert!(!g.contains([0.0, 0.0, -0.91]) && !g.contains([0.3, 0.0, 0.01]));
        // The far ends absorb; the shaft wall reflects.
        assert_eq!(
            resolve_step(&g, [2.99, 0.0, -0.45], [3.01, 0.0, -0.45]),
            StepOutcome::Absorbed
        );
        match resolve_step(&g, [1.0, 0.0, -0.88], [1.0, 0.0, -0.92]) {
            StepOutcome::Inside(r) => assert!((r[2] + 0.88).abs() < 1e-9, "{r:?}"),
            StepOutcome::Absorbed => panic!("shaft wall absorbed"),
        }
    }

    #[test]
    fn steps_stay_inside_and_only_the_ends_absorb() {
        let mut rng = Xoshiro256pp::new(12);
        for g in domains() {
            for _ in 0..200_000 {
                let a = sample(&g, &mut rng);
                if !g.contains(a) {
                    continue;
                }
                let b = [0, 1, 2].map(|k| a[k] + 0.02 * rng.normal());
                match resolve_step(&g, a, b) {
                    StepOutcome::Inside(r) => assert!(g.contains(r), "{a:?} -> {b:?} left: {r:?}"),
                    StepOutcome::Absorbed => {
                        assert!(b[0].abs() >= g.half_length(), "absorbed: {a:?} -> {b:?}")
                    }
                }
            }
        }
    }

    #[test]
    fn closed_form_exits_lie_on_the_boundary() {
        let mut rng = Xoshiro256pp::new(8);
        let mut found = 0;
        for g in domains() {
            for _ in 0..300_000 {
                let a = sample(&g, &mut rng);
                if !g.contains(a) {
                    continue;
                }
                let b = [0, 1, 2].map(|k| a[k] + 0.05 * rng.normal());
                if g.contains(b) {
                    continue;
                }
                let Some(e) = g.segment_exit(a, b) else {
                    continue;
                };
                found += 1;
                assert!(g.contains(e.inside));
                assert!(g.sdf(e.boundary).abs() < 1e-9, "sdf {}", g.sdf(e.boundary));
                let n = g.normal(e.boundary);
                let dot: f64 = (0..3).map(|k| n[k] * e.normal[k]).sum();
                let inner = [0, 1, 2].map(|k| e.boundary[k] - 1e-4 * e.normal[k]);
                if g.sdf(inner) < -0.9e-4 {
                    assert!(dot > 0.999, "normal {:?} vs {n:?}", e.normal);
                }
            }
        }
        assert!(found > 2000, "only {found} closed-form exits");
    }
}
