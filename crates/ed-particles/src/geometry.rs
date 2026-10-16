//! Domain geometry as signed distance functions (negative inside, µm).
//!
//! The dendritic spine is axisymmetric about the z axis. The absorbing disk at the
//! base of the neck lies in the plane z = 0; the neck rises along +z into the head.

/// A point or vector in µm.
pub type Vec3 = [f64; 3];

/// A closed domain that particles diffuse inside.
pub trait Geometry {
    /// Signed distance to the boundary in µm, negative inside.
    fn sdf(&self, p: Vec3) -> f64;

    /// Whether `p` lies inside the domain. Must agree with the sign of [`Geometry::sdf`];
    /// implementations override it with a cheaper test because it runs every step.
    fn contains(&self, p: Vec3) -> bool {
        self.sdf(p) < 0.0
    }

    /// Whether the boundary point `b`, where a particle left the domain, belongs to an
    /// absorbing part of the boundary. Other walls reflect.
    fn absorbs(&self, b: Vec3) -> bool;

    /// Outward unit normal of the boundary near `p`, from the gradient of the SDF.
    fn normal(&self, p: Vec3) -> Vec3 {
        const H: f64 = 1e-6;
        let mut g = [0.0; 3];
        for (k, gk) in g.iter_mut().enumerate() {
            let mut a = p;
            let mut b = p;
            a[k] += H;
            b[k] -= H;
            *gk = self.sdf(a) - self.sdf(b);
        }
        let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
        if len > 0.0 {
            [g[0] / len, g[1] / len, g[2] / len]
        } else {
            [0.0, 0.0, 1.0]
        }
    }

    /// Where the straight segment from `a` (inside) to `b` (outside) first leaves the
    /// domain, for geometries that can compute it in closed form. `None` (the default) makes
    /// the stepper locate the crossing numerically from the SDF and [`Geometry::normal`].
    fn segment_exit(&self, _a: Vec3, _b: Vec3) -> Option<SegmentExit> {
        None
    }

    /// Where particles are released.
    fn release_point(&self) -> Vec3;

    /// Axis-aligned bounding box `(min, max)` of the domain, for cameras and sampling.
    fn bounds(&self) -> (Vec3, Vec3);
}

/// Where a step segment leaves the domain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmentExit {
    /// A point on the segment just inside the domain, before the exit.
    pub inside: Vec3,
    /// The exit point on the boundary.
    pub boundary: Vec3,
    /// Outward unit normal at the exit point.
    pub normal: Vec3,
}

/// Unbounded space with no walls; useful for testing the free-motion update.
#[derive(Debug, Clone, Copy, Default)]
pub struct FreeSpace;

impl Geometry for FreeSpace {
    fn sdf(&self, _p: Vec3) -> f64 {
        f64::NEG_INFINITY
    }
    fn contains(&self, _p: Vec3) -> bool {
        true
    }
    fn absorbs(&self, _b: Vec3) -> bool {
        false
    }
    fn release_point(&self) -> Vec3 {
        [0.0; 3]
    }
    fn bounds(&self) -> (Vec3, Vec3) {
        ([f64::NEG_INFINITY; 3], [f64::INFINITY; 3])
    }
}

/// How the neck joins the head.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Junction {
    /// The cylinder meets the sphere directly.
    Sharp,
    /// A toroidal fillet with the given tube radius (µm), tangent to both the neck
    /// and the head.
    Funnel { radius: f64 },
}

/// Thesis defaults (Publication 3): head radius, neck radius, neck lengths, funnel ε.
pub const HEAD_RADIUS: f64 = 0.5;
pub const NECK_RADIUS: f64 = 0.1;
pub const SHARP_NECK_LENGTH: f64 = 1.0;
pub const FUNNEL_NECK_LENGTH: f64 = 0.89;
pub const FUNNEL_EPSILON: f64 = 1e-3;

/// Boundary points this close to z = 0 (µm) count as hitting the absorbing base.
const ABSORB_TOL: f64 = 1e-7;

/// Dendritic spine: spherical head on a cylindrical neck whose base (z = 0) absorbs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpineGeometry {
    head_radius: f64,
    neck_radius: f64,
    neck_length: f64,
    junction: Junction,
    /// z of the head centre.
    head_z: f64,
    /// Funnel only: tube centre ring at (r, z) = (neck_radius + ρ, neck_length);
    /// the fillet spans z in [neck_length, fillet_top].
    fillet_top: f64,
}

impl SpineGeometry {
    /// Builds a spine. `neck_length` is the length of the straight cylinder: from the base
    /// to where it meets the head (sharp) or to where the funnel starts (funnel).
    pub fn new(head_radius: f64, neck_radius: f64, neck_length: f64, junction: Junction) -> Self {
        assert!(
            neck_radius > 0.0 && neck_radius < head_radius,
            "need 0 < neck radius < head radius"
        );
        assert!(neck_length > 0.0, "neck length must be positive");
        let (head_z, fillet_top) = match junction {
            Junction::Sharp => (
                neck_length + (head_radius * head_radius - neck_radius * neck_radius).sqrt(),
                neck_length,
            ),
            Junction::Funnel { radius: rho } => {
                assert!(rho > 0.0, "funnel radius must be positive");
                let sum = head_radius + rho;
                let ring = neck_radius + rho;
                let head_z = neck_length + (sum * sum - ring * ring).sqrt();
                // Tangent point between torus and sphere, on the line joining their centres.
                let top = head_z + head_radius * (neck_length - head_z) / sum;
                (head_z, top)
            }
        };
        Self {
            head_radius,
            neck_radius,
            neck_length,
            junction,
            head_z,
            fillet_top,
        }
    }

    /// Thesis geometry with a sharp neck-head junction.
    pub fn sharp() -> Self {
        Self::new(HEAD_RADIUS, NECK_RADIUS, SHARP_NECK_LENGTH, Junction::Sharp)
    }

    /// Thesis geometry with a toroidal funnel of radius r_h/2 − ε.
    pub fn funneled() -> Self {
        Self::new(
            HEAD_RADIUS,
            NECK_RADIUS,
            FUNNEL_NECK_LENGTH,
            Junction::Funnel {
                radius: HEAD_RADIUS / 2.0 - FUNNEL_EPSILON,
            },
        )
    }

    pub fn head_radius(&self) -> f64 {
        self.head_radius
    }
    pub fn neck_radius(&self) -> f64 {
        self.neck_radius
    }
    pub fn neck_length(&self) -> f64 {
        self.neck_length
    }
    pub fn junction(&self) -> Junction {
        self.junction
    }
    /// Centre of the spherical head.
    pub fn head_centre(&self) -> Vec3 {
        [0.0, 0.0, self.head_z]
    }
    /// Height where the neck (including any funnel) meets the head.
    pub fn neck_top(&self) -> f64 {
        self.fillet_top
    }

    #[inline]
    fn sdf_rz(&self, r: f64, z: f64) -> f64 {
        let a = self.neck_radius;
        let dz = z - self.head_z;
        let sphere = (r * r + dz * dz).sqrt() - self.head_radius;
        // The neck cylinder runs up to the head centre so it overlaps the head.
        let neck = box_sdf(r, z, a, 0.0, self.head_z);
        let mut d = sphere.min(neck);
        if let Junction::Funnel { radius: rho } = self.junction {
            let ring = a + rho;
            let block = box_sdf(r, z, ring, self.neck_length, self.fillet_top);
            let dr = r - ring;
            let dzt = z - self.neck_length;
            let tube = (dr * dr + dzt * dzt).sqrt() - rho;
            d = d.min(block.max(-tube));
        }
        d
    }

    /// Analytic gradient of [`SpineGeometry::sdf_rz`]: the gradient of whichever term is active.
    fn grad_rz(&self, r: f64, z: f64) -> (f64, f64) {
        let a = self.neck_radius;
        let dz = z - self.head_z;
        let sphere = (r * r + dz * dz).sqrt() - self.head_radius;
        let neck = box_sdf(r, z, a, 0.0, self.head_z);
        let (mut d, mut g) = if sphere < neck {
            (sphere, unit2(r, dz))
        } else {
            (neck, box_grad(r, z, a, 0.0, self.head_z))
        };
        if let Junction::Funnel { radius: rho } = self.junction {
            let ring = a + rho;
            let block = box_sdf(r, z, ring, self.neck_length, self.fillet_top);
            let dr = r - ring;
            let dzt = z - self.neck_length;
            let tube = (dr * dr + dzt * dzt).sqrt() - rho;
            let (f, gf) = if -tube > block {
                let (ur, uz) = unit2(dr, dzt);
                (-tube, (-ur, -uz))
            } else {
                (
                    block,
                    box_grad(r, z, ring, self.neck_length, self.fillet_top),
                )
            };
            if f < d {
                d = f;
                g = gf;
            }
        }
        let _ = d;
        g
    }
}

/// Gradient (∂/∂r, ∂/∂z) of [`box_sdf`].
#[inline]
fn box_grad(r: f64, z: f64, half_r: f64, z0: f64, z1: f64) -> (f64, f64) {
    let dx = r - half_r;
    let sz = if z >= 0.5 * (z0 + z1) { 1.0 } else { -1.0 };
    let dz = (z - 0.5 * (z0 + z1)).abs() - 0.5 * (z1 - z0);
    if dx > 0.0 || dz > 0.0 {
        let ox = dx.max(0.0);
        let oz = dz.max(0.0);
        let len = (ox * ox + oz * oz).sqrt();
        (ox / len, sz * oz / len)
    } else if dx > dz {
        (1.0, 0.0)
    } else {
        (0.0, sz)
    }
}

/// Unit vector along (u, v); zero vector if both vanish.
#[inline]
fn unit2(u: f64, v: f64) -> (f64, f64) {
    let len = (u * u + v * v).sqrt();
    if len > 0.0 {
        (u / len, v / len)
    } else {
        (0.0, 0.0)
    }
}

/// SDF of the axisymmetric solid cylinder r ≤ half_r, z0 ≤ z ≤ z1, in the (r, z) plane.
#[inline]
fn box_sdf(r: f64, z: f64, half_r: f64, z0: f64, z1: f64) -> f64 {
    let dx = r - half_r;
    let dz = (z - 0.5 * (z0 + z1)).abs() - 0.5 * (z1 - z0);
    let ox = dx.max(0.0);
    let oz = dz.max(0.0);
    (ox * ox + oz * oz).sqrt() + dx.max(dz).min(0.0)
}

impl Geometry for SpineGeometry {
    fn sdf(&self, p: Vec3) -> f64 {
        self.sdf_rz((p[0] * p[0] + p[1] * p[1]).sqrt(), p[2])
    }

    #[inline]
    fn contains(&self, p: Vec3) -> bool {
        // Evaluated without short-circuiting: particles are spread over neck and head, so
        // data-dependent branches here would mispredict often.
        let z = p[2];
        let r2 = p[0] * p[0] + p[1] * p[1];
        let a = self.neck_radius;
        let dz = z - self.head_z;
        let in_neck = (r2 < a * a) & (z < self.head_z);
        let in_head = r2 + dz * dz < self.head_radius * self.head_radius;
        let mut inside = in_neck | in_head;
        if let Junction::Funnel { radius: rho } = self.junction {
            let ring = a + rho;
            let dzt = z - self.neck_length;
            // Outside the tube, (r − ring)² + dzt² > ρ², rearranged to avoid the square root:
            // 2·r·ring < q with q = r² + ring² + dzt² − ρ², i.e. q > 0 and 4·r²·ring² < q².
            let q = r2 + ring * ring + dzt * dzt - rho * rho;
            let in_fillet = (z > self.neck_length)
                & (z < self.fillet_top)
                & (r2 < ring * ring)
                & (q > 0.0)
                & (4.0 * r2 * ring * ring < q * q);
            inside |= in_fillet;
        }
        inside & (z > 0.0)
    }

    fn absorbs(&self, b: Vec3) -> bool {
        // The base disk is the plane z = 0; crossing points are located to ~1e-9 µm.
        b[2] < ABSORB_TOL
    }

    fn normal(&self, p: Vec3) -> Vec3 {
        let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
        let (gr, gz) = self.grad_rz(r, p[2]);
        let (cx, cy) = if r > 1e-12 {
            (p[0] / r, p[1] / r)
        } else {
            (0.0, 0.0)
        };
        let v = [gr * cx, gr * cy, gz];
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len > 0.0 {
            [v[0] / len, v[1] / len, v[2] / len]
        } else {
            [0.0, 0.0, 1.0]
        }
    }

    /// Closed-form exit through the sphere, the neck cylinder or the base plane. Segments
    /// that reach the funnel's height range return `None` (the torus would need a quartic).
    fn segment_exit(&self, a: Vec3, b: Vec3) -> Option<SegmentExit> {
        if matches!(self.junction, Junction::Funnel { .. })
            && a[2].max(b[2]) > self.neck_length
            && a[2].min(b[2]) < self.fillet_top
        {
            return None;
        }
        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let len2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if len2 == 0.0 {
            return None;
        }
        // Candidate crossings (t along the segment, surface): 0 sphere, 1 cylinder, 2 base.
        let mut cand = [(f64::INFINITY, 0u8); 5];
        let mut nc = 0;
        let mut push = |t: f64, s: u8| {
            if t > 0.0 && t <= 1.0 {
                cand[nc] = (t, s);
                nc += 1;
            }
        };
        let oc = [a[0], a[1], a[2] - self.head_z];
        let bq = oc[0] * d[0] + oc[1] * d[1] + oc[2] * d[2];
        let cq =
            oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - self.head_radius * self.head_radius;
        let disc = bq * bq - len2 * cq;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            push((-bq - sq) / len2, 0);
            push((-bq + sq) / len2, 0);
        }
        let ar = d[0] * d[0] + d[1] * d[1];
        if ar > 0.0 {
            let br = a[0] * d[0] + a[1] * d[1];
            let cr = a[0] * a[0] + a[1] * a[1] - self.neck_radius * self.neck_radius;
            let disc = br * br - ar * cr;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                push((-br - sq) / ar, 1);
                push((-br + sq) / ar, 1);
            }
        }
        if d[2] != 0.0 {
            push(-a[2] / d[2], 2);
        }
        let cand = &mut cand[..nc];
        cand.sort_unstable_by(|x, y| x.0.total_cmp(&y.0));
        // Membership only changes at candidates, so the first candidate past which the
        // segment is outside is the exit.
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
            let normal = match surface {
                0 => {
                    let inv = 1.0 / self.head_radius;
                    [p[0] * inv, p[1] * inv, (p[2] - self.head_z) * inv]
                }
                1 => {
                    let inv = 1.0 / self.neck_radius;
                    [p[0] * inv, p[1] * inv, 0.0]
                }
                _ => [0.0, 0.0, -1.0],
            };
            let boundary = if surface == 2 { [p[0], p[1], 0.0] } else { p };
            return Some(SegmentExit {
                inside,
                boundary,
                normal,
            });
        }
        None
    }

    fn release_point(&self) -> Vec3 {
        self.head_centre()
    }

    fn bounds(&self) -> (Vec3, Vec3) {
        let rh = self.head_radius;
        ([-rh, -rh, 0.0], [rh, rh, self.head_z + rh])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Xoshiro256pp;

    fn sample_box(g: &SpineGeometry, rng: &mut Xoshiro256pp) -> Vec3 {
        let (lo, hi) = g.bounds();
        let pad = 0.05;
        [
            lo[0] - pad + (hi[0] - lo[0] + 2.0 * pad) * rng.uniform(),
            lo[1] - pad + (hi[1] - lo[1] + 2.0 * pad) * rng.uniform(),
            lo[2] - pad + (hi[2] - lo[2] + 2.0 * pad) * rng.uniform(),
        ]
    }

    #[test]
    fn contains_agrees_with_sdf_sign() {
        let mut rng = Xoshiro256pp::new(3);
        for g in [SpineGeometry::sharp(), SpineGeometry::funneled()] {
            for _ in 0..200_000 {
                let p = sample_box(&g, &mut rng);
                let d = g.sdf(p);
                if d.abs() > 1e-9 {
                    assert_eq!(g.contains(p), d < 0.0, "{p:?} sdf {d} in {g:?}");
                }
            }
        }
    }

    #[test]
    fn sharp_spine_landmarks() {
        let g = SpineGeometry::sharp();
        let c = g.head_centre();
        assert!(g.contains(c));
        assert!(g.contains([0.0, 0.0, 0.01]));
        assert!(!g.contains([0.0, 0.0, -0.01]));
        assert!(!g.contains([0.15, 0.0, 0.5])); // outside the neck wall
        assert!((g.sdf([0.0, 0.0, 0.5]) + NECK_RADIUS).abs() < 1e-12);
        assert!((g.sdf([0.0, 0.0, c[2] + HEAD_RADIUS])).abs() < 1e-12);
        assert!((g.neck_top() - SHARP_NECK_LENGTH).abs() < 1e-12);
    }

    #[test]
    fn funnel_is_tangent_and_fills_the_junction() {
        let g = SpineGeometry::funneled();
        let Junction::Funnel { radius: rho } = g.junction() else {
            unreachable!()
        };
        let a = g.neck_radius();
        // Just above where the funnel starts the wall flares out; a sharp spine with the
        // same neck would not contain this point.
        let p = [a + 0.02, 0.0, FUNNEL_NECK_LENGTH + 0.12];
        assert!(g.contains(p));
        // The tube boundary is on the domain boundary and its normal points into the tube.
        let theta: f64 = 0.6;
        let b = [
            a + rho - rho * theta.cos(),
            0.0,
            FUNNEL_NECK_LENGTH + rho * theta.sin(),
        ];
        assert!(g.sdf(b).abs() < 1e-9, "sdf on fillet {}", g.sdf(b));
        let n = g.normal(b);
        let expect = [theta.cos(), 0.0, -theta.sin()];
        let dot: f64 = (0..3).map(|k| n[k] * expect[k]).sum();
        assert!(dot > 0.999, "normal {n:?}");
        // Fillet meets the sphere tangentially: the top of the fillet is on the head surface.
        let c = g.head_centre();
        let rk = HEAD_RADIUS * (a + rho) / (HEAD_RADIUS + rho);
        let dist = (rk * rk + (g.neck_top() - c[2]).powi(2)).sqrt();
        assert!((dist - HEAD_RADIUS).abs() < 1e-12);
    }

    #[test]
    fn analytic_normal_matches_numerical_gradient_on_the_boundary() {
        struct Numeric(SpineGeometry);
        impl Geometry for Numeric {
            fn sdf(&self, p: Vec3) -> f64 {
                self.0.sdf(p)
            }
            fn absorbs(&self, o: Vec3) -> bool {
                self.0.absorbs(o)
            }
            fn release_point(&self) -> Vec3 {
                self.0.release_point()
            }
            fn bounds(&self) -> (Vec3, Vec3) {
                self.0.bounds()
            }
        }
        let mut rng = Xoshiro256pp::new(9);
        for g in [SpineGeometry::sharp(), SpineGeometry::funneled()] {
            let num = Numeric(g);
            let mut checked = 0;
            while checked < 2000 {
                let p = sample_box(&g, &mut rng);
                // Project onto the boundary by marching along the numerical gradient.
                let mut q = p;
                for _ in 0..50 {
                    let d = g.sdf(q);
                    let n = num.normal(q);
                    q = [q[0] - d * n[0], q[1] - d * n[1], q[2] - d * n[2]];
                }
                if g.sdf(q).abs() > 1e-9 || q[2] < 1e-3 {
                    continue;
                }
                // Skip the creases where two surfaces meet at an angle.
                let a = num.normal(q);
                let b = g.normal(q);
                let dot: f64 = (0..3).map(|k| a[k] * b[k]).sum();
                let near_crease = (q[2] - g.neck_length()).abs() < 1e-3
                    || matches!(g.junction(), Junction::Sharp)
                        && (q[2] - g.neck_top()).abs() < 1e-2;
                if !near_crease {
                    assert!(dot > 0.9999, "{q:?}: numeric {a:?} analytic {b:?}");
                    checked += 1;
                }
            }
        }
    }

    #[test]
    fn closed_form_exit_matches_the_sdf() {
        let mut rng = Xoshiro256pp::new(21);
        for g in [SpineGeometry::sharp(), SpineGeometry::funneled()] {
            let mut found = 0;
            for _ in 0..200_000 {
                let a = sample_box(&g, &mut rng);
                if !g.contains(a) {
                    continue;
                }
                let s = 0.05;
                let b = [
                    a[0] + s * rng.normal(),
                    a[1] + s * rng.normal(),
                    a[2] + s * rng.normal(),
                ];
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
                // Away from creases the analytic normal equals the SDF gradient.
                if g.sdf([
                    e.boundary[0] - 1e-4 * e.normal[0],
                    e.boundary[1] - 1e-4 * e.normal[1],
                    e.boundary[2] - 1e-4 * e.normal[2],
                ]) < -0.9e-4
                {
                    assert!(
                        dot > 0.999,
                        "normal {:?} vs {n:?} at {:?}",
                        e.normal,
                        e.boundary
                    );
                }
            }
            assert!(found > 1000, "only {found} closed-form exits");
        }
    }

    #[test]
    fn normals_point_outward() {
        let g = SpineGeometry::sharp();
        let n = g.normal([NECK_RADIUS, 0.0, 0.5]);
        assert!((n[0] - 1.0).abs() < 1e-6);
        let c = g.head_centre();
        let n = g.normal([0.0, HEAD_RADIUS, c[2]]);
        assert!((n[1] - 1.0).abs() < 1e-6);
        let n = g.normal([0.0, 0.0, 0.0]);
        assert!((n[2] + 1.0).abs() < 1e-6);
    }
}
