//! Electric fields E = −∇φ in mV/µm acting on the particles.

use crate::geometry::{SpineGeometry, Vec3};

/// A static electric field. Add new fields by implementing this trait.
pub trait ElectricField: Send + Sync {
    /// E(p) in mV/µm.
    fn field(&self, p: Vec3) -> Vec3;

    /// Evaluates the field at many points: `out[i] = E(x[i], y[i], z[i])`. The stepper calls
    /// this once per batch of particles, so the per-point calls are statically dispatched.
    fn field_batch(&self, x: &[f64], y: &[f64], z: &[f64], out: &mut [Vec3]) {
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.field([x[i], y[i], z[i]]);
        }
    }

    /// `Some(E)` when the field is the same everywhere, so the stepper can skip
    /// per-particle evaluation.
    fn uniform(&self) -> Option<Vec3> {
        None
    }
}

/// No field: pure diffusion.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoField;

impl ElectricField for NoField {
    fn field(&self, _p: Vec3) -> Vec3 {
        [0.0; 3]
    }
    fn uniform(&self) -> Option<Vec3> {
        Some([0.0; 3])
    }
}

/// The same field everywhere.
#[derive(Debug, Clone, Copy)]
pub struct UniformField(pub Vec3);

impl UniformField {
    /// Field along the spine axis. Positive `e_out` (mV/µm) points out of the spine,
    /// towards the neck base, which pushes cations out.
    pub fn axial_out(e_out: f64) -> Self {
        Self([0.0, 0.0, -e_out])
    }
}

impl ElectricField for UniformField {
    fn field(&self, _p: Vec3) -> Vec3 {
        self.0
    }
    fn uniform(&self) -> Option<Vec3> {
        Some(self.0)
    }
}

/// A potential drop concentrated in the neck: uniform axial field for z below `z_top`,
/// zero in the head.
#[derive(Debug, Clone, Copy)]
pub struct NeckField {
    /// Field magnitude in mV/µm; positive points out of the spine.
    pub e_out: f64,
    /// Height (µm) above which the field vanishes.
    pub z_top: f64,
}

impl NeckField {
    /// Places a potential drop of `delta_v_mv` (head minus base) across the neck of `spine`.
    pub fn for_spine(spine: &SpineGeometry, delta_v_mv: f64) -> Self {
        let z_top = spine.neck_top();
        Self {
            e_out: delta_v_mv / z_top,
            z_top,
        }
    }
}

impl ElectricField for NeckField {
    fn field(&self, p: Vec3) -> Vec3 {
        if p[2] < self.z_top {
            [0.0, 0.0, -self.e_out]
        } else {
            [0.0; 3]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neck_field_integrates_to_the_potential_drop() {
        let g = SpineGeometry::sharp();
        let f = NeckField::for_spine(&g, 10.0);
        // ΔV = −∫ E·dz from base to head = e_out * z_top.
        let n = 1000;
        let h = g.head_centre()[2] / n as f64;
        let dv: f64 = (0..n)
            .map(|i| -f.field([0.0, 0.0, (i as f64 + 0.5) * h])[2] * h)
            .sum();
        assert!((dv - 10.0).abs() < 0.05, "ΔV = {dv}");
        assert_eq!(f.field(g.head_centre()), [0.0; 3]);
        assert!(f.uniform().is_none());
    }

    #[test]
    fn axial_out_points_down_the_neck() {
        assert_eq!(
            UniformField::axial_out(2.0).field([0.0; 3]),
            [0.0, 0.0, -2.0]
        );
    }
}
