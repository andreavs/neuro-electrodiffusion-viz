//! Neuron morphology as a tree of cylindrical compartments.

/// Membrane type of a compartment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Hodgkin-Huxley channels.
    Soma,
    /// Passive leak only; receives synaptic input.
    Dendrite,
    /// Hodgkin-Huxley channels.
    Axon,
}

impl Kind {
    /// Whether the compartment carries Hodgkin-Huxley channels.
    pub fn is_active(self) -> bool {
        !matches!(self, Kind::Dendrite)
    }
}

/// A cylinder from `start` to `end` (µm) with diameter `diam` (µm).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Compartment {
    /// Parent compartment; always has a smaller index. `None` only for the root.
    pub parent: Option<usize>,
    pub start: [f64; 3],
    pub end: [f64; 3],
    pub diam: f64,
    pub kind: Kind,
}

impl Compartment {
    /// Length, µm.
    pub fn length(&self) -> f64 {
        dist(self.start, self.end)
    }

    /// Lateral membrane area π d L, µm².
    pub fn area(&self) -> f64 {
        std::f64::consts::PI * self.diam * self.length()
    }

    /// Centre point, µm. This is where the compartment's point source sits.
    pub fn centre(&self) -> [f64; 3] {
        [0, 1, 2].map(|k| 0.5 * (self.start[k] + self.end[k]))
    }

    /// Axial resistance from end to end, MΩ, for axial resistivity `ra`
    /// in Ω·cm: `4 r_a L / (π d²)`.
    pub fn axial_resistance(&self, ra: f64) -> f64 {
        // 1 Ω·cm = 1e4 Ω·µm = 1e-2 MΩ·µm.
        4.0 * ra * 1e-2 * self.length() / (std::f64::consts::PI * self.diam * self.diam)
    }
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Compartments ordered so every parent precedes its children (index 0 is
/// the root), which is what the Hines solver needs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Morphology {
    pub compartments: Vec<Compartment>,
}

impl Morphology {
    /// A morphology with a single root compartment.
    pub fn with_root(start: [f64; 3], end: [f64; 3], diam: f64, kind: Kind) -> Self {
        Self {
            compartments: vec![Compartment {
                parent: None,
                start,
                end,
                diam,
                kind,
            }],
        }
    }

    pub fn len(&self) -> usize {
        self.compartments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.compartments.is_empty()
    }

    /// Append a straight section from `start` to `end` split into `n` equal
    /// compartments, attached to `parent`, with diameter tapering linearly
    /// from `d0` to `d1`. Returns the indices of the new compartments,
    /// proximal first.
    #[allow(clippy::too_many_arguments)]
    pub fn add_section(
        &mut self,
        parent: usize,
        start: [f64; 3],
        end: [f64; 3],
        d0: f64,
        d1: f64,
        n: usize,
        kind: Kind,
    ) -> std::ops::Range<usize> {
        assert!(parent < self.len(), "parent {parent} does not exist");
        assert!(n > 0 && d0 > 0.0 && d1 > 0.0);
        let first = self.len();
        let lerp = |s: f64| [0, 1, 2].map(|k| start[k] + s * (end[k] - start[k]));
        for j in 0..n {
            let (s0, s1) = (j as f64 / n as f64, (j + 1) as f64 / n as f64);
            let mid = 0.5 * (s0 + s1);
            self.compartments.push(Compartment {
                parent: Some(if j == 0 { parent } else { first + j - 1 }),
                start: lerp(s0),
                end: lerp(s1),
                diam: d0 + mid * (d1 - d0),
                kind,
            });
        }
        first..first + n
    }

    /// Default layer-5-like pyramidal cell, about 1.2 mm from basal tips to
    /// apical tuft, soma centred at the origin and the apical axis along +z:
    /// a 30 × 30 µm soma, a tapering 750 µm apical trunk with two obliques and a
    /// three-branch tuft reaching z ≈ 1000 µm, four basal dendrites, and a
    /// 100 µm axon initial segment pointing down. 115 compartments.
    pub fn pyramidal() -> Self {
        let mut m = Self::with_root([0.0, 0.0, -15.0], [0.0, 0.0, 15.0], 30.0, Kind::Soma);
        let dend = Kind::Dendrite;
        let trunk = m.add_section(0, [0.0, 0.0, 15.0], [0.0, 0.0, 765.0], 4.0, 1.5, 30, dend);
        // Obliques from the trunk at z ≈ 260 µm and 460 µm.
        for (k, sx) in [(10, 1.0), (18, -1.0)] {
            let c = m.compartments[trunk.start + k].end;
            let tip = [c[0] + sx * 140.0, 0.0, c[2] + 50.0];
            m.add_section(trunk.start + k, c, tip, 1.0, 0.6, 6, dend);
        }
        // Tuft: three branches fanning out from the top of the trunk.
        let top = m.compartments[trunk.end - 1].end;
        for a in [0.0f64, 120.0, 240.0] {
            let (s, c) = a.to_radians().sin_cos();
            let tip = [top[0] + 180.0 * c, top[1] + 180.0 * s, 1000.0];
            m.add_section(trunk.end - 1, top, tip, 1.2, 0.5, 10, dend);
        }
        // Basal dendrites: four branches spreading sideways and down.
        for a in [45.0f64, 135.0, 225.0, 315.0] {
            let (s, c) = a.to_radians().sin_cos();
            let base = [15.0 * c, 15.0 * s, -5.0];
            let tip = [base[0] + 170.0 * c, base[1] + 170.0 * s, -120.0];
            m.add_section(0, base, tip, 1.5, 0.6, 8, dend);
        }
        m.add_section(
            0,
            [0.0, 0.0, -10.0],
            [0.0, 0.0, -110.0],
            1.2,
            1.0,
            10,
            Kind::Axon,
        );
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_morphology_is_a_valid_tree() {
        let m = Morphology::pyramidal();
        assert_eq!(m.len(), 115);
        assert_eq!(m.compartments[0].kind, Kind::Soma);
        assert!(m.compartments[0].parent.is_none());
        for (i, c) in m.compartments.iter().enumerate().skip(1) {
            assert!(c.parent.unwrap() < i);
            assert!(c.length() > 0.0 && c.diam > 0.0);
        }
        let zs: Vec<f64> = m.compartments.iter().map(|c| c.centre()[2]).collect();
        let top = zs.iter().cloned().fold(f64::MIN, f64::max);
        let bottom = zs.iter().cloned().fold(f64::MAX, f64::min);
        assert!((900.0..1300.0).contains(&(top - bottom)), "{top} {bottom}");
    }

    #[test]
    fn sections_connect_and_taper() {
        let mut m = Morphology::with_root([0.0; 3], [0.0, 0.0, 10.0], 10.0, Kind::Soma);
        let r = m.add_section(
            0,
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 110.0],
            2.0,
            1.0,
            4,
            Kind::Dendrite,
        );
        assert_eq!(r, 1..5);
        let c = &m.compartments;
        assert_eq!(c[1].parent, Some(0));
        for i in 2..5 {
            assert_eq!(c[i].parent, Some(i - 1));
            assert_eq!(c[i].start, c[i - 1].end);
            assert!(c[i].diam < c[i - 1].diam);
        }
        assert_eq!(c[4].end, [0.0, 0.0, 110.0]);
        assert!((c[1].length() - 25.0).abs() < 1e-12);
        assert!((c[1].diam - 1.875).abs() < 1e-12);
        assert_eq!(c[1].centre(), [0.0, 0.0, 22.5]);
    }

    #[test]
    fn axial_resistance_and_area_units() {
        // 100 µm of 1 µm cable at 100 Ω·cm: 4·1e6 Ω·µm·100 µm / π µm² ≈ 127.3 MΩ.
        let c = Compartment {
            parent: None,
            start: [0.0; 3],
            end: [100.0, 0.0, 0.0],
            diam: 1.0,
            kind: Kind::Axon,
        };
        assert!((c.axial_resistance(100.0) - 400.0 / std::f64::consts::PI).abs() < 1e-9);
        assert!((c.area() - 100.0 * std::f64::consts::PI).abs() < 1e-9);
        assert!(c.kind.is_active() && !Kind::Dendrite.is_active());
    }
}
