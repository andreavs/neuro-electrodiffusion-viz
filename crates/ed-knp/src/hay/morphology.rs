//! The Hay et al. morphology, from `data/hay_l5pc.txt` (see `data/gen_hay_l5pc.py`).
//!
//! Each NEURON segment becomes one [`Compartment`]: a straight cylinder at the segment's
//! centre, along the chord of the segment, whose diameter and length are chosen so that
//! its membrane area and axial resistance equal those of the curved, tapering segment.

use crate::neuron::{Compartment, Kind, Morphology};

const DATA: &str = include_str!("../../data/hay_l5pc.txt");

/// Where a compartment sits; decides its channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    Soma,
    Axon,
    Basal,
    Apical,
}

impl Region {
    fn kind(self) -> Kind {
        match self {
            Region::Soma => Kind::Soma,
            Region::Axon => Kind::Axon,
            Region::Basal | Region::Apical => Kind::Dendrite,
        }
    }
}

/// The cell's compartments with what the channel distributions need.
#[derive(Debug, Clone)]
pub struct HayMorphology {
    /// Soma at the origin, apical axis along +z, µm.
    pub morph: Morphology,
    pub region: Vec<Region>,
    /// Path distance from the soma centre to each compartment centre, µm.
    pub distance: Vec<f64>,
    /// Longest path from the soma to an apical tip, µm (normalises the Ih gradient).
    pub max_apical: f64,
}

impl HayMorphology {
    /// The 642-compartment cell (1 soma, 2 axon, 262 basal, 377 apical).
    pub fn cell1() -> Self {
        Self::parse(DATA).expect("bundled Hay morphology is valid")
    }

    fn parse(text: &str) -> Result<Self, String> {
        let mut out = Self {
            morph: Morphology {
                compartments: Vec::new(),
            },
            region: Vec::new(),
            distance: Vec::new(),
            max_apical: 0.0,
        };
        // Index of the last compartment of each section.
        let mut last = Vec::new();
        for (ln, line) in text.lines().enumerate() {
            let err = |m: &str| format!("line {}: {m}", ln + 1);
            let mut f = line.split_whitespace();
            let region = match f.next() {
                None => continue,
                Some(w) if w.starts_with('#') => continue,
                Some("max_apical") => {
                    out.max_apical = num(f.next()).ok_or_else(|| err("bad max_apical"))?;
                    continue;
                }
                Some("soma") => Region::Soma,
                Some("axon") => Region::Axon,
                Some("basal") => Region::Basal,
                Some("apical") => Region::Apical,
                Some(w) => return Err(err(&format!("unknown region {w:?}"))),
            };
            let parent: i64 = f
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| err("parent"))?;
            let nseg: usize = f
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| err("nseg"))?;
            let v: Vec<f64> = f
                .map(|s| num(Some(s)))
                .collect::<Option<_>>()
                .ok_or_else(|| err("number"))?;
            if nseg == 0 || v.len() != 12 * nseg {
                return Err(err("expected 12 numbers per segment"));
            }
            let mut prev = match parent {
                -1 if last.is_empty() && region == Region::Soma => None,
                -1 => return Err(err("only the first section, the soma, has no parent")),
                p if p >= 0 && (p as usize) < last.len() && region != Region::Soma => {
                    Some(last[p as usize])
                }
                _ => return Err(err("parent section must come earlier")),
            };
            for s in v.chunks_exact(12) {
                let (start, centre, end) =
                    ([s[0], s[1], s[2]], [s[3], s[4], s[5]], [s[6], s[7], s[8]]);
                let (area, k) = (s[9], s[10]);
                if !(area > 0.0 && k > 0.0) {
                    return Err(err("area and ∫ds/d² must be positive"));
                }
                // π d L = area and L/d² = k.
                let diam = (area / (std::f64::consts::PI * k)).cbrt();
                let len = k * diam * diam;
                let chord = [0, 1, 2].map(|a| end[a] - start[a]);
                let norm = chord.iter().map(|c| c * c).sum::<f64>().sqrt();
                if norm <= 0.0 {
                    return Err(err("zero-length segment"));
                }
                let half = [0, 1, 2].map(|a| 0.5 * len * chord[a] / norm);
                out.morph.compartments.push(Compartment {
                    parent: prev,
                    start: [0, 1, 2].map(|a| centre[a] - half[a]),
                    end: [0, 1, 2].map(|a| centre[a] + half[a]),
                    diam,
                    kind: region.kind(),
                });
                out.region.push(region);
                out.distance.push(s[11]);
                prev = Some(out.morph.len() - 1);
            }
            last.push(out.morph.len() - 1);
        }
        out.morph.validate()?;
        if out.max_apical <= 0.0 {
            return Err("missing or non-positive max_apical".into());
        }
        Ok(out)
    }

    pub fn len(&self) -> usize {
        self.morph.len()
    }

    pub fn is_empty(&self) -> bool {
        self.morph.is_empty()
    }
}

fn num(s: Option<&str>) -> Option<f64> {
    s?.parse().ok().filter(|x: &f64| x.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell1_has_the_neuron_segment_counts() {
        let m = HayMorphology::cell1();
        let count = |r| m.region.iter().filter(|&&x| x == r).count();
        assert_eq!(m.len(), 642);
        assert_eq!(
            [Region::Soma, Region::Axon, Region::Basal, Region::Apical].map(count),
            [1, 2, 262, 377]
        );
        assert_eq!(m.region[0], Region::Soma);
        assert!((1250.0..1350.0).contains(&m.max_apical));
        // Apical tuft about 1.2 mm above the soma, basal tips about 0.2 mm below.
        let z = |c: &Compartment| c.centre()[2];
        let cs = &m.morph.compartments;
        let top = cs.iter().map(z).fold(f64::MIN, f64::max);
        let bottom = cs.iter().map(z).fold(f64::MAX, f64::min);
        assert!((1100.0..1250.0).contains(&top) && (-250.0..-150.0).contains(&bottom));
        // Without spines, the cell's membrane is about 3·10⁴ µm².
        let area: f64 = cs.iter().map(Compartment::area).sum();
        assert!((2.5e4..3.5e4).contains(&area), "{area}");
    }

    #[test]
    fn equivalent_cylinders_keep_area_and_resistance() {
        let line = "max_apical 100\n# comment\n\
                    soma -1 1 0 0 -10 0 0 0 0 0 10 1256.6 0.05 0\n\
                    basal 0 2 0 0 10 0 0 15 0 0 20 30 2 5 0 0 20 0 0 25 0 0 30 20 4 15";
        let m = HayMorphology::parse(line).unwrap();
        let c = &m.morph.compartments;
        assert_eq!(c.len(), 3);
        assert_eq!((c[1].parent, c[2].parent), (Some(0), Some(1)));
        assert!((c[1].area() - 30.0).abs() < 1e-9);
        // 4 r_a k / π at r_a = 100 Ω·cm, MΩ.
        let r = 4.0 * 100.0 * 1e-2 * 2.0 / std::f64::consts::PI;
        assert!((c[1].axial_resistance(100.0) - r).abs() < 1e-9);
        assert_eq!(
            c[2].centre().map(|x| (x * 1e9).round() / 1e9),
            [0.0, 0.0, 25.0]
        );
        assert_eq!(m.distance, vec![0.0, 5.0, 15.0]);
    }

    #[test]
    fn bad_input_is_rejected() {
        assert!(HayMorphology::parse("basal 3 1 0 0 0 0 0 1 0 0 2 1 1 0").is_err());
        assert!(HayMorphology::parse("soma -1 1 0 0 0").is_err());
        assert!(HayMorphology::parse("dend -1 1").is_err());
        // Without max_apical the Ih gradient is undefined.
        assert!(HayMorphology::parse("soma -1 1 0 0 -10 0 0 0 0 0 10 1256.6 0.05 0").is_err());
    }
}
