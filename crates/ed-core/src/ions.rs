//! Ion species and electrochemical helpers.

use crate::constants::{psi_mv, T_DEFAULT};

/// An ion species with its valence and free diffusion coefficient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IonSpecies {
    pub name: &'static str,
    /// Valence z.
    pub z: i32,
    /// Free diffusion coefficient, µm²/ms.
    pub d: f64,
}

/// Diffusion coefficients from Publication 1, Table 2 (Halnes et al. 2013).
pub const NA: IonSpecies = IonSpecies {
    name: "Na+",
    z: 1,
    d: 1.33,
};
pub const K: IonSpecies = IonSpecies {
    name: "K+",
    z: 1,
    d: 1.96,
};
pub const CL: IonSpecies = IonSpecies {
    name: "Cl-",
    z: -1,
    d: 2.03,
};
pub const CA: IonSpecies = IonSpecies {
    name: "Ca2+",
    z: 2,
    d: 0.71,
};

/// Nernst reversal potential in mV for a species with valence `z`,
/// given extracellular and intracellular concentrations (any matching unit).
pub fn nernst_mv(z: i32, c_out: f64, c_in: f64, t: f64) -> f64 {
    assert!(z != 0, "Nernst potential is undefined for neutral species");
    psi_mv(t) / z as f64 * (c_out / c_in).ln()
}

/// Nernst potential at the default temperature.
pub fn nernst_mv_default(z: i32, c_out: f64, c_in: f64) -> f64 {
    nernst_mv(z, c_out, c_in, T_DEFAULT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn potassium_reversal_is_negative() {
        // KNP-EMI defaults: [K]e = 4 mM, [K]i = 125 mM.
        let e_k = nernst_mv_default(K.z, 4.0, 125.0);
        assert!((e_k - (-88.9)).abs() < 0.5, "E_K = {e_k}");
    }

    #[test]
    fn equal_concentrations_give_zero() {
        assert_eq!(nernst_mv_default(CA.z, 2.0, 2.0), 0.0);
    }

    #[test]
    fn valence_scales_potential() {
        let e1 = nernst_mv_default(1, 10.0, 1.0);
        let e2 = nernst_mv_default(2, 10.0, 1.0);
        assert!((e1 - 2.0 * e2).abs() < 1e-12);
    }
}
