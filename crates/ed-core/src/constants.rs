//! Physical constants, with values as used in the thesis (Solbrå 2019).

/// Gas constant, J/(K mol).
pub const R: f64 = 8.314;
/// Faraday's constant, C/mol.
pub const F: f64 = 9.648e4;
/// Default temperature, K.
pub const T_DEFAULT: f64 = 300.0;
/// Vacuum permittivity, F/m.
pub const EPS0: f64 = 8.854e-12;
/// Relative permittivity of the extracellular medium.
pub const EPS_R: f64 = 80.0;

/// ψ = RT/F in mV at temperature `t` (K).
pub fn psi_mv(t: f64) -> f64 {
    1e3 * R * t / F
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psi_at_300k_is_about_25_85_mv() {
        let psi = psi_mv(T_DEFAULT);
        assert!((psi - 25.852).abs() < 1e-2, "psi = {psi}");
    }
}
