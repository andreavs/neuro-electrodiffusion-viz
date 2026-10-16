//! Browser bindings. Each scale gets a `#[wasm_bindgen]` wrapper struct here
//! that owns the solver and exposes zero-copy views of its field buffers.

use wasm_bindgen::prelude::*;

pub mod emi;
pub mod knp;
pub mod particles;

/// Crate version, used by the front end to confirm the WASM module loaded.
#[wasm_bindgen]
pub fn core_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Nernst potential (mV) at 300 K.
#[wasm_bindgen]
pub fn nernst_potential_mv(z: i32, c_out: f64, c_in: f64) -> f64 {
    ed_core::ions::nernst_mv_default(z, c_out, c_in)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_semver() {
        assert_eq!(core_version().split('.').count(), 3);
    }
}
