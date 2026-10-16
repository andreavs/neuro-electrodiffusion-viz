//! Small, fast, seedable random number generation with no external dependencies,
//! so the crate compiles to `wasm32-unknown-unknown` without `getrandom`.
//!
//! - [`Xoshiro256pp`]: xoshiro256++ (Blackman & Vigna), seeded through SplitMix64.
//! - Standard normals via Doornik's ZIGNOR ziggurat (128 layers), which needs one
//!   32-bit draw per sample in ~99% of cases. Of the 32 bits, 7 pick the layer and 25 give
//!   the signed abscissa, so samples lie on a grid of spacing ≤ 2.2·10⁻⁷, far below what
//!   matters for Brownian steps.
//! - [`Xoshiro256pp::fill_normal`] fills buffers from four interleaved xoshiro streams so
//!   the CPU can overlap (or vectorize) the generator updates; this is the hot path of the
//!   particle stepper.

/// SplitMix64 step, used to expand a 64-bit seed into the xoshiro state.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

const ZIG_LAYERS: usize = 128;
const ZIG_R: f64 = 3.442_619_855_899;
const ZIG_V: f64 = 9.912_563_035_262_17e-3;
const TWO_POW_M53: f64 = 1.0 / (1u64 << 53) as f64;

/// Scale of the signed 25-bit abscissa: u = j / 2²⁴ in [-1, 1).
const TWO_POW_24: f64 = 16_777_216.0;

/// Ziggurat tables: layer edges `x[i]`, the ratio `x[i + 1] / x[i]` scaled to 2²⁴ for an
/// integer fast-path comparison, and `x[i]` / 2²⁴ to scale the accepted integer.
#[derive(Clone)]
struct ZigTables {
    x: [f64; ZIG_LAYERS + 1],
    k: [u32; ZIG_LAYERS],
    w: [f64; ZIG_LAYERS],
}

impl ZigTables {
    fn new() -> Self {
        let mut x = [0.0; ZIG_LAYERS + 1];
        let mut f = (-0.5 * ZIG_R * ZIG_R).exp();
        x[0] = ZIG_V / f;
        x[1] = ZIG_R;
        for i in 2..ZIG_LAYERS {
            x[i] = (-2.0 * (ZIG_V / x[i - 1] + f).ln()).sqrt();
            f = (-0.5 * x[i] * x[i]).exp();
        }
        x[ZIG_LAYERS] = 0.0;
        let mut k = [0u32; ZIG_LAYERS];
        let mut w = [0.0; ZIG_LAYERS];
        for i in 0..ZIG_LAYERS {
            k[i] = (x[i + 1] / x[i] * TWO_POW_24) as u32;
            w[i] = x[i] / TWO_POW_24;
        }
        Self { x, k, w }
    }
}

/// xoshiro256++ generator with a ziggurat normal sampler.
#[derive(Clone)]
pub struct Xoshiro256pp {
    s: [u64; 4],
    /// Four extra streams for batched generation, stored word-major: `lanes[w][lane]`.
    lanes: [[u64; 4]; 4],
    zig: ZigTables,
}

#[inline(always)]
fn rotl4(x: [u64; 4], k: u32) -> [u64; 4] {
    [
        x[0].rotate_left(k),
        x[1].rotate_left(k),
        x[2].rotate_left(k),
        x[3].rotate_left(k),
    ]
}

#[inline(always)]
fn map2(a: [u64; 4], b: [u64; 4], f: impl Fn(u64, u64) -> u64) -> [u64; 4] {
    [f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2]), f(a[3], b[3])]
}

impl Xoshiro256pp {
    /// Creates a generator from a 64-bit seed. Different seeds give independent streams.
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        let mut lanes = [[0u64; 4]; 4];
        for lane in 0..4 {
            for word in lanes.iter_mut() {
                word[lane] = splitmix64(&mut sm);
            }
        }
        Self {
            s,
            lanes,
            zig: ZigTables::new(),
        }
    }

    /// One step of the four interleaved streams.
    #[inline(always)]
    fn next_u64x4(&mut self) -> [u64; 4] {
        let [s0, s1, s2, s3] = &mut self.lanes;
        let result = map2(
            rotl4(map2(*s0, *s3, u64::wrapping_add), 23),
            *s0,
            u64::wrapping_add,
        );
        let t = s1.map(|v| v << 17);
        *s2 = map2(*s2, *s0, |a, b| a ^ b);
        *s3 = map2(*s3, *s1, |a, b| a ^ b);
        *s1 = map2(*s1, *s2, |a, b| a ^ b);
        *s0 = map2(*s0, *s3, |a, b| a ^ b);
        *s2 = map2(*s2, t, |a, b| a ^ b);
        *s3 = rotl4(*s3, 45);
        result
    }

    /// Fills `out` with independent standard normals.
    ///
    /// Works in blocks: raw bits for the whole block first (four interleaved streams),
    /// then a branch-free rectangle test for every sample, then the ~1% of samples that
    /// fall outside the rectangles are redone with the full ziggurat.
    pub fn fill_normal(&mut self, out: &mut [f64]) {
        const BLOCK: usize = 256;
        let mut bits = [0u32; BLOCK];
        for block in out.chunks_mut(BLOCK) {
            let len = block.len();
            for w8 in bits[..len.next_multiple_of(8)].chunks_exact_mut(8) {
                let w = self.next_u64x4();
                for l in 0..4 {
                    w8[l] = w[l] as u32;
                    w8[l + 4] = (w[l] >> 32) as u32;
                }
            }
            // Indices of samples outside the rectangles, written without branching.
            let mut rejected = [0u16; BLOCK + 1];
            let mut n_rej = 0;
            for (k, (o, &b)) in block.iter_mut().zip(&bits[..len]).enumerate() {
                let j = (b as i32) >> 7;
                let i = (b & (ZIG_LAYERS as u32 - 1)) as usize;
                *o = j as f64 * self.zig.w[i];
                rejected[n_rej] = k as u16;
                n_rej += (j.unsigned_abs() >= self.zig.k[i]) as usize;
            }
            for &k in &rejected[..n_rej] {
                let k = k as usize;
                block[k] = self.zig_slow(bits[k]).unwrap_or_else(|| self.normal());
            }
        }
    }

    /// Next raw 64-bit output.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1) with 53 bits of precision.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * TWO_POW_M53
    }

    /// Standard normal N(0, 1).
    #[inline]
    pub fn normal(&mut self) -> f64 {
        loop {
            let bits = (self.next_u64() >> 32) as u32;
            if let Some(x) = self.zig_fast(bits) {
                return x;
            }
            if let Some(x) = self.zig_slow(bits) {
                return x;
            }
        }
    }

    /// Ziggurat rectangle test. The low 7 bits pick the layer; the upper 25 bits as a signed
    /// integer j give u = j / 2²⁴ in [-1, 1).
    #[inline(always)]
    fn zig_fast(&self, bits: u32) -> Option<f64> {
        let j = (bits as i32) >> 7;
        let i = (bits & (ZIG_LAYERS as u32 - 1)) as usize;
        (j.unsigned_abs() < self.zig.k[i]).then(|| j as f64 * self.zig.w[i])
    }

    /// Ziggurat tail or wedge for bits that failed [`Self::zig_fast`]; `None` means the
    /// draw is rejected and the caller starts over with fresh bits.
    #[cold]
    fn zig_slow(&mut self, bits: u32) -> Option<f64> {
        let j = (bits as i32) >> 7;
        let i = (bits & (ZIG_LAYERS as u32 - 1)) as usize;
        if i == 0 {
            return Some(self.normal_tail(j < 0));
        }
        let x = j as f64 * self.zig.w[i];
        let xi = self.zig.x[i];
        let xi1 = self.zig.x[i + 1];
        let f0 = (-0.5 * (xi * xi - x * x)).exp();
        let f1 = (-0.5 * (xi1 * xi1 - x * x)).exp();
        (f1 + self.uniform() * (f0 - f1) < 1.0).then_some(x)
    }

    #[cold]
    fn normal_tail(&mut self, negative: bool) -> f64 {
        loop {
            // 1 - U lies in (0, 1], so the logarithms are finite.
            let x = (1.0 - self.uniform()).ln() / ZIG_R;
            let y = (1.0 - self.uniform()).ln();
            if -2.0 * y >= x * x {
                return if negative { x - ZIG_R } else { ZIG_R - x };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream_and_different_seeds_differ() {
        let mut a = Xoshiro256pp::new(42);
        let mut b = Xoshiro256pp::new(42);
        let mut c = Xoshiro256pp::new(43);
        for _ in 0..100 {
            let va = a.next_u64();
            assert_eq!(va, b.next_u64());
            assert_ne!(va, c.next_u64());
        }
    }

    #[test]
    fn ziggurat_layers_have_equal_area() {
        let z = ZigTables::new();
        // Top layer: x[C-1] * (1 - f(x[C-1])) should equal V up to the table constants' precision.
        let x = z.x[ZIG_LAYERS - 1];
        let top = x * (1.0 - (-0.5 * x * x).exp());
        assert!((top - ZIG_V).abs() / ZIG_V < 1e-6, "top layer area {top}");
        assert!(z.x.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn uniform_moments() {
        let mut rng = Xoshiro256pp::new(1);
        let n = 1_000_000;
        let (mut s1, mut s2) = (0.0, 0.0);
        for _ in 0..n {
            let u = rng.uniform();
            assert!((0.0..1.0).contains(&u));
            s1 += u;
            s2 += u * u;
        }
        let mean = s1 / n as f64;
        let var = s2 / n as f64 - mean * mean;
        assert!((mean - 0.5).abs() < 2e-3);
        assert!((var - 1.0 / 12.0).abs() < 1e-3);
    }

    #[test]
    fn normal_moments_and_tails() {
        check_normal_moments(|rng, buf| {
            for x in buf.iter_mut() {
                *x = rng.normal();
            }
        });
    }

    #[test]
    fn batched_normal_moments_and_tails() {
        check_normal_moments(|rng, buf| rng.fill_normal(buf));
    }

    #[test]
    fn batched_lanes_are_uncorrelated() {
        let mut rng = Xoshiro256pp::new(5);
        let mut buf = vec![0.0; 4_000_000];
        rng.fill_normal(&mut buf);
        let n = buf.len() / 4;
        for lag in 1..=4 {
            let c: f64 = buf.windows(lag + 1).map(|w| w[0] * w[lag]).sum::<f64>() / (4 * n) as f64;
            assert!(c.abs() < 3e-3, "lag {lag} correlation {c}");
        }
    }

    fn check_normal_moments(fill: impl Fn(&mut Xoshiro256pp, &mut [f64])) {
        let mut rng = Xoshiro256pp::new(7);
        let n = 2_000_000;
        let mut buf = vec![0.0; n];
        fill(&mut rng, &mut buf);
        let (mut s1, mut s2, mut s3, mut s4) = (0.0, 0.0, 0.0, 0.0);
        let (mut beyond1, mut beyond3) = (0usize, 0usize);
        for &x in &buf {
            s1 += x;
            s2 += x * x;
            s3 += x * x * x;
            s4 += x * x * x * x;
            if x.abs() > 1.0 {
                beyond1 += 1;
            }
            if x.abs() > 3.0 {
                beyond3 += 1;
            }
        }
        let nf = n as f64;
        assert!((s1 / nf).abs() < 3e-3, "mean {}", s1 / nf);
        assert!((s2 / nf - 1.0).abs() < 5e-3, "var {}", s2 / nf);
        assert!((s3 / nf).abs() < 1e-2, "skew {}", s3 / nf);
        assert!((s4 / nf - 3.0).abs() < 3e-2, "kurtosis {}", s4 / nf);
        // P(|X| > 1) = 0.317311, P(|X| > 3) = 0.0026998.
        let p1 = beyond1 as f64 / nf;
        let p3 = beyond3 as f64 / nf;
        assert!((p1 - 0.317_311).abs() < 2e-3, "P(|X|>1) = {p1}");
        assert!((p3 - 0.002_699_8).abs() < 2e-4, "P(|X|>3) = {p3}");
    }
}
