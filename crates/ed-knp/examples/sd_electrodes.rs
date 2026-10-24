//! Virtual electrodes in the spreading-depression tissue of `ed_knp::spreading`, measuring what
//! Enger et al. (2015, Cereb Cortex 25:4469) measured in mouse visual cortex: how long the
//! extracellular K⁺ rise leads the depolarisation, [K⁺]e when the depolarisation arrives, the
//! wave speed, the extracellular potential shift and how long the depolarisation lasts, with
//! and without astrocytes ([`ed_knp::glia`]). Prints a Markdown table.
//!
//! `cargo run --release -p ed-knp --example sd_electrodes` (a few minutes; runs in parallel
//! on all cores). `SD_MS` sets the simulated time per run (default 40000 ms).
use ed_core::Simulation;
use ed_knp::glia::GliaConfig;
use ed_knp::spreading::{SdConfig, SdTissue, DEPOLARISED_MV};
use ed_knp::tissue::K;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

struct Case {
    g_nap: f64,
    driven: usize,
    buffering: f64,
    /// Astrocytic membrane area per tissue volume (µm⁻¹) and gap-junction coupling.
    glia: Option<(f64, f64)>,
}

/// Drive on each driven soma, nS (as in `spreading_scan`).
const DRIVE_NS: f64 = 20.0;
/// Electrode distances from the drive along x and their offset in y, µm. With the default
/// 25 µm grid these, and the depths below, put each electrode at the centre of a grid cell, so
/// the K⁺ it reads is centred on the neurons it is compared with.
const R: [f64; 2] = [212.5, 412.5];
const Y_OFF: f64 = 12.5;
/// Depths of the soma and dendrite layers, µm.
const Z_SOMA: f64 = 62.5;
const Z_DEND: f64 = 162.5;
/// Neurons whose soma lies within this horizontal distance of an electrode report to it, µm.
const NEAR: f64 = 30.0;
/// [K⁺]e rise that marks the K⁺ onset, mM (Enger et al.'s threshold).
const K_ONSET: f64 = 0.25;
/// Depolarised when at least half the nearby dendrites are above −40 mV, recovered when fewer
/// than a quarter are (the gap keeps cells that flicker across −40 mV from ending it early).
const DEP_ON: f64 = 0.5;
const DEP_OFF: f64 = 0.25;

#[derive(Default, Clone)]
struct Electrode {
    /// K⁺ onset, depolarisation and recovery times, ms.
    t_k: Option<f64>,
    t_dep: Option<f64>,
    t_rec: Option<f64>,
    /// [K⁺]e at `t_dep` and its peak, mM.
    k_dep: f64,
    k_max: f64,
    /// Most negative φ(dendrite layer) − φ(soma layer), mV.
    dphi_min: f64,
}

fn run(c: &Case, ms: f64) -> Vec<Electrode> {
    let mut cfg = SdConfig::default();
    cfg.population.g_nap = c.g_nap;
    cfg.buffering = c.buffering;
    cfg.glia = c.glia.map(|(area_per_volume, coupling)| GliaConfig {
        area_per_volume,
        coupling,
        ..GliaConfig::default()
    });
    let size = cfg.knp.size;
    let mut t = SdTissue::new(&cfg);
    let centre = [size[0] / 2.0, size[1] / 2.0 + Y_OFF, Z_SOMA];
    t.set_drive(centre, c.driven, DRIVE_NS);
    let at = |t: &SdTissue, r: f64, z: f64| t.knp.cell_at([centre[0] + r, centre[1], z]).unwrap();
    let soma = R.map(|r| at(&t, r, Z_SOMA));
    let dend = R.map(|r| at(&t, r, Z_DEND));
    let near = R.map(|r| {
        (0..t.pop.len())
            .filter(|&i| {
                let p = t.positions()[2 * i];
                (p[0] - centre[0] - r).hypot(p[1] - centre[1]) < NEAR
            })
            .collect::<Vec<_>>()
    });
    let k0 = cfg.knp.baseline[K];
    let mut e = vec![Electrode::default(); R.len()];
    let steps = (ms / cfg.knp.dt).round() as usize;
    for _ in 0..steps {
        t.step(1);
        let now = t.time_ms();
        let (k, phi, v) = (t.knp.concentration(K), t.knp.potential(), t.pop.v());
        for (j, e) in e.iter_mut().enumerate() {
            let kj = k[soma[j]];
            e.k_max = e.k_max.max(kj);
            e.dphi_min = e.dphi_min.min(phi[dend[j]] - phi[soma[j]]);
            if e.t_k.is_none() && kj >= k0 + K_ONSET {
                e.t_k = Some(now);
            }
            let dep = near[j]
                .iter()
                .filter(|&&i| v[2 * i + 1] > DEPOLARISED_MV)
                .count();
            let f = dep as f64 / near[j].len() as f64;
            if e.t_dep.is_none() && f >= DEP_ON {
                e.t_dep = Some(now);
                e.k_dep = kj;
            } else if e.t_dep.is_some() && e.t_rec.is_none() && f < DEP_OFF {
                e.t_rec = Some(now);
            }
        }
    }
    e
}

fn main() {
    let ms: f64 = std::env::var("SD_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40_000.0);
    // The least excitable drive that still ignites at each g_NaP (see spreading_scan), with
    // and without the weak glial uptake that lets the tissue recover.
    let mut cases = Vec::new();
    for buffering in [0.0, 5e-4] {
        for (g_nap, driven) in [(0.25, 1), (0.2, 10), (0.15, 100)] {
            cases.push(Case {
                g_nap,
                driven,
                buffering,
                glia: None,
            });
        }
    }
    // Astrocytes: Halnes et al.'s membrane density only lets the most excitable tissue
    // ignite; an eighth of it, with and without gap junctions, against the same drive alone.
    for (g_nap, driven, glia) in [
        (0.3, 100, Some((8.0, 1.0))),
        (0.25, 400, Some((8.0, 1.0))),
        (0.2, 100, None),
        (0.2, 100, Some((1.0, 1.0))),
        (0.2, 100, Some((1.0, 0.01))),
    ] {
        cases.push(Case {
            g_nap,
            driven,
            buffering: 0.0,
            glia,
        });
    }

    let next = AtomicUsize::new(0);
    let results = Mutex::new(vec![Vec::new(); cases.len()]);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(c) = cases.get(i) else { break };
                let out = run(c, ms);
                eprintln!("case {i} done");
                results.lock().unwrap()[i] = out;
            });
        }
    });

    // Times at the far electrode, which sits in tissue the wave reached on its own. A negative
    // K⁺ lead means the neurons depolarised before [K⁺]e rose by 0.25 mM.
    let s = |t: f64| format!("{:.1}", t / 1e3);
    println!("| g_NaP (mS/cm²) | driven | buffering (1/ms) | glia (µm⁻¹, coupling) | speed (mm/min) | K⁺ lead (s) | [K⁺]e at depolarisation (mM) | peak [K⁺]e (mM) | depolarisation lasts (s) | laminar Δφ (mV) |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for (c, e) in cases.iter().zip(results.into_inner().unwrap()) {
        let (near, far) = (&e[0], &e[1]);
        let (speed, lead, k_dep, lasts) = match far.t_dep {
            None => ("–".into(), "–".into(), "–".into(), "–".into()),
            Some(td) => (
                // µm/ms → mm/min.
                near.t_dep.map_or("–".into(), |tn| {
                    format!("{:.1}", (R[1] - R[0]) / (td - tn) * 60.0)
                }),
                far.t_k
                    .map_or("–".into(), |tk| format!("{:.2}", (td - tk) / 1e3)),
                format!("{:.1}", far.k_dep),
                far.t_rec
                    .map_or(format!("> {}", s(ms - td)), |tr| s(tr - td)),
            ),
        };
        let glia = c.glia.map_or("–".into(), |(a, k)| format!("{a}, {k}"));
        println!(
            "| {} | {} | {} | {glia} | {speed} | {lead} | {k_dep} | {:.1} | {lasts} | {:.1} |",
            c.g_nap, c.driven, c.buffering, far.k_max, far.dphi_min
        );
    }
}
