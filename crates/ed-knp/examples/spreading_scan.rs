//! Can one driven neuron start spreading depression? Scans the reduced-neuron tissue of
//! `ed_knp::spreading` over the persistent Na⁺ conductance, the number of driven cells, pump
//! strength, glial buffering and density, and prints a Markdown table.
//!
//! `cargo run --release -p ed-knp --example spreading_scan` (several minutes; runs in
//! parallel on all cores). `SD_MS` sets the simulated time per run (default 12000 ms).
use ed_core::Simulation;
use ed_knp::spreading::{SdConfig, SdTissue};
use ed_knp::tissue::K;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

#[derive(Clone)]
struct Case {
    g_nap: f64,
    driven: usize,
    pump: f64,
    buffering: f64,
    neurons: usize,
}

const BASE: Case = Case {
    g_nap: 0.25,
    driven: 1,
    pump: 1.0,
    buffering: 0.0,
    neurons: 10_000,
};
/// Drive on each driven soma, nS (enough for depolarisation block).
const DRIVE_NS: f64 = 20.0;

struct Outcome {
    /// Depolarised fraction at the end, and the first times it passed 10 % and 50 %.
    end: f64,
    t10: Option<f64>,
    t50: Option<f64>,
    k_max: f64,
}

fn run(c: &Case, ms: f64) -> Outcome {
    let mut cfg = SdConfig::default();
    cfg.population.g_nap = c.g_nap;
    cfg.population.pump_scale = c.pump;
    cfg.buffering = c.buffering;
    cfg.neurons = c.neurons;
    let size = cfg.knp.size;
    let mut t = SdTissue::new(&cfg);
    t.set_drive([size[0] / 2.0, size[1] / 2.0, 50.0], c.driven, DRIVE_NS);
    let (mut t10, mut t50, mut k_max) = (None, None, 0.0f64);
    let steps = (ms / cfg.knp.dt).round() as usize;
    for _ in 0..steps {
        t.step(1);
        let f = t.depolarised_fraction();
        if f >= 0.1 && t10.is_none() {
            t10 = Some(t.time_ms());
        }
        if f >= 0.5 && t50.is_none() {
            t50 = Some(t.time_ms());
        }
        k_max = t.knp.concentration(K).iter().fold(k_max, |a, &b| a.max(b));
    }
    Outcome {
        end: t.depolarised_fraction(),
        t10,
        t50,
        k_max,
    }
}

fn main() {
    let ms: f64 = std::env::var("SD_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12_000.0);
    let mut cases = Vec::new();
    for g_nap in [0.0, 0.1, 0.15, 0.2, 0.25, 0.3] {
        for driven in [1, 10, 100] {
            cases.push(Case {
                g_nap,
                driven,
                ..BASE
            });
        }
    }
    for driven in [1000] {
        for g_nap in [0.0, 0.1] {
            cases.push(Case {
                g_nap,
                driven,
                ..BASE
            });
        }
    }
    for pump in [0.8, 1.2] {
        cases.push(Case { pump, ..BASE });
    }
    for buffering in [5e-4, 2e-3] {
        cases.push(Case { buffering, ..BASE });
    }
    for neurons in [5000, 20_000] {
        cases.push(Case { neurons, ..BASE });
    }
    cases.push(Case { driven: 0, ..BASE });

    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<Outcome>>> = Mutex::new((0..cases.len()).map(|_| None).collect());
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(c) = cases.get(i) else { break };
                let out = run(c, ms);
                eprintln!("case {i} done");
                results.lock().unwrap()[i] = Some(out);
            });
        }
    });

    // Front speed from the radius of a disc covering the depolarised fraction of the slab.
    let area = SdConfig::default().knp.size;
    let radius = |f: f64| (f * area[0] * area[1] / std::f64::consts::PI).sqrt();
    println!("| g_NaP (mS/cm²) | driven | pump | buffering (1/ms) | neurons | depolarised at end | t to 50 % (s) | front speed (mm/min) | peak [K⁺]e (mM) |");
    println!("|---|---|---|---|---|---|---|---|---|");
    let fmt_t = |t: Option<f64>| t.map_or("–".into(), |t| format!("{:.1}", t / 1e3));
    for (c, r) in cases.iter().zip(results.into_inner().unwrap()) {
        let r = r.unwrap();
        let speed = match (r.t10, r.t50) {
            // µm/ms → mm/min.
            (Some(a), Some(b)) if b > a => {
                format!("{:.1}", (radius(0.5) - radius(0.1)) / (b - a) * 60.0)
            }
            _ => "–".into(),
        };
        println!(
            "| {} | {} | {} | {} | {} | {:.0} % | {} | {} | {:.1} |",
            c.g_nap,
            c.driven,
            c.pump,
            c.buffering,
            c.neurons,
            100.0 * r.end,
            fmt_t(r.t50),
            speed,
            r.k_max
        );
    }
}
