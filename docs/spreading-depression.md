# Can one neuron start spreading depression through the extracellular space?

Question (Andreas): in the KNP point-source model, with ~10,000 neurons of simplified geometry
and no synapses, can driving **one** neuron start a runaway, spreading-depression-like wave
carried only by the extracellular ion concentrations?

Short answer: **yes, but only when the tissue is already close to its own ignition threshold.**
In this model that closeness is set mainly by a persistent Na⁺ current. Without one, not even
1,000 driven neurons start a wave. With g_NaP = 0.25 mS/cm², a single neuron held in
depolarisation block ignites a wave that crosses the 1 mm slab at about 6 mm/min (the speed of
real spreading depression is 2–9 mm/min) and leaves [K⁺]e at 30–40 mM.

## Model (`ed_knp::population`, `ed_knp::spreading`)

- **Tissue:** 1 × 1 × 0.2 mm slab, KNP electrodiffusion (Na⁺, K⁺, X⁻, Ca²⁺) on 40 × 40 × 8
  cells of 25 µm, α = 0.2, λ = 1.6, closed boundaries, backward Euler with 4 ms steps.
- **Neurons:** 10,000 cells (5·10⁴ per mm³, about half the neuron density of cortex), placed at
  random with somata in the lower half of the slab. Each cell has a 1500 µm² soma and a
  6000 µm² lumped dendrite whose centre sits 100 µm above it.
  - Soma: Hodgkin-Huxley channels. Both compartments: K⁺, Na⁺ and Cl⁻ leaks, a Na⁺/K⁺ pump
    (Kager et al. 2000) and an optional persistent Na⁺ current, `g_NaP / (1 + e^{−(V+50)/5})`.
  - The Na⁺ leak and pump capacity are calibrated so that the baseline is an exact ionic
    steady state.
- **Coupling, both ways:** every 4 ms each compartment reads the Na⁺, K⁺ and X⁻ of its grid
  cell. Its reversal potentials follow, its own intracellular Na⁺, K⁺ and Cl⁻ change with its
  currents, and its currents are the KNP point sources. The only interaction between cells is
  through the extracellular ions.
- **Neuron time stepping:** each neuron takes its own adaptive steps, from 0.025 ms while
  spiking to 4 ms at rest or in block. Backward Euler is used for V and Rush-Larsen for the
  gates. Spike counts agree with a fixed 0.025 ms step to within 10 %, and the wave evolution
  is the same at 1, 2 and 4 ms KNP steps.
- **Drive:** a 20 nS Na⁺ conductance on the soma(s) nearest the centre. This holds a cell in
  depolarisation block, where it loses K⁺ steadily.
- **"Depolarised":** dendrite above −40 mV.
- **Glial buffering (optional):** KCl is taken up wherever [K⁺]e exceeds baseline, at a rate
  proportional to the excess (exact relaxation over each step, never below baseline).

## The threshold of a single cell

With the extracellular concentrations held fixed, an isolated cell takes up K⁺ (the pump wins)
below a threshold [K⁺]e and releases it above. The threshold is the critical number:

| g_NaP (mS/cm²) | [K⁺]e threshold | K⁺ release above it |
|---|---|---|
| 0 | ≈ 9 mM | ≈ 0.5 nA per cell |
| 0.1 | ≈ 6 mM | ≈ 1 nA |
| 0.2 | ≈ 4 mM | ≈ 2 nA |

Baseline [K⁺]e is 3 mM. One driven neuron raises [K⁺]e in its own 25 µm grid cell by only
1–3 mM, and less further out. So it can only recruit its neighbours when the threshold sits
within about 1 mM of baseline.

## Scan

Output of `cargo run --release -p ed-knp --example spreading_scan` (12 s per run; pump 1 means
the calibrated pump):

| g_NaP (mS/cm²) | driven | pump | buffering (1/ms) | neurons | depolarised at end | t to 50 % (s) | front speed (mm/min) | peak [K⁺]e (mM) |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 1 | 0 | 10000 | 0 % | – | – | 4.2 |
| 0 | 10 | 1 | 0 | 10000 | 0 % | – | – | 8.7 |
| 0 | 100 | 1 | 0 | 10000 | 0 % | – | – | 17.1 |
| 0 | 1000 | 1 | 0 | 10000 | 11 % | – | – | 29.9 |
| 0.1 | 1 | 1 | 0 | 10000 | 0 % | – | – | 4.6 |
| 0.1 | 10 | 1 | 0 | 10000 | 0 % | – | – | 10.1 |
| 0.1 | 100 | 1 | 0 | 10000 | 18 % | – | – | 23.5 |
| 0.1 | 1000 | 1 | 0 | 10000 | 53 % | 11.4 | 1.2 | 36.2 |
| 0.15 | 1 | 1 | 0 | 10000 | 0 % | – | – | 4.7 |
| 0.15 | 10 | 1 | 0 | 10000 | 21 % | – | – | 25.7 |
| 0.15 | 100 | 1 | 0 | 10000 | 61 % | 10.7 | 2.1 | 29.1 |
| 0.2 | 1 | 1 | 0 | 10000 | 0 % | – | – | 6.0 |
| 0.2 | 10 | 1 | 0 | 10000 | 100 % | 7.2 | 3.7 | 32.3 |
| 0.2 | 100 | 1 | 0 | 10000 | 100 % | 6.0 | 3.7 | 33.5 |
| 0.25 | 1 | 1 | 0 | 10000 | 100 % | 4.7 | 6.0 | 37.2 |
| 0.25 | 10 | 1 | 0 | 10000 | 100 % | 4.1 | 6.0 | 37.3 |
| 0.25 | 100 | 1 | 0 | 10000 | 100 % | 3.6 | 6.0 | 37.5 |
| 0.3 | 1 | 1 | 0 | 10000 | 100 % | 2.7 | 9.8 | 39.8 |
| 0.3 | 10 | 1 | 0 | 10000 | 100 % | 2.5 | 9.8 | 39.9 |
| 0.3 | 100 | 1 | 0 | 10000 | 100 % | 2.2 | 9.8 | 39.9 |
| 0.25 | 0 | 1 | 0 | 10000 | 0 % | – | – | 3.0 |
| 0.25 | 1 | 0.8 | 0 | 10000 | 100 % | 0.0 | – | 40.6 |
| 0.25 | 1 | 1.2 | 0 | 10000 | 0 % | – | – | 5.7 |
| 0.25 | 1 | 1 | 0.0005 | 10000 | 100 % | 5.1 | 5.7 | 20.7 |
| 0.25 | 1 | 1 | 0.002 | 10000 | 0 % | – | – | 5.5 |
| 0.25 | 1 | 1 | 0 | 5000 | 100 % | 6.8 | 4.5 | 23.6 |
| 0.25 | 1 | 1 | 0 | 20000 | 100 % | 3.5 | 7.9 | 51.0 |

Front speed is the growth rate of the radius of a disc with the depolarised area, between 10 %
and 50 % of the slab.

## Findings

1. **One neuron is enough only near threshold.** With g_NaP ≥ 0.25 mS/cm² a single driven cell
   ignites a wave. At 0.2 it takes about 10 cells, at 0.15 about 100 (slow, partial), and at
   0.1 about 1000. Without a persistent Na⁺ current, nothing propagates, even when 1000 cells
   (10 % of the slab) are driven and [K⁺]e reaches 30 mM among them. HH cells plus leaks and a
   pump release too little K⁺ once depolarised for their neighbours' pumps not to win.
2. **The undriven tissue is stable.** With g_NaP = 0.25 and no drive nothing happens: the wave
   is triggered, not spontaneous. Weakening the pump to 0.8× is different: the resting state
   itself is then unstable, and the whole slab depolarises at once without any trigger.
3. **The pump and glia set the margin.** A 20 % stronger pump, or glial K⁺ uptake at
   2/s, stops the wave from one driven cell. Weaker uptake (0.5/s) still lets it spread
   at nearly the same speed, but caps [K⁺]e near 20 mM.
4. **Density matters for speed and amplitude, not for ignition here.** Half the density
   (5000 cells) still ignites, at 4.5 mm/min and 24 mM. Double the density (20,000 cells) gives
   7.9 mm/min and 51 mM.
5. **Speeds are realistic.** 4–10 mm/min across the cases that ignite, and the wave leaves
   [K⁺]e at 20–50 mM. Both are in the range reported for spreading depression, from a model with
   no synapses, no glutamate and no swelling.

## Limitations

- Grid cells are 25 µm, so a cell sees the [K⁺]e averaged over its grid cell. The real
  concentration next to a driven membrane is higher, which would favour ignition by one cell.
- No volume changes (α fixed at 0.2; in real SD it drops to about 0.05, which concentrates K⁺),
  no glutamate/NMDA and no Ca²⁺ currents. All of these would lower the threshold.
- The persistent Na⁺ current is a generic one standing in for all sustained inward currents. The
  values that make the tissue excitable (0.2–0.3 mS/cm²) are well above typical somatic NaP
  densities, so read g_NaP as "how close to threshold the tissue is" rather than as a measured
  conductance.
- The slab is 200 µm thick with closed boundaries, a stand-in for one layer of a larger tissue.
