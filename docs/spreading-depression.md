# Can one neuron start spreading depression through the extracellular space?

Question (Andreas): in the KNP point-source model, with ~10,000 neurons of simplified geometry
and no synapses, can driving **one** neuron start a runaway, spreading-depression-like wave
carried only by the extracellular ion concentrations?

Short answer: **yes, but only when the tissue is already close to its own ignition threshold.**
In this model that closeness is set mainly by a persistent Na⁺ current. Without one, not even
1,000 driven neurons start a wave. With g_NaP = 0.25 mS/cm², a single neuron held in
depolarisation block ignites a wave that crosses the 1 mm slab at about 6 mm/min (the speed of
real spreading depression is 2–9 mm/min), with peak [K⁺]e near 37 mM.

## Model (`ed_knp::population`, `ed_knp::spreading`)

- **Tissue:** 1 × 1 × 0.2 mm slab, KNP electrodiffusion (Na⁺, K⁺, X⁻, Ca²⁺) on 40 × 40 × 8
  cells of 25 µm, α = 0.2, λ = 1.6, closed boundaries, backward Euler with 4 ms steps.
- **Neurons:** 10,000 cells (5·10⁴ per mm³, about half the neuron density of rodent cortex), placed at
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
  gates. Spike counts agree with a fixed 0.025 ms step to within 10 % (a unit test). In a
  one-off check during development, the wave evolved the same at 1, 2 and 4 ms KNP steps.
- **Drive:** a 20 nS Na⁺ conductance on the soma(s) nearest the centre. This holds a cell in
  depolarisation block, where it loses K⁺ steadily.
- **"Depolarised":** dendrite above −40 mV.
- **Glial buffering (optional):** KCl is taken up wherever [K⁺]e exceeds baseline, at a rate
  proportional to the excess (exact relaxation over each step, never below baseline).
- **Astrocytes (optional, `SdConfig::glia`):** an astrocytic syncytium in every grid cell
  (`ed_knp::glia`, after Halnes et al. 2013): a second KNP domain for the astrocytes' interior
  (α = 0.4, λ = 3.2) behind a membrane with Kir4.1, Na⁺ and Cl⁻ leaks and a pump, its
  potential from a cable equation. They take up K⁺ where it is high and release it through
  the syncytium where it is low. See [Astrocytes](#astrocytes-spatial-buffering).

## The threshold of a single cell

With the extracellular concentrations held fixed, an isolated cell takes up K⁺ (the pump wins)
below a threshold [K⁺]e and releases it above. These values come from one-off runs of a single
cell during development, not from the scan:

| g_NaP (mS/cm²) | [K⁺]e threshold | K⁺ release above it |
|---|---|---|
| 0 | ≈ 9 mM | ≈ 0.5 nA per cell |
| 0.1 | ≈ 6 mM | ≈ 1 nA |
| 0.2 | ≈ 4 mM | ≈ 2 nA |

Baseline [K⁺]e is 3 mM. One driven neuron raises [K⁺]e in its own 25 µm grid cell by only
1–3 mM, and less further out. So it can only recruit its neighbours when the threshold sits
within about 1 mM of baseline. Exceeding the threshold locally is not enough on its own,
though: a recruited cell must also release K⁺ faster than diffusion carries it away from a
small hot spot (see finding 1).

## Scan

Output of `cargo run --release -p ed-knp --example spreading_scan` (12 s per run; pump 1 means
the calibrated pump; "t to 50 %" of 0.0 means under 50 ms):

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
| 0.3 | 0 | 1 | 0 | 10000 | 0 % | – | – | 3.0 |
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
   (10 % of the slab) are driven and [K⁺]e reaches 30 mM among them. With 100 driven cells
   [K⁺]e peaks at 17 mM, well above the 9 mM single-cell threshold, yet no neighbour is
   recruited (and even the driven cells' dendrites stay below −40 mV). Without NaP a
   depolarised cell releases only about 0.5 nA of K⁺, so diffusion and the pumps of the
   surrounding cells clear the excess faster than the few recruited cells can add to it.
2. **The undriven tissue is stable, but only just.** With g_NaP = 0.25 or 0.3 and no drive
   nothing happens: the wave is triggered, not spontaneous. Weakening the pump to 0.8× removes
   the resting state altogether. All cells depolarise within 50 ms, before [K⁺]e has risen
   (3.3 mM at 20 ms), so this is each cell losing its own resting potential, not an ionic
   wave. So at g_NaP = 0.25 the resting state is within 20 % of pump capacity of vanishing,
   which is what "near threshold" means here.
3. **The pump and glia set the margin.** A 20 % stronger pump, or glial K⁺ uptake at
   2/s, stops the wave from one driven cell. Weaker uptake (0.5/s) still lets it spread
   at nearly the same speed, but caps [K⁺]e near 20 mM.
4. **Density matters for speed and amplitude, not for ignition here.** Half the density
   (5000 cells) still ignites, at 4.5 mm/min and 24 mM. Double the density (20,000 cells) gives
   7.9 mm/min and 51 mM. At that density the cells fill about 70 % of the slab, which with
   α = 0.2 leaves little room for anything else, so the 51 mM partly reflects the packing.
5. **Speeds are in the physiological range.** Waves that ignite travel at 1–10 mm/min
   (4–10 where one or a few cells suffice), with peak [K⁺]e of 20–50 mM. Both are close to
   what is reported for spreading depression (2–9 mm/min, tens of mM), from a model with no
   synapses, no glutamate and no swelling.

## Limitations

- Grid cells are 25 µm, so a cell sees the [K⁺]e averaged over its grid cell. The real
  concentration next to a driven membrane is higher, which would favour ignition by one cell.
- No volume changes (α fixed at 0.2; in real SD it drops to about 0.05, which concentrates K⁺),
  no glutamate/NMDA and no Ca²⁺ currents. All of these would lower the threshold.
- The persistent Na⁺ current is a generic one standing in for all sustained inward currents. The
  values that make the tissue excitable (0.2–0.3 mS/cm², 0.2 % of ḡ_Na) are at the upper end
  of what models use, and here the current is on the large dendrite as well as the soma. So
  read g_NaP as "how close to threshold the tissue is" rather than as a measured conductance.
- The slab is 200 µm thick with closed boundaries, a stand-in for one layer of a larger tissue.

## Comparison with Enger et al. (2015)

Enger, Tang, Vindedal, …, Looger and Nagelhus, *Dynamics of ionic shifts in cortical spreading
depression*, Cerebral Cortex 25:4469–4476 (2015),
[PMC4816793](https://pmc.ncbi.nlm.nih.gov/articles/PMC4816793/). This is an experimental paper,
not a model. In anaesthetised mice they triggered CSD with 1 M KCl on visual cortex and
recorded, in layer 2/3, [K⁺]e with ion-sensitive microelectrodes, the DC potential, and
two-photon Ca²⁺ (GCaMP6f in neurons or astrocytes) and glutamate (iGluSnFR). Their main
numbers:

| Measure | Enger et al. |
|---|---|
| Wave speed (neuronal Ca²⁺) | 56.7 ± 3.3 µm/s ≈ 3.4 mm/min (glutamate 3.1, astrocytic Ca²⁺ 3.3) |
| [K⁺]e rise (0.25 mM onset) ahead of the DC shift | 5.9 ± 0.4 s |
| [K⁺]e when the neuronal Ca²⁺ front arrives | 7.3 ± 0.6 mM |
| Order of events | K⁺, DC shift, neuronal Ca²⁺ (+1.1 s), glutamate (+1.9 s), astrocytic Ca²⁺ (+4.0 s) |
| Negative DC shift lasts | 66.4 ± 3.8 s |

Their conclusion is that interstitial K⁺, not glutamate or astrocytic Ca²⁺ waves, leads the
wave. That is the mechanism this model has by construction, so the question is how well its
numbers match.

`cargo run --release -p ed-knp --example sd_electrodes` places virtual electrodes about 210 and
410 µm from the drive. At each, a K⁺ electrode in the soma layer (the grid cell at z = 50–75 µm)
records the 0.25 mM onset, and the neurons with somata within 30 µm stand in for the imaged
neurons: the depolarisation arrives when half their dendrites are above −40 mV (a proxy for
the neuronal Ca²⁺ rise) and ends when fewer than a quarter are. The table gives values at the
far electrode, 40 s runs, without astrocytes (the example's runs with astrocytes are in
[Astrocytes](#astrocytes-spatial-buffering)):

| g_NaP (mS/cm²) | driven | buffering (1/ms) | speed (mm/min) | K⁺ lead (s) | [K⁺]e at depolarisation (mM) | peak [K⁺]e (mM) | depolarisation lasts (s) | laminar Δφ (mV) |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 1 | 0 | 5.8 | 0.05 | 3.4 | 29.3 | > 35.3 | −3.9 |
| 0.2 | 10 | 0 | 3.5 | 0.32 | 4.0 | 27.4 | > 32.5 | −3.0 |
| 0.15 | 100 | 0 | 2.0 | 1.36 | 5.6 | 25.1 | > 29.0 | −2.1 |
| 0.25 | 1 | 0.0005 | 5.4 | 0.06 | 3.4 | 11.8 | 18.2 | −4.0 |
| 0.2 | 10 | 0.0005 | 2.9 | 0.42 | 4.0 | 10.3 | 10.2 | −3.2 |
| 0.15 | 100 | 0.0005 | – | – | – | 3.0 | – | 0.0 |

Laminar Δφ is the most negative φ(dendrite layer, z = 150–175 µm) − φ(soma layer). It is a
reference-free potential, used because the slab is too small for a remote reference that the
wave never reaches. It is not the same quantity as a DC potential against a remote reference.

What matches and what does not:

1. **Speed matches.** At g_NaP = 0.2 the wave travels at 2.9–3.5 mm/min, as in the paper.
2. **The K⁺ precursor is far too short.** Here [K⁺]e starts rising at most about 1.4 s before
   the cells depolarise, and they depolarise at 3.4–5.6 mM. In the paper the K⁺ onset leads the
   DC shift by 5.9 s and the neuronal Ca²⁺ rise (which follows the DC shift by about 1.1 s) by
   about 7 s, and [K⁺]e is 7.3 mM when the neurons fire. Making the tissue less excitable
   raises the threshold and the lead (g_NaP 0.15: 5.6 mM, 1.4 s) but slows the wave and soon
   stops it.
   This is a property of any front carried by K⁺ diffusion, not only of this model. Ahead of
   such a front the K⁺ excess falls off as e^(−x/ℓ) with ℓ = D/v, where D = 1.96/1.6² ≈
   0.77 µm²/ms is the effective K⁺ diffusion coefficient. At v = 3.4 mm/min that gives ℓ ≈
   14 µm, and going from +0.25 to +4.3 mM (7.3 mM) takes (ℓ/v)·ln(4.3/0.25) ≈ 0.7 s, not 7 s.
   A 7 s lead at that speed needs a decay length ℓ of about 140 µm, an effective spread about
   10 times faster than extracellular K⁺ diffusion (the precursor then spans v·7 s ≈ 400 µm).
   Candidates the model leaves out are astrocytic K⁺ spatial buffering through the
   gap-junction syncytium, the wave arriving first in another layer and K⁺ reaching layer 2/3
   from there, and the electrode sampling a different depth than the imaging plane. (This
   estimate is ours, not the paper's.)
3. **Recovery needs glia, and then peak [K⁺]e is low.** Without glial uptake the closed slab
   never recovers, and [K⁺]e peaks at 25–29 mM. With uptake at 0.5/s the cells repolarise after
   10–18 s, but [K⁺]e peaks near 11 mM, below the 30–60 mM usually reported for CSD (a
   literature range; Enger et al. give no peak value). The paper's DC shift lasts 66 s.
4. **The potential shift looks too small, but this is not a like-for-like comparison.** The
   laminar Δφ is −2 to −4 mV, while DC shifts in CSD are typically −15 to −30 mV against a
   remote reference (a literature range; Enger et al. give the duration, not the amplitude, in
   the text we could read). The membranes here do not see φ and, in these runs, there are no
   glial currents, both of which shape the DC shift.

A closer replication would need astrocytes (K⁺ uptake and gap-junction redistribution, which
targets finding 2 and the recovery time; added below), a thicker, layered slab with an open
boundary or a remote reference for the DC shift, and Ca²⁺ and glutamate signals if the timing
of those is to be compared.

## Astrocytes (spatial buffering)

`SdConfig::glia` fills the tissue with the astrocytic syncytium of `ed_knp::glia`, after Halnes
et al. (2013, PLoS Comput Biol 9:e1003386). Every grid cell holds astrocytic interior
(volume fraction 0.4, tortuosity 3.2) with its own Na⁺, K⁺ and Cl⁻ as a second KNP domain,
behind a membrane with Kir4.1 (its conductance rising with √[K⁺]e), Na⁺ and Cl⁻ leaks and a
Na⁺/K⁺ pump, calibrated so that rest (−85 mV) is a steady state. The membrane potential comes
from a cable equation through the gap-junction-coupled interior, and the membrane currents move
ions between the two domains. In Halnes et al.'s own setting (a 300 µm column with K⁺ released
at one end, `glia::tests`) the astrocytes take up K⁺ at the source (peak [K⁺]e 10 instead of
16 mM) and release it 100–300 µm away, where extracellular diffusion alone has not yet
arrived. Cutting the gap junctions removes the effect.

The runs of `sd_electrodes` with astrocytes, each next to the same drive without them (40 s,
far electrode, no KCl uptake; membrane area in µm² per µm³ of tissue, Halnes et al.'s is 8):

| g_NaP (mS/cm²) | driven | glia (µm⁻¹, coupling) | speed (mm/min) | K⁺ lead (s) | [K⁺]e at depolarisation (mM) | peak [K⁺]e (mM) | depolarisation lasts (s) | laminar Δφ (mV) |
|---|---|---|---|---|---|---|---|---|
| 0.3 | 100 | – | 9.6 | −0.04 | 3.1 | 31.0 | > 37.8 | −4.7 |
| 0.3 | 100 | 8, 1 | 17.2 | −0.02 | 3.2 | 13.4 | > 38.7 | −4.3 |
| 0.25 | 400 | – | 5.8 | 0.06 | 3.4 | 29.4 | > 36.9 | −3.9 |
| 0.25 | 400 | 8, 1 | 6.0 | 0.16 | 3.4 | 12.6 | > 36.8 | −3.0 |
| 0.2 | 100 | – | 3.5 | 0.36 | 4.1 | 27.4 | > 33.8 | −3.0 |
| 0.2 | 100 | 8, 1 | – | – | – | 3.1 | – | 0.0 |
| 0.2 | 100 | 1, 1 | 4.8 | 0.86 | 4.2 | 12.9 | > 34.6 | −1.8 |
| 0.2 | 100 | 1, 0.01 | 2.5 | 0.54 | 4.1 | 17.1 | > 31.6 | −3.1 |

1. **Halnes et al.'s astrocytes raise the ignition threshold.** At their membrane density the
   astrocytes take up K⁺ fast. In the 200 µm test box at g_NaP = 0.25, two driven cells raise
   [K⁺]e to 3.4 mM in the first second with astrocytes and to 11.5 mM without, where the
   tissue is already igniting (test
   `astrocytes_conserve_ions_and_take_up_the_released_potassium`). A wave then starts only in
   the most excitable tissue: g_NaP = 0.3, or 0.25 with 400 driven cells, but not 0.2 with 100.
   In exploratory runs outside the example (12–20 s), it also did not start at 0.25 with one
   driven cell, or with 4 µm⁻¹ at 0.25/1, 0.2/100 and 0.15/400. Once a wave runs, the
   astrocytes do not slow it. [K⁺]e then peaks at about 13 mM instead of 30, below the
   30–60 mM usually reported for CSD.
2. **They lengthen the K⁺ precursor, but not nearly enough.** With an eighth of that density, at
   g_NaP = 0.2 and the same 100 driven cells, the K⁺ lead grows from 0.36 to 0.86 s, and the
   precursor in space (lead × speed) from 21 to 69 µm. Enger et al.'s K⁺ onset comes about 7 s
   before the neuronal Ca²⁺ rise, about 400 µm, and 5.9 s (335 µm) before the DC shift. The
   neurons still depolarise at about 4 mM, because the neurons set that threshold. A lead of
   seconds needs neurons that hold out to about 7 mM while the wave still propagates. Without
   astrocytes that happens at g_NaP = 0.1 with twice the neuron density: a 1.6 s lead at
   7.2 mM, 1.9 mm/min, in a 15 s exploratory run. With astrocytes at 8 or 2 µm⁻¹ at that
   density, no wave started at g_NaP 0.1–0.25 in exploratory runs.
3. **The syncytium speeds the wave up, while experiments suggest the opposite.** Coupled
   astrocytes carry K⁺ ahead of the front. At 1 µm⁻¹ the wave runs at 4.8 mm/min, against 2.5
   with the coupling cut to 1 % and 3.5 without astrocytes. At Halnes et al.'s density and
   g_NaP = 0.3 it runs at 17.2 mm/min instead of 9.6. In mice whose astrocytes lack Cx43, SD
   instead propagates about 20 % faster (Theis et al. 2003, J Neurosci 23:766). This model
   leaves out whatever makes coupling slow SD down.
4. **Astrocytes do not end the depolarisation.** They redistribute K⁺ but remove none, so the
   closed slab still does not recover within 40 s. Only the KCl uptake option lets it recover.
5. **They make the laminar Δφ smaller, not larger:** −1.8 to −4.3 mV, against −3.0 to
   −4.7 mV without them at the same drive. Their currents do not bring the potential shift
   closer to the −15 to −30 mV of CSD recordings in this slab.
