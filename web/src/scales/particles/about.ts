import type { AboutContent } from '../../ui/about';

export const about: AboutContent = {
  intro: `
    In a dendritic spine there may be only a handful of free Ca²⁺ ions at a time, so a smooth
    concentration is a poor description. Instead each ion is followed individually as it
    random-walks through the spine, with a drift added by the electric field.

    Ions start in the spine head and are removed when they reach the base of the neck. The mean
    first passage time (MFPT) is how long, on average, an ion stays in the spine. An electric
    field along the neck, such as during an excitatory postsynaptic potential, changes it
    substantially.`,
  equations: [
    {
      tex: String.raw`\mathbf{r}(t+\Delta t) = \mathbf{r}(t) + \mathbf{X} + \frac{z D\,\Delta t}{\psi}\mathbf{E}, \qquad \mathbf{X}\sim\mathcal{N}(0,\, 2D\Delta t)\ \text{per axis}`,
      caption:
        'Displacement rule: Brownian motion plus electric drift. For a constant field $\\mathbf{E} = -\\nabla\\phi$ in free space this is exact for any $\\Delta t$.',
    },
    {
      tex: String.raw`p_E(\mathbf{r}, \Delta t) = \frac{1}{\pi^{3/2}\ell^3}\exp\!\left(-\frac{\lVert \mathbf{r} - zD\mathbf{E}\Delta t/\psi\rVert^2}{\ell^2}\right), \qquad \ell = \sqrt{4D\Delta t}`,
      caption:
        'The same rule as a probability density: the solution of the Nernst-Planck equation started from a single point.',
    },
    {
      tex: String.raw`D\nabla^2 T + \frac{zD}{\psi}\mathbf{E}\cdot\nabla T = -1, \qquad T = 0 \text{ at the neck base}, \quad \partial_n T = 0 \text{ on the walls}`,
      caption:
        'The mean first passage time $T(\\mathbf{r})$ from a starting point $\\mathbf{r}$ solves this backward equation, which the simulation estimates by averaging over many ions.',
    },
  ],
  assumptions: [
    'Ions move independently: no ion-ion interactions, buffering or reactions.',
    'The electric field is prescribed (constant along the neck) and not affected by the ions themselves.',
    'Spine walls reflect ions; the base of the neck absorbs them.',
    'Near walls the step is no longer exact, so boundary effects only converge for small steps ($\\Delta t \\lesssim 10^{-4}$ ms).',
  ],
  parameters: [
    { symbol: String.raw`z`, description: 'Valence of Ca²⁺', value: '2' },
    { symbol: String.raw`D`, description: 'Diffusion coefficient of Ca²⁺', value: '0.4 µm²/ms' },
    { symbol: String.raw`\psi`, description: 'Thermal voltage', value: '25.85 mV' },
    { symbol: String.raw`r_h`, description: 'Spine head radius', value: '0.5 µm' },
    { symbol: String.raw`r_n`, description: 'Spine neck radius', value: '0.1 µm' },
    {
      symbol: String.raw`L_n`,
      description: 'Neck length (sharp junction / with funnel of radius $r_h/2$)',
      value: '1.0 / 0.89 µm',
    },
  ],
  references: [
    {
      text: 'A. Solbrå, T. Bartol. Monte Carlo simulations of charged molecules in MCell. Publication 3 in A. V. Solbrå, <em>Modeling Electrical and Diffusive Transports in Neural Tissue</em>, PhD thesis, University of Oslo (2019).',
    },
    {
      text: 'Modified MCell with electric drift: andreavs/mcell (emcell branch)',
      href: 'https://github.com/andreavs/mcell/tree/emcell',
    },
  ],
};
