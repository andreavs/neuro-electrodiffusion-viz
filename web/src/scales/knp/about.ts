import type { AboutContent } from '../../ui/about';

export const about: AboutContent = {
  intro: `
    The extracellular space of about a cubic millimetre of cortex is treated as a continuous
    porous medium. Neurons are not resolved geometrically: every compartment of a neuron model
    injects or removes ions at a point, and the ions then spread by diffusion and electric drift.

    Ion concentrations follow the Nernst-Planck equations. Instead of Poisson's equation, the
    potential is chosen so that the bulk stays electroneutral (the Kirchhoff-Nernst-Planck, or
    KNP, scheme). This makes millimetre-scale simulations over seconds feasible, and on scales
    coarser than tens of nanometres it agrees closely with the full Poisson-Nernst-Planck model.`,
  equations: [
    {
      tex: String.raw`\mathbf{J}_k = -\tilde D_k \nabla c_k - \frac{\tilde D_k z_k}{\psi}\, c_k \nabla\phi`,
      caption:
        'Nernst-Planck flux of ion species $k$: diffusion down its concentration gradient plus migration in the electric field.',
    },
    {
      tex: String.raw`\frac{\partial c_k}{\partial t} = -\nabla\cdot\mathbf{J}_k + \tilde f_k`,
      caption: 'Conservation of each ion species, with neuronal point sources $\\tilde f_k$.',
    },
    {
      tex: String.raw`\frac{\partial \rho}{\partial t} = F\sum_k z_k \frac{\partial c_k}{\partial t} = -\tilde I_{\mathrm{cap}}`,
      caption:
        'Electroneutrality: free charge only changes where capacitive membrane current charges or discharges a neuron.',
    },
    {
      tex: String.raw`\nabla\cdot(\sigma\nabla\phi) = -F\sum_k z_k \nabla\cdot(\tilde D_k\nabla c_k) - F\sum_k z_k \tilde f_k - \tilde I_{\mathrm{cap}}`,
      caption:
        'Combining the three gives a linear elliptic equation for $\\phi$, solved once per time step. The conductivity $\\sigma = (F/\\psi)\\sum_k \\tilde D_k z_k^2 c_k$ follows from the local concentrations.',
    },
    {
      tex: String.raw`\tilde D_k = \frac{D_k}{\lambda^2}, \qquad \tilde f_k = \frac{f_k}{\alpha}`,
      caption:
        'Tissue averaging: tortuosity $\\lambda$ slows diffusion, and sources are concentrated into the extracellular volume fraction $\\alpha$.',
    },
  ],
  assumptions: [
    'The extracellular space is a homogeneous porous medium; cells only enter through $\\lambda$ and $\\alpha$.',
    'The bulk solution is electroneutral. Charge relaxation and Debye layers (nanometres, nanoseconds) are not resolved.',
    'Neurons are one-way coupled point sources: their membrane currents drive the extracellular dynamics, but extracellular changes do not feed back on the neurons (no ephaptic effects).',
    'The generic anion X⁻ stands in for Cl⁻ and other anions and carries Cl⁻ properties.',
    'No advection or bulk fluid flow; temperature is constant.',
  ],
  parameters: [
    { symbol: String.raw`\psi = RT/F`, description: 'Thermal voltage at $T = 300$ K', value: '25.85 mV' },
    { symbol: String.raw`\lambda`, description: 'Tortuosity', value: '1.6' },
    { symbol: String.raw`\alpha`, description: 'Extracellular volume fraction', value: '0.2' },
    { symbol: String.raw`D_{\mathrm{Na}}`, description: 'Free diffusion coefficient, Na⁺', value: '1.33 µm²/ms' },
    { symbol: String.raw`D_{\mathrm{K}}`, description: 'Free diffusion coefficient, K⁺', value: '1.96 µm²/ms' },
    { symbol: String.raw`D_{\mathrm{X}}`, description: 'Free diffusion coefficient, X⁻', value: '2.03 µm²/ms' },
    { symbol: String.raw`D_{\mathrm{Ca}}`, description: 'Free diffusion coefficient, Ca²⁺', value: '0.71 µm²/ms' },
    {
      symbol: String.raw`c^0_{\mathrm{Na}},\ c^0_{\mathrm{K}},\ c^0_{\mathrm{X}},\ c^0_{\mathrm{Ca}}`,
      description: 'Baseline extracellular concentrations',
      value: '150, 3, 155.8, 1.4 mM',
    },
  ],
  references: [
    {
      text: 'A. Solbrå, A. W. Bergersen, J. van den Brink, A. Malthe-Sørenssen, G. T. Einevoll, G. Halnes. A Kirchhoff-Nernst-Planck framework for modeling large scale extracellular electrodiffusion surrounding morphologically detailed neurons. PLOS Computational Biology 14(10): e1006510 (2018).',
      href: 'https://doi.org/10.1371/journal.pcbi.1006510',
    },
    {
      text: 'Original implementation (FEniCS): CINPLA/KNPsim',
      href: 'https://github.com/CINPLA/KNPsim',
    },
  ],
};
