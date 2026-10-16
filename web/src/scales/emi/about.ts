import type { AboutContent } from '../../ui/about';

export const about: AboutContent = {
  intro: `
    Here the cells have a shape. The intracellular space $\\Omega_i$ of each axon and the
    extracellular space $\\Omega_e$ around them are separate domains, joined at the cell membrane
    $\\Gamma$. Ions move by electrodiffusion inside each domain and cross the membrane through
    Hodgkin-Huxley ion channels.

    Because both sides of the membrane are solved together, one axon firing changes the
    extracellular potential and concentrations around its neighbours. This lets the model show
    ephaptic coupling in a bundle of unmyelinated axons, and how K⁺ builds up in the narrow gaps
    between them.`,
  equations: [
    {
      tex: String.raw`\frac{\partial [k]_r}{\partial t} + \nabla\cdot\mathbf{J}_r^k = 0, \qquad \mathbf{J}_r^k = -D_r^k\nabla[k]_r - \frac{D_r^k z_k}{\psi}[k]_r\nabla\phi_r \qquad \text{in } \Omega_r,\ r\in\{i,e\}`,
      caption: 'Ion conservation with Nernst-Planck fluxes in each region.',
    },
    {
      tex: String.raw`F\sum_k z_k\,\nabla\cdot\mathbf{J}_r^k = 0 \qquad \text{in } \Omega_r`,
      caption: 'Bulk electroneutrality (KNP) determines the potentials $\\phi_i$ and $\\phi_e$.',
    },
    {
      tex: String.raw`\phi_M = \phi_i - \phi_e, \qquad C_M\frac{\partial \phi_M}{\partial t} = I_M - I_{\mathrm{ch}} \qquad \text{on } \Gamma`,
      caption:
        'The membrane potential is the jump in potential across $\\Gamma$. The total membrane current $I_M$ is the channel current plus the capacitive current.',
    },
    {
      tex: String.raw`-\mathbf{J}_i^k\cdot\mathbf{n}_i = \frac{I_{\mathrm{ch}}^k + \alpha_i^k (I_M - I_{\mathrm{ch}})}{F z_k}, \qquad \mathbf{J}_e^k\cdot\mathbf{n}_e = \frac{I_{\mathrm{ch}}^k + \alpha_e^k (I_M - I_{\mathrm{ch}})}{F z_k}`,
      caption:
        'Ion-specific fluxes across the membrane: the channel current of ion $k$ plus its share of the capacitive current.',
    },
    {
      tex: String.raw`\alpha_r^k = \frac{D_r^k z_k^2 [k]_r}{\sum_l D_r^l z_l^2 [l]_r}`,
      caption: 'Each ion carries a share of the capacitive current proportional to its contribution to the local conductivity.',
    },
    {
      tex: String.raw`I_{\mathrm{ch}}^{\mathrm{Na}} = (\bar g_{\mathrm{Na}} m^3 h + g_L^{\mathrm{Na}})(\phi_M - E_{\mathrm{Na}}), \quad I_{\mathrm{ch}}^{\mathrm{K}} = (\bar g_{\mathrm{K}} n^4 + g_L^{\mathrm{K}})(\phi_M - E_{\mathrm{K}}), \quad E_k = \frac{\psi}{z_k}\ln\frac{[k]_e}{[k]_i}`,
      caption:
        'Hodgkin-Huxley channels, with reversal potentials computed from the local, changing concentrations.',
    },
  ],
  assumptions: [
    'Both intra- and extracellular bulk solutions are electroneutral; charge only sits on the membrane.',
    'Axons are rectangular blocks inside an extracellular box. In this app they are aligned with a voxel grid (finite volumes) instead of the mortar finite elements used in the paper.',
    'Membrane channels follow the classic squid-axon Hodgkin-Huxley kinetics plus passive leaks. There are no pumps or cotransporters, so concentrations drift during long runs.',
    'Capacitive current is split between ion species in proportion to their local conductivity.',
    'Firing is triggered by a synaptic conductance on part of one axon.',
  ],
  parameters: [
    { symbol: String.raw`C_M`, description: 'Membrane capacitance', value: '1 µF/cm²' },
    { symbol: String.raw`\bar g_{\mathrm{Na}},\ \bar g_{\mathrm{K}}`, description: 'Maximal HH conductances', value: '120, 36 mS/cm²' },
    {
      symbol: String.raw`g_L^{\mathrm{Na}},\ g_L^{\mathrm{K}},\ g_L^{\mathrm{Cl}}`,
      description: 'Leak conductances',
      value: '0.2, 0.8, 0 mS/cm²',
    },
    { symbol: String.raw`[\mathrm{Na}]_i,\ [\mathrm{Na}]_e`, description: 'Initial Na⁺ (inside, outside)', value: '12, 100 mM' },
    { symbol: String.raw`[\mathrm{K}]_i,\ [\mathrm{K}]_e`, description: 'Initial K⁺ (inside, outside)', value: '125, 4 mM' },
    { symbol: String.raw`[\mathrm{Cl}]_i,\ [\mathrm{Cl}]_e`, description: 'Initial Cl⁻ (inside, outside)', value: '137, 104 mM' },
    { symbol: String.raw`\phi_M^0`, description: 'Resting membrane potential', value: '−67.74 mV' },
    { symbol: String.raw`m_0,\ h_0,\ n_0`, description: 'Initial gating variables', value: '0.0379, 0.688, 0.276' },
    { symbol: String.raw`g_{\mathrm{syn}},\ \tau_{\mathrm{syn}}`, description: 'Synaptic stimulus', value: '4 mS/cm², 2 ms' },
    { symbol: String.raw`\Delta t`, description: 'PDE time step (membrane ODE uses 25 substeps)', value: '0.1 ms' },
  ],
  references: [
    {
      text: 'A. J. Ellingsrud*, A. Solbrå*, K. H. Pettersen, G. T. Einevoll, G. Halnes, M. E. Rognes. A mathematical framework for electrodiffusion based on explicit geometrical representation applied to ephaptic coupling in unmyelinated olfactory axon bundles. Publication 2 in A. V. Solbrå, <em>Modeling Electrical and Diffusive Transports in Neural Tissue</em>, PhD thesis, University of Oslo (2019).',
    },
  ],
};
