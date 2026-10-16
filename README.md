# Neural Electrodiffusion Viz

Real-time, in-browser simulations of ion electrodiffusion in neural tissue at three scales,
based on the PhD thesis *Modeling Electrical and Diffusive Transports in Neural Tissue*
(Andreas Våvang Solbrå, University of Oslo, 2019).

**Live app:** https://andreavs.github.io/neuro-electrodiffusion-viz/

| Scale | Model | What you see |
|---|---|---|
| Tissue (~1 mm³) | Kirchhoff-Nernst-Planck (KNP), neurons as point sources | Extracellular K⁺/Na⁺ and potential spreading around an active neuron |
| Cell (~100 µm) | KNP-EMI, explicit intra/extracellular geometry | Action potentials in an axon bundle, ion build-up, ephaptic coupling |
| Molecular (~1 µm) | Brownian dynamics with electric drift | Ca²⁺ ions escaping a dendritic spine, mean first passage time |

## Stack

- **Solvers:** Rust (`crates/`), compiled to WebAssembly with `wasm-pack`, run in a Web Worker.
- **Front end:** TypeScript + Vite (`web/`), WebGL rendering.
- **CI/CD:** GitHub Actions. PRs run Rust and web tests; merges to `main` run the same tests, then deploy to GitHub Pages.

## Develop

Requirements: Rust stable (with `wasm32-unknown-unknown`), [`wasm-pack`](https://github.com/drager/wasm-pack), Node 22.

```sh
cargo test --workspace          # Rust unit tests
cd web
npm ci
npm run wasm                    # build the WASM package into web/src/wasm/pkg
npm run dev                     # http://localhost:5173
npm test                        # TypeScript unit tests
npm run build                   # production build into web/dist
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the models and code layout, and
[CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.
[docs/spreading-depression.md](docs/spreading-depression.md) asks whether one neuron can start
spreading depression through the extracellular ions alone.
