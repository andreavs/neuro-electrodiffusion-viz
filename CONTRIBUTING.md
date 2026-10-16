# Contributing

Work is tracked as GitHub issues. Each issue lists its dependencies and acceptance criteria.

## Workflow

1. Branch from `main`: `issue-<number>-<short-slug>`.
2. Keep the change scoped to the issue. If you find extra work, open a new issue instead.
3. Add tests for every behaviour you add: Rust unit tests for solvers (conservation, analytic
   comparisons, convergence), Vitest for TypeScript logic.
4. Before pushing, run locally:
   ```sh
   cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
   cd web && npm run wasm && npm run typecheck && npm test && npm run build && npm run e2e
   ```
5. Open a PR that says `Closes #<number>`, describe what a user would see, and get a review.
   CI must be green before merge. Merging to `main` deploys automatically.

## Layout rules (to keep parallel work conflict-free)

- Each scale owns its own files: `crates/ed-<scale>`, `crates/ed-wasm/src/<scale>.rs`,
  `web/src/sim/solvers/<scale>.ts`, `web/src/scales/<scale>/`. Avoid touching another scale's files.
- Shared code goes in `crates/ed-core` (physics, numerics, membrane models) and
  `web/src/sim/`, `web/src/render/`, `web/src/ui/` (runtime, rendering, controls).
- Units everywhere: µm, ms, mM, mV, µm²/ms (see `crates/ed-core/src/lib.rs`).

## Performance budget

The app must stay smooth on a mid-range laptop:

- The UI thread renders at 60 fps; solvers run only in the worker.
- Worker ticks spend at most ~12 ms of solver time before posting a frame.
- Frames transfer typed arrays (no copies of large buffers on the UI thread, no JSON for field data).
- Build WASM in release mode; avoid allocation inside time-step loops.
