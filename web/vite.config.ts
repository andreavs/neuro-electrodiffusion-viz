import { defineConfig } from 'vite';

// Relative base so the build works under https://andreavs.github.io/neuro-electrodiffusion-viz/
export default defineConfig({
  base: './',
  build: { target: 'es2022' },
  worker: { format: 'es' },
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
});
