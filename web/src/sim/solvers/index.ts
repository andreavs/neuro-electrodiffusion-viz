import type { ScaleId } from '../../scales/registry';
import type { SolverFactory } from '../solver';
import { createEmiSolver } from './emi';
import { createKnpSolver } from './knp';
import { createParticlesSolver } from './particles';

export const SOLVERS: Record<ScaleId, SolverFactory> = {
  knp: createKnpSolver,
  emi: createEmiSolver,
  particles: createParticlesSolver,
};
