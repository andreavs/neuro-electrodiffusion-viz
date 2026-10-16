import type { SolverId } from '../protocol';
import type { SolverFactory } from '../solver';
import { createEmiSolver } from './emi';
import { createKnpSolver } from './knp';
import { createParticlesSolver } from './particles';
import { createSpreadingSolver } from './spreading';

export const SOLVERS: Record<SolverId, SolverFactory> = {
  knp: createKnpSolver,
  emi: createEmiSolver,
  particles: createParticlesSolver,
  'knp-sd': createSpreadingSolver,
};
