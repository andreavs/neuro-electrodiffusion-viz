import { describe, expect, it } from 'vitest';
import { scenarioFor } from './scenario';

describe('tissue scenarios', () => {
  it('picks the scenario from the route', () => {
    expect(scenarioFor('#/knp').path).toBe('');
    expect(scenarioFor('#/knp/spreading').path).toBe('spreading');
    expect(scenarioFor('#/knp/spreading?x=1').path).toBe('spreading');
    expect(scenarioFor('#/knp/nonsense').path).toBe('');
  });
});
