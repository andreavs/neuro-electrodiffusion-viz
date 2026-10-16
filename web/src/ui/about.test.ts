import temml from 'temml';
import { describe, expect, it } from 'vitest';
import { about as emi } from '../scales/emi/about';
import { about as knp } from '../scales/knp/about';
import { about as particles } from '../scales/particles/about';
import { SCALES, type ScaleId } from '../scales/registry';
import { loadAboutHtml, renderAboutBody, type AboutContent } from './about';

const CONTENT: Record<ScaleId, AboutContent> = { knp, emi, particles };

/** Every TeX fragment in a content object, display and inline. */
function texFragments(c: AboutContent): string[] {
  const prose = [c.intro, ...c.equations.map((e) => e.caption), ...c.assumptions]
    .concat(c.parameters.flatMap((p) => [p.description, p.value]));
  const inline = prose.flatMap((s) => [...s.matchAll(/\$([^$]+)\$/g)].map((m) => m[1]!));
  return [...c.equations.map((e) => e.tex), ...c.parameters.map((p) => p.symbol), ...inline];
}

describe.each(SCALES.map((s) => s.id))('about content for %s', (id) => {
  const c = CONTENT[id];

  it('has every section filled in', () => {
    expect(c.intro.trim()).not.toBe('');
    expect(c.equations.length).toBeGreaterThan(0);
    expect(c.assumptions.length).toBeGreaterThan(0);
    expect(c.parameters.length).toBeGreaterThan(0);
    expect(c.references.length).toBeGreaterThan(0);
  });

  it('only contains valid TeX', () => {
    for (const tex of texFragments(c)) {
      expect(() => temml.renderToString(tex, { throwOnError: true, strict: true }), tex).not.toThrow();
    }
  });

  it('has balanced inline math delimiters', () => {
    const prose = [c.intro, ...c.equations.map((e) => e.caption), ...c.assumptions];
    for (const s of prose) expect(s.split('$').length % 2, s).toBe(1);
  });

  it('renders to HTML with MathML and no leftover TeX', async () => {
    const html = await loadAboutHtml(id);
    expect(html).toContain('<math');
    expect(html).not.toMatch(/\$[^$]+\$/);
    expect(html).not.toContain('temml-error');
  });
});

describe('renderAboutBody', () => {
  const fake = (tex: string, display: boolean) => `[${display ? 'D' : 'I'}:${tex}]`;
  const body = renderAboutBody(
    {
      intro: 'First $a$.\n\n   Second.',
      equations: [{ tex: 'x=1', caption: 'cap $b$' }],
      assumptions: ['one'],
      parameters: [{ symbol: 'y', description: 'why', value: '2 mM' }],
      references: [{ text: 'Paper', href: 'https://doi.org/x' }, { text: 'Thesis' }],
    },
    fake,
  );

  it('splits paragraphs and renders inline and display math', () => {
    expect(body).toContain('<p>First [I:a].</p><p>Second.</p>');
    expect(body).toContain('[D:x=1]');
    expect(body).toContain('cap [I:b]');
    expect(body).toContain('<td>[I:y]</td><td>why</td><td>2 mM</td>');
  });

  it('links references that have a URL and opens them in a new tab safely', () => {
    expect(body).toContain('<a href="https://doi.org/x" target="_blank" rel="noopener">Paper</a>');
    expect(body).toContain('<li>Thesis</li>');
  });

  it('cites Publication 1 by DOI', async () => {
    expect(await loadAboutHtml('knp')).toContain('https://doi.org/10.1371/journal.pcbi.1006510');
  });
});
