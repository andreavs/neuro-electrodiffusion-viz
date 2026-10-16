import { expect, test, type Page } from '@playwright/test';

interface ScaleCase {
  id: 'knp' | 'emi' | 'particles';
  /**
   * Set to true once the scale's real view lands (#7 knp, #9 emi, #11 particles):
   * the view must then render a <canvas> whose pixels change while it runs.
   */
  frames: boolean;
}

const SCALES: ScaleCase[] = [
  { id: 'knp', frames: false },
  { id: 'emi', frames: false },
  { id: 'particles', frames: false },
];

/** Collects console errors, uncaught exceptions and unhandled rejections from the page and its workers. */
function watchErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(`console: ${msg.text()}`);
  });
  page.on('pageerror', (err) => errors.push(`pageerror: ${err.message}`));
  page.on('requestfailed', (req) => errors.push(`requestfailed: ${req.url()} ${req.failure()?.errorText ?? ''}`));
  page.on('response', (res) => {
    if (res.status() >= 400) errors.push(`http ${res.status()}: ${res.url()}`);
  });
  return errors;
}

test('menu shows three scale cards', async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto('./');
  const cards = page.locator('.menu .card');
  await expect(cards).toHaveCount(3);
  for (const { id } of SCALES) {
    await expect(page.locator(`.card[href="#/${id}"]`)).toBeVisible();
  }
  expect(errors).toEqual([]);
});

for (const scale of SCALES) {
  test(`${scale.id}: card opens the view and the WASM solver loads`, async ({ page }) => {
    const errors = watchErrors(page);
    const wasm = page.waitForResponse((res) => res.url().endsWith('.wasm') && res.ok());

    await page.goto('./');
    await page.locator(`.card[href="#/${scale.id}"]`).click();
    await expect(page).toHaveURL(new RegExp(`#/${scale.id}$`));

    // The worker fetched and instantiated the module: the request succeeded and the
    // worker answered `init` with `ready` (no "Failed to load solver" in the view).
    await wasm;
    const view = page.locator('#view');
    await expect(view).not.toBeEmpty();
    await expect(view).not.toContainText(/failed to load/i);

    if (scale.frames) {
      const canvas = view.locator('canvas').first();
      await expect(canvas).toBeVisible();
      const first = await canvas.screenshot();
      await expect
        .poll(async () => !(await canvas.screenshot()).equals(first), { timeout: 10_000 })
        .toBe(true);
    } else {
      // Placeholder view: the status line reports the loaded solver core.
      await expect(view.locator('[data-status]')).toContainText(/loaded/i, { timeout: 10_000 });
    }

    // Let any late async failure (worker errors, rejected promises) surface.
    await page.waitForTimeout(250);
    expect(errors).toEqual([]);
  });
}

test('deep link opens a scale directly and back returns to the menu', async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto('./#/particles');
  await expect(page.locator('#view')).not.toBeEmpty();
  await page.locator('.brand').click();
  await expect(page.locator('.menu .card')).toHaveCount(3);
  expect(errors).toEqual([]);
});
