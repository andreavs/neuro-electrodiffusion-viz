import { expect, test, type Locator, type Page } from '@playwright/test';

const SCALE_IDS = ['knp', 'emi', 'particles'] as const;

/**
 * Collects console errors, uncaught exceptions and unhandled rejections from the page and
 * its workers (Chromium reports worker errors on the page), failed requests, and assets
 * that `vite preview`'s SPA fallback answered with index.html instead of a 404.
 */
function watchErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(`console: ${msg.text()}`);
  });
  page.on('pageerror', (err) => errors.push(`pageerror: ${err.message}`));
  page.on('requestfailed', (req) => errors.push(`requestfailed: ${req.url()} ${req.failure()?.errorText ?? ''}`));
  page.on('response', (res) => {
    const url = new URL(res.url());
    if (res.status() >= 400) errors.push(`http ${res.status()}: ${url.pathname}`);
    else if (url.pathname.includes('/assets/') && (res.headers()['content-type'] ?? '').includes('text/html')) {
      errors.push(`missing asset (got index.html): ${url.pathname}`);
    }
  });
  return errors;
}

/**
 * Waits until the worker reports the WASM solver loaded (`<html data-solver="ready">`, set by
 * SimClient.ready()). Fails as soon as an error is collected or the solver reports an error,
 * with the errors in the message, instead of running into the timeout.
 */
async function expectSolverReady(page: Page, errors: string[], timeoutMs = 10_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const state = await page.locator('html').getAttribute('data-solver');
    if (errors.length || state === 'error') {
      throw new Error(`solver failed to load (data-solver=${state}):\n${errors.join('\n')}`);
    }
    if (state === 'ready') return;
    if (Date.now() > deadline) throw new Error(`solver not ready after ${timeoutMs} ms (data-solver=${state})`);
    await page.waitForTimeout(50);
  }
}

/** True if the element's screenshot contains more than one colour. */
async function isNonBlank(page: Page, el: Locator): Promise<boolean> {
  const png = (await el.screenshot()).toString('base64');
  return page.evaluate(async (b64) => {
    const img = new Image();
    img.src = `data:image/png;base64,${b64}`;
    await img.decode();
    const c = document.createElement('canvas');
    c.width = img.width;
    c.height = img.height;
    const ctx = c.getContext('2d')!;
    ctx.drawImage(img, 0, 0);
    const px = new Uint32Array(ctx.getImageData(0, 0, c.width, c.height).data.buffer);
    return px.some((p) => p !== px[0]);
  }, png);
}

test('menu shows three scale cards', async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto('./');
  await expect(page.locator('.menu .card')).toHaveCount(3);
  for (const id of SCALE_IDS) {
    await expect(page.locator(`.card[href="#/${id}"]`)).toBeVisible();
  }
  expect(errors).toEqual([]);
});

for (const id of SCALE_IDS) {
  test(`${id}: card opens the view, the WASM solver loads and it renders`, async ({ page }) => {
    const errors = watchErrors(page);
    await page.goto('./');
    await page.locator(`.card[href="#/${id}"]`).click();
    await expect(page).toHaveURL(new RegExp(`#/${id}$`));
    await expectSolverReady(page, errors);

    const view = page.locator('#view');
    if ((await view.locator('[data-placeholder]').count()) === 0) {
      // A real view (#7 knp, #9 emi, #11 particles) must draw a frame into a canvas.
      const canvas = view.locator('canvas').first();
      await expect(canvas).toBeVisible();
      await expect.poll(() => isNonBlank(page, canvas), { message: 'canvas should not be blank', timeout: 10_000 }).toBe(true);
    }

    // Let any late async failure (worker errors, rejected promises) surface.
    await page.waitForTimeout(250);
    expect(errors).toEqual([]);
  });
}

test('deep link boots a scale directly and the brand link returns to the menu', async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto('./#/particles');
  await expectSolverReady(page, errors);
  await page.locator('.brand').click();
  await expect(page.locator('.menu .card')).toHaveCount(3);
  await expect(page.locator('html')).not.toHaveAttribute('data-solver');
  expect(errors).toEqual([]);
});

test('tissue scenario bar switches to spreading depression, which loads and renders', async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto('./#/knp');
  await expectSolverReady(page, errors);
  await page.locator('.kv-scenarios a[href="#/knp/spreading"]').click();
  await expect(page).toHaveURL(/#\/knp\/spreading$/);
  await expect(page.locator('.kv-scenarios [aria-current="page"]')).toHaveText(/Spreading depression/);
  await expectSolverReady(page, errors);
  const canvas = page.locator('#view canvas').first();
  await expect.poll(() => isNonBlank(page, canvas), { message: 'canvas should not be blank', timeout: 10_000 }).toBe(true);
  await expect.poll(() => page.locator('[data-out="cost"]').innerText(), { timeout: 10_000 }).toMatch(/per cell/);
  await page.waitForTimeout(250);
  expect(errors).toEqual([]);
});
