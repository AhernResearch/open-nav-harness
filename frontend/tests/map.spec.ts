import { test, expect } from '@playwright/test';

test('whole map and live overlays fit; zoom and drag can be reset on short and mobile screens', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.setViewportSize({ width: 1440, height: 640 });
  const events = [
    { type: 'session.started', instruction: 'Inspect the room.', scene_id: 'S1' },
    { type: 'item.started', item: { id: 'map', type: 'map', name: 'observe_map', arguments: {}, status: 'in_progress' } },
    { type: 'item.completed', item: { id: 'map', status: 'completed', map: { artifact: 'a'.repeat(64), size: [1600, 600], places_px: { '0': [700, 300], '1': [800, 300] }, transform: { matrix: [1, 0, 0, 0, 1, 0], ppm: 1, residual_px: 0, nodes: 2 } } } },
    { type: 'item.started', item: { id: 'move', type: 'motion', name: 'execute_subtask', arguments: { instruction: 'Inspect the room.' }, status: 'in_progress' } },
    { type: 'item.updated', item: { id: 'move', pose: { position: [1900, 0, -150], heading_rad: 0 }, frame: { front: 'camera.png' } } },
    { type: 'item.completed', item: { id: 'move', status: 'completed' } },
  ];
  const body = `id: 10.4\ndata: ${JSON.stringify(events)}\n\nevent: end\ndata: {"status":"completed"}\n\n`;
  const map = '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="600"><rect width="1600" height="600" fill="#e4e2db"/><path d="M0 0H1600V600H0Z" fill="none" stroke="#5656d9" stroke-width="40"/><text x="80" y="100" font-size="64">Full map</text></svg>';
  await page.route('**/api/runs/r/episodes/map-fit/stream**', route => route.fulfill({ headers: { 'content-type': 'text/event-stream' }, body }));
  await page.route('**/api/runs/r/episodes/map-fit/artifact/**', route => route.fulfill({ contentType: 'image/svg+xml', body: map }));
  await page.route('**/api/runs/r/episodes/map-fit/files/camera.png', route => route.fulfill({ contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="800"><rect width="800" height="800" fill="#e0ddf8"/></svg>' }));
  await page.goto('/harness?run=r/map-fit');
  const svg = page.getByRole('img', { name: '记忆地图' });
  await expect(svg).toBeVisible();
  const viewBox = () => svg.evaluate((element: SVGSVGElement) => {
    const b = element.viewBox.baseVal;
    return { x: b.x, y: b.y, width: b.width, height: b.height };
  });
  const fit = await viewBox();
  expect(fit.x).toBeLessThanOrEqual(0);
  expect(fit.y).toBeLessThanOrEqual(-182);
  expect(fit.x + fit.width).toBeGreaterThanOrEqual(1932);
  expect(fit.y + fit.height).toBeGreaterThanOrEqual(600);
  const withinCard = () => svg.evaluate(element => {
    const image = element.getBoundingClientRect();
    const card = element.closest('.card')!.getBoundingClientRect();
    return image.top >= card.top && image.bottom <= card.bottom && image.left >= card.left && image.right <= card.right;
  });
  expect(await withinCard()).toBe(true);
  await page.getByRole('button', { name: '放大地图' }).click();
  expect((await viewBox()).width).toBeLessThan(fit.width);
  const zoomed = await viewBox();
  const bounds = (await svg.boundingBox())!;
  await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
  await page.mouse.down();
  await page.mouse.move(bounds.x + bounds.width / 2 + 60, bounds.y + bounds.height / 2 + 30, { steps: 4 });
  await page.mouse.up();
  expect((await viewBox()).x).toBeLessThan(zoomed.x);
  await page.getByRole('button', { name: '显示全图' }).click();
  expect(await viewBox()).toEqual(fit);
  await page.getByRole('button', { name: '缩小地图' }).click();
  expect((await viewBox()).width).toBeGreaterThan(fit.width);
  await page.getByRole('button', { name: '显示全图' }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await svg.scrollIntoViewIfNeeded();
  expect(await withinCard()).toBe(true);
  expect(await viewBox()).toEqual(fit);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});
