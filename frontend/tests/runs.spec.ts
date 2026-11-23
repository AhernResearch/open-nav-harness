import { test, expect } from '@playwright/test';

const png = '<svg xmlns="http://www.w3.org/2000/svg" width="80" height="60"><rect width="80" height="60" fill="#e0ddf8"/></svg>';

test('a kernel run episode replays legs, plan, views and frames, and cancels the run', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  const leg = { leg_id: 1, kind: 'point_goal', status: 'reached', reason: null };
  const views = ['front', 'right', 'back', 'left'].map((v, i) => ({ artifact: String(i).repeat(64), caption: `${v} view` }));
  const events = [
    { type: 'session.started', instruction: 'Walk to the sofa and stop.', episode_id: '7' },
    { type: 'item.started', item: { id: 'p', type: 'plan', name: 'update_plan', arguments: { steps: [{ text: 'go to the sofa', done: false }] }, status: 'in_progress' } },
    { type: 'item.completed', item: { id: 'p', status: 'completed', state: { tasks: [{ instruction: 'go to the sofa', status: 'in_progress' }] } } },
    { type: 'item.started', item: { id: 'd', type: 'motion', name: 'delegate', arguments: { view: 'left', point: [500, 800], instruction: 'Walk to the sofa' }, status: 'in_progress' } },
    { type: 'item.completed', item: { id: 'd', status: 'completed', leg, result: ['{"leg": {}}'], image_views: views, images: views.map(v => v.artifact) } },
    { type: 'item.updated', item: { id: 'd', frames: [{ policy: 'trace/media/' + 'f'.repeat(64) }, { policy: 'trace/media/' + 'e'.repeat(64) }] } },
  ];
  let cancelled = false;
  const base = '**/api/runs/r1/episodes/7';
  await page.route('**/api/harness/attempts', route => route.fulfill({ json: [
    { id: 'r1/7', run_id: 'r1', episode: '7', status: 'running', name: 'kernel', created: 1, base: '/api/runs/r1/episodes/7', meta: { instruction: 'Walk to the sofa and stop.' } }] }));
  await page.route(base + '/stream**', route => route.fulfill({ headers: { 'content-type': 'text/event-stream' },
    body: `id: 9\ndata: ${JSON.stringify(events)}\n\nevent: end\ndata: {"status":"running","run_id":"r1"}\n\n` }));
  await page.route(base + '/artifact/**', route => route.fulfill({ contentType: 'image/svg+xml', body: png }));
  await page.route(base + '/files/**', route => route.fulfill({ contentType: 'image/svg+xml', body: png }));
  await page.route('**/api/runs/r1/cancel', route => { cancelled = true; return route.fulfill({ json: {} }); });
  await page.goto('/harness?run=r1%2F7');
  await expect(page.getByText('Walk to the sofa · 左侧')).toBeVisible();
  await expect(page.getByText('已到达 · 2 步')).toBeVisible();
  await expect(page.locator('.plan-dock')).toContainText('go to the sofa');
  await expect(page.locator('.hx-panel img')).toHaveCount(1);
  await page.getByText('Walk to the sofa · 左侧').click();
  await expect(page.locator('.thumbs figcaption').first()).toHaveText('前方');
  await expect(page.getByRole('button', { name: '完成当前条后暂停' })).toHaveCount(0);
  page.once('dialog', dialog => dialog.accept());
  await page.getByRole('button', { name: '停止' }).click();
  await expect.poll(() => cancelled).toBe(true);
  expect(errors).toEqual([]);
});
