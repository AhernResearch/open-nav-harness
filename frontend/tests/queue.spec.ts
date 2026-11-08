import { test, expect } from '@playwright/test';

test('pending run survives reload, explains blocked resources and opens its attempt without resubmitting', async ({ page }) => {
  await page.clock.install();
  let state: object = { status: 'queued', waiting_reason: 'needs_reconciliation', blocked_attempt_id: 'blocked-attempt' };
  let posts = 0;
  page.on('request', request => { if (request.method() === 'POST') posts++; });
  await page.route('**/api/harness/jobs/pending', route => route.fulfill({ json: state }));
  await page.route('**/api/attempts/started/stream**', route => route.fulfill({
    contentType: 'text/event-stream',
    body: 'data: [{"type":"session.started","instruction":"Walk down the hall."}]\n\nevent: end\ndata: {"status":"completed"}\n\n',
  }));
  await page.goto('/harness?job=pending');
  await expect(page.getByRole('status')).toHaveText('上次运行已中断，资源仍被锁定，需核验后释放。');
  await expect(page.getByRole('link', { name: '查看中断运行' })).toHaveAttribute('href', '/workbench?attempt=blocked-attempt');
  await page.reload();
  await expect(page.getByRole('status')).toContainText('资源仍被锁定');
  state = { status: 'paused' };
  await page.clock.fastForward(1600);
  await expect(page.getByRole('status')).toHaveText('运行已暂停。');
  state = { status: 'running', attempt_id: 'started' };
  await page.clock.fastForward(1600);
  await expect(page).toHaveURL(/\/harness\?attempt=started$/);
  await expect(page.locator('.user-msg')).toContainText('Walk down the hall.');
  expect(posts).toBe(0);
});

test('pending run reports a connection failure and a terminal failure without pretending to wait for resources', async ({ page }) => {
  await page.clock.install();
  let failed = true;
  await page.route('**/api/harness/jobs/error-job', route => failed
    ? route.fulfill({ status: 502, json: { detail: 'temporary failure' } })
    : route.fulfill({ json: { status: 'failed', error: 'Configuration rejected' } }));
  await page.goto('/harness?job=error-job');
  await expect(page.getByRole('status')).toHaveText('暂时无法读取运行状态，正在重新连接…');
  failed = false;
  await page.clock.fastForward(1600);
  await expect(page.getByRole('status')).toHaveText('运行失败。Configuration rejected');
  await expect(page.getByRole('link', { name: '查看工作台', exact: true })).toBeVisible();
});
