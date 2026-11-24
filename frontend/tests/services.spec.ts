import { test, expect } from '@playwright/test';

test('the workbench registers services without a login', async ({ page, request }) => {
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('/workbench');
  await expect(page.getByRole('heading', { name: '模型与服务', exact: true })).toBeVisible();
  await page.getByLabel('名称', { exact: true }).fill('test-planner');
  await page.getByRole('button', { name: '注册服务', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'test-planner' })).toBeVisible();
  await page.getByLabel('名称', { exact: true }).fill('test-env');
  await page.getByLabel('类型', { exact: true }).selectOption('environment');
  await page.getByLabel('API 地址', { exact: true }).fill('http://127.0.0.1:19200');
  await page.getByRole('button', { name: '注册服务', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'test-env' })).toBeVisible();
  expect((await (await request.get('/api/services')).json()).map((s: any) => s.spec.name))
    .toEqual(['test-planner', 'test-env']);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('heading', { name: '模型与服务', exact: true })).toBeVisible();
  await expect(page.getByLabel('管理口令')).toHaveCount(0);
  expect(errors).toEqual([]);
});
