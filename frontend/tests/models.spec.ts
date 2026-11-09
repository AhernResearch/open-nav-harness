import { test, expect, type Page, type APIRequestContext } from '@playwright/test';

async function setup(page: Page, request: APIRequestContext) {
  const prefix = `models-${Date.now()}`;
  const rows: any[] = [];
  for (const [kind, suffix] of [['planner', 'fp4'], ['planner', 'bf16'], ['planner', 'offline'], ['environment', 'env'], ['vla', 'vla']]) {
    const url = `http://${prefix}-${suffix}:18000${kind === 'planner' ? '/v1' : ''}`;
    const spec = {
      name: `${prefix}-${suffix}`, kind, url,
      ...(kind === 'planner' ? { planner: { model: `qwen27b-${suffix}`, base_url: url, context_limit: suffix === 'bf16' ? 32768 : 16384 } } : {}),
      metadata: kind === 'planner' ? { display_name: `Qwen 27B ${suffix.toUpperCase()}` } : {},
    };
    const response = await request.post('/api/services', { data: spec });
    expect(response.status()).toBe(201);
    const { id } = await response.json();
    rows.push({ id, spec, status: 'stopped', available: suffix !== 'offline', unavailable_reason: suffix === 'offline' ? '服务暂不可用' : '' });
  }
  await page.route('**/api/harness/services', route => route.fulfill({ json: rows }));
  await page.route('**/api/catalog', route => route.fulfill({ json: [{ name: 'r2r_train', dataset: 'r2r', split: 'train', scenes: [{ scene: 'S1', count: 1 }] }] }));
  await page.route('**/api/catalog/r2r_train/episodes?**', route => route.fulfill({ json: [{ index: 7, episode_id: 70, scene: 'S1', instruction: 'Walk up the stairs.' }] }));
  return rows;
}

test('choose a model and freeze its exact registered profile with the run', async ({ page, request }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  const rows = await setup(page, request);
  // A reused configuration must not restore the old nine-tool surface or mode.
  await page.route('**/api/experiments', route => route.request().method() === 'GET'
    ? route.fulfill({ json: [{ spec: {
      environment_service: rows[3].id, vla_service: rows[4].id,
      config: { harness: { mode: 'global' }, vla: { backend: 'lightnav' },
        planner: { max_turns: 70, tools: ['observe', 'terminate_episode', 'observe_map', 'navigate_relative', 'navigate_to_node', 'update_plan', 'execute_subtask', 'update_memory', 'read_memory'] } },
    } }] })
    : route.continue());
  await page.route('**/api/jobs', route => route.fulfill({ json: { id: 'model-picker-job' } }));
  await page.route('**/api/jobs/model-picker-job/attempts', route => route.fulfill({ json: [] }));
  await page.route('**/api/harness/jobs/model-picker-job', route => route.fulfill({ json: { status: 'queued', waiting_reason: 'queued' } }));
  await page.goto('/harness');
  await expect(page.getByRole('button', { name: '选择模型' })).toContainText('Qwen 27B FP4');
  await page.getByRole('button', { name: '选择模型' }).click();
  await expect(page.getByRole('menuitemradio', { name: /Qwen 27B OFFLINE/ })).toBeDisabled();
  await page.getByRole('menuitemradio', { name: /Qwen 27B BF16/ }).click();
  await expect(page.getByRole('button', { name: '选择模型' })).toContainText('Qwen 27B BF16');
  await page.getByRole('button', { name: 'Walk up the stairs.' }).click();
  const frozen = page.waitForResponse(response => response.url().endsWith('/api/experiments') && response.request().method() === 'POST');
  await page.getByRole('button', { name: '开始运行' }).click();
  const response = await frozen;
  expect(response.status()).toBe(201);
  const { id } = await response.json();
  const experiments = await (await request.get('/api/experiments')).json();
  const spec = experiments.find((experiment: any) => experiment.id === id).spec;
  expect(spec.planner_service).toBe(rows[1].id);
  expect(spec.config.planner.profile.model).toBe('qwen27b-bf16');
  expect(spec.config.planner.profile.base_url).toBe(rows[1].spec.url);
  expect(spec.config.planner.profile.context_limit).toBe(32768);
  expect(spec.episodes).toEqual([7]);
  expect(spec.config.harness.mode).toBe('agentic');
  expect(spec.config.planner.max_turns).toBe(70);
  expect(spec.config.planner.tools).toHaveLength(7);
  expect(spec.config.planner.tools).not.toContain('read_memory');
  expect(spec.config.planner.tools).not.toContain('navigate_relative');
  expect(spec.config.planner.tools).toContain('navigate_to_node');
  await expect(page.getByText('已加入队列，等待环境和 VLA 资源…')).toBeVisible();
  await expect(page).toHaveURL(/\/harness\?job=model-picker-job$/);
  expect(errors).toEqual([]);
});

test('service refresh preserves the selected model and blocks sending when it goes offline', async ({ page, request }) => {
  await page.clock.install();
  const rows = await setup(page, request);
  let posts = 0;
  await page.route('**/api/experiments', async route => {
    if (route.request().method() === 'POST') posts++;
    await route.continue();
  });
  await page.goto('/harness');
  await page.clock.fastForward(200);
  await page.getByRole('button', { name: 'Walk up the stairs.' }).click();
  await expect(page.getByRole('button', { name: '开始运行' })).toBeEnabled();
  rows[0].available = false;
  rows[0].unavailable_reason = '服务暂不可用';
  await page.clock.fastForward(10000);
  await expect(page.getByRole('button', { name: '开始运行' })).toBeDisabled();
  await expect(page.getByRole('button', { name: '选择模型' })).toContainText('Qwen 27B FP4');
  await page.getByLabel('指令', { exact: true }).press('Enter');
  expect(posts).toBe(0);
  await page.getByRole('button', { name: '选择模型' }).click();
  await page.keyboard.press('End');
  await page.keyboard.press('Enter');
  await expect(page.getByRole('button', { name: '选择模型' })).toContainText('Qwen 27B BF16');
  await expect(page.getByRole('button', { name: '开始运行' })).toBeEnabled();
});

test('empty model list and missing actor keep click and Enter from creating a run', async ({ page, request }) => {
  const rows = await setup(page, request);
  await page.route('**/api/harness/services', route => route.fulfill({ json: rows.filter(row => row.spec.kind === 'environment') }));
  await page.goto('/harness');
  await page.getByRole('button', { name: 'Walk up the stairs.' }).click();
  await expect(page.getByRole('button', { name: '开始运行' })).toBeDisabled();
  await page.getByRole('button', { name: '选择模型' }).click();
  await expect(page.getByText('在 Workbench 注册模型后即可选择。')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('menu', { name: '模型列表' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: '选择模型' })).toBeFocused();
});

test('shared VLA selects a matching execution lane and freezes the pool binding', async ({ page, request }) => {
  const rows = await setup(page, request);
  const prefix = `shared-${Date.now()}`;
  const register = async (spec: any) => {
    const response = await request.post('/api/services', { data: spec });
    expect(response.status()).toBe(201);
    const { id } = await response.json();
    const row = { id, spec, status: 'registered', available: true };
    rows.push(row);
    return row;
  };
  const vla = await register({ name: `${prefix}-vla`, kind: 'vla', url: `http://${prefix}:18381`, metadata: { shared_inference: true }, max_concurrency: 4 });
  const lanes = [];
  for (let i = 0; i < 2; i++) {
    const env = await register({ name: `${prefix}-env-${i}`, kind: 'environment', url: `http://${prefix}:1958${i}` });
    const worker = await register({ name: `${prefix}-worker-${i}`, kind: 'execution', url: `http://${prefix}:1968${i}`, metadata: { environment_url: env.spec.url, inference_url: vla.spec.url } });
    lanes.push({ env, worker });
  }
  await page.route('**/api/jobs', route => route.fulfill({ json: { id: 'shared-picker-job' } }));
  await page.route('**/api/jobs/shared-picker-job/attempts', route => route.fulfill({ json: [] }));
  await page.route('**/api/harness/jobs/shared-picker-job', route => route.fulfill({ json: { status: 'queued', waiting_reason: 'queued' } }));
  await page.goto('/harness');
  await page.getByLabel('VLA', { exact: true }).selectOption(vla.id);
  await expect(page.getByLabel('环境', { exact: true })).toHaveValue(lanes[0].env.id);
  await expect(page.getByLabel('执行器', { exact: true })).toHaveValue(lanes[0].worker.id);
  await expect(page.getByLabel('环境', { exact: true }).locator('option')).toHaveCount(2);
  await page.getByLabel('环境', { exact: true }).selectOption(lanes[1].env.id);
  await expect(page.getByLabel('执行器', { exact: true })).toHaveValue(lanes[1].worker.id);
  await page.getByRole('button', { name: 'Walk up the stairs.' }).click();
  await expect(page.getByRole('button', { name: '开始运行' })).toBeEnabled();
  const frozen = page.waitForResponse(response => response.url().endsWith('/api/experiments') && response.request().method() === 'POST');
  await page.getByRole('button', { name: '开始运行' }).click();
  const response = await frozen;
  expect(response.status()).toBe(201);
  const { id } = await response.json();
  const experiments = await (await request.get('/api/experiments')).json();
  const spec = experiments.find((experiment: any) => experiment.id === id).spec;
  expect(spec.environment_service).toBeNull();
  expect(spec.vla_service).toBe(vla.id);
  expect(spec.execution_pool).toHaveLength(1);
  expect(spec.execution_pool[0].environment_service).toBe(lanes[1].env.id);
  expect(spec.execution_pool[0].execution_service).toBe(lanes[1].worker.id);
  expect(spec.config.harness.execution_url).toBe(lanes[1].worker.spec.url);
  expect(spec.config.environment.url).toBe(lanes[1].env.spec.url);
  expect(spec.config.planner.tools).toHaveLength(7);
  expect(spec.config.planner.tools).not.toContain('read_memory');
  expect(spec.config.planner.tools).not.toContain('navigate_relative');
});
