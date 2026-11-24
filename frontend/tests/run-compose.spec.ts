import {test, expect} from '@playwright/test';

test('a run is composed from server-side names and opens on its first episode', async ({page}) => {
  let composed: any = null;
  await page.route('**/api/conversations/options', route => route.fulfill({json:{models:['sol','27b'],default_model:'sol',mcp:[],
    navigation:[{name:'sim',dataset:'r2r_train',camera:'visible_88cm_hfov120'}]}}));
  await page.route('**/api/catalog', route => route.fulfill({json:[{name:'r2r_train',dataset:'r2r',split:'train',
    scenes:[{scene:'S1',count:2},{scene:'S2',count:1}]}]}));
  await page.route('**/api/catalog/r2r_train/episodes?**', route => {
    const scene = new URL(route.request().url()).searchParams.get('scene');
    route.fulfill({json: scene === 'S1'
      ? [{index:898,scene:'S1',instruction:'Exit the bathroom and enter the hall.'},
         {index:3473,scene:'S1',instruction:'Turn right and find the fireplace.'}]
      : [{index:4956,scene:'S2',instruction:'Walk to the door on the right.'}]});
  });
  await page.route('**/api/runs/compose', route => (composed = route.request().postDataJSON(),
    route.fulfill({status:201,json:{id:'r1'}})));
  await page.route('**/api/runs/r1/episodes/898/stream**', route => route.fulfill({contentType:'text/event-stream',body:''}));
  await page.goto('/harness');
  await page.getByRole('button',{name:'从数据集运行'}).click();
  await expect(page.getByRole('heading',{name:'新建运行'})).toBeVisible();
  const start = page.getByRole('button',{name:'开始运行'});
  await expect(start).toBeDisabled();
  await page.getByLabel('planner').selectOption('27b');
  await page.getByRole('group',{name:'episode'}).getByText('#898').click();
  // Choices survive a scene change, so one run can span scenes.
  await page.getByLabel('场景').selectOption('S2');
  await page.getByRole('group',{name:'episode'}).getByText('#4956').click();
  await page.getByLabel('lanes').fill('2');
  await page.getByLabel('运行名称').fill('ui smoke');
  await start.click();
  await expect.poll(() => composed).not.toBeNull();
  expect(composed).toEqual({name:'ui smoke',purpose:'evaluation',planner:'27b',environment:'sim',
    episodes:['898','4956'],lanes:2});
  await expect(page).toHaveURL(/run=r1%2F898/);
});
