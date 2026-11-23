import {test, expect} from '@playwright/test';

test('a conversation binds one dataset episode and keeps it in the created spec', async ({page}) => {
  let created: any = null;
  await page.route('**/api/conversations/options', route => route.fulfill({json:{models:['Sonnet'],default_model:'Sonnet',mcp:[],
    navigation:[{name:'sim',dataset:'r2r_train',camera:'visible_88cm_hfov120'}]}}));
  await page.route('**/api/catalog', route => route.fulfill({json:[{name:'r2r_train',dataset:'r2r',split:'train',scenes:[{scene:'S1',count:2}]}]}));
  await page.route('**/api/catalog/r2r_train/episodes?**', route => route.fulfill({json:[
    {index:898,episode_id:898,scene:'S1',instruction:'Exit the bathroom and enter the hall.'},
    {index:3473,episode_id:3473,scene:'S1',instruction:'Turn right and find the fireplace.'}]}));
  await page.route('**/api/conversations', route => route.request().method()==='POST'
    ? (created = route.request().postDataJSON(), route.fulfill({json:{id:'bound'}}))
    : route.fulfill({json:[]}));
  await page.route('**/api/conversations/bound', route => route.fulfill({json:{id:'bound',events:[],
    spec:{model:'Sonnet',mcp:[],navigation:'sim',episode:'898'},status:'idle'}}));
  await page.route('**/api/conversations/bound/events?**', route => route.fulfill({contentType:'text/event-stream',body:''}));
  await page.route('**/api/conversations/bound/messages', route => route.fulfill({json:{request_id:'r'}}));
  await page.goto('/harness');
  await page.getByLabel('环境').selectOption('sim');
  await expect(page.getByLabel('场景')).toHaveValue('S1');
  await expect(page.getByLabel('episode')).toContainText('#3473 Turn right');
  await page.getByLabel('episode').selectOption('898');
  await expect(page.locator('.composer')).toContainText('#898');
  await page.getByRole('textbox',{name:'指令',exact:true}).fill('走到浴室门口');
  await page.getByRole('button',{name:'发送指令'}).click();
  await expect.poll(() => created).not.toBeNull();
  expect(created.navigation).toBe('sim');
  expect(created.episode).toBe('898');
});
