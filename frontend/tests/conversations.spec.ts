import {test, expect} from '@playwright/test';

test('free instructions continue inside the existing Harness layout and survive reload', async ({page}) => {
  const events: any[] = [];
  const spec = {model:'Sonnet',mcp:['map'],navigation:null};
  await page.route('**/api/conversations/options', route => route.fulfill({json:{models:['Sonnet'],default_model:'Sonnet',mcp:['map'],navigation:[]}}));
  await page.route('**/api/conversations', route => route.request().method()==='POST' ? route.fulfill({json:{id:'dialogue-fixture'}}) : route.fulfill({json:events.length ? [{id:'dialogue-fixture',spec,status:'idle',title:'你好'}] : []}));
  await page.route('**/api/conversations/dialogue-fixture', route => route.fulfill({json:{id:'dialogue-fixture',spec,status:'idle',events}}));
  await page.route('**/api/conversations/dialogue-fixture/events?**', route => route.fulfill({contentType:'text/event-stream',body:events.map(event => `id: ${event.seq}\ndata: ${JSON.stringify(event)}\n\n`).join('')}));
  await page.route('**/api/conversations/dialogue-fixture/messages', async route => {
    const content = route.request().postDataJSON().content;
    events.push({seq:events.length+1,kind:'user_text',payload:{text:content}});
    if(content.includes('地图')) {
      events.push({seq:events.length+1,kind:'tool_use',payload:{id:'search',name:'catalog__lookup',input:{instruction:content}}});
      events.push({seq:events.length+1,kind:'tool_result',payload:{tool_use_id:'search',texts:['会议室 R01'],images:[],is_error:false}});
    }
    events.push({seq:events.length+1,kind:'assistant_text',payload:{text:content.includes('地图')?'可以去会议室。':'你好，我在。'}});
    events.push({seq:events.length+1,kind:'status',payload:{status:'idle'}});
    await route.fulfill({json:{request_id:route.request().postDataJSON().request_id}});
  });
  await page.goto('/harness');
  await expect(page.locator('.hx-side')).toBeVisible();
  await page.getByRole('textbox',{name:'指令',exact:true}).fill('你好');
  await page.getByRole('button',{name:'发送指令'}).click();
  await expect(page.locator('.thread .agent-msg')).toContainText('你好，我在。');
  await expect(page).toHaveURL(/\/harness\?conversation=dialogue-fixture/);
  await page.getByRole('textbox',{name:'指令',exact:true}).fill('查一下地图');
  await page.getByRole('button',{name:'发送指令'}).click();
  // Reconnect the fixture stream to publish the second turn.
  await page.reload();
  await expect(page.locator('.agent-msg').last()).toHaveText('可以去会议室。');
  await expect(page.locator('.tool-row.completed')).toContainText('catalog__lookup');
  await expect(page.locator('.conversation-input .composer')).toBeVisible();
  await expect(page.locator('.hx-panel')).toBeVisible();
  await page.getByRole('button',{name:'新的运行',exact:true}).click();
  await expect(page.getByRole('heading',{name:'今天让机器人去哪里？'})).toBeVisible();
  await page.getByRole('button',{name:'从数据集运行'}).click();
  await expect(page.getByPlaceholder('搜索这个场景里的指令，例如 “stairs” 或 “kitchen”…')).toBeVisible();
});
