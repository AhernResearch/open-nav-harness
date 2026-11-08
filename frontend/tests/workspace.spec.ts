import {test,expect} from '@playwright/test';
import {readFileSync} from 'node:fs';

test('login, register services, freeze experiment, queue and logout',async({page,request})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');
 await expect(page.getByRole('heading',{name:'Nav Harness',exact:true})).toBeVisible();
 await page.getByLabel('管理口令').fill(readFileSync(process.env.NAVH_E2E_STATE+'/admin-password','utf8'));
 await page.getByRole('button',{name:'进入工作台'}).click();
 await expect(page.getByRole('heading',{name:'任务队列',exact:true})).toBeVisible();
 await page.getByRole('button',{name:'02 模型与服务'}).click();
 await page.getByLabel('名称',{exact:true}).fill('test-planner');
 await page.getByRole('button',{name:'注册服务',exact:true}).click();
 await expect(page.getByRole('heading',{name:'test-planner'})).toBeVisible();
 await page.getByLabel('名称',{exact:true}).fill('test-env');
 await page.getByLabel('类型',{exact:true}).selectOption('environment');
 await page.getByLabel('API 地址',{exact:true}).fill('http://127.0.0.1:19200');
 await page.getByRole('button',{name:'注册服务',exact:true}).click();
 await expect(page.getByRole('heading',{name:'test-env'})).toBeVisible();
 await page.getByLabel('名称',{exact:true}).fill('test-vla');
 await page.getByLabel('类型',{exact:true}).selectOption('vla');
 await page.getByLabel('API 地址',{exact:true}).fill('http://127.0.0.1:18200');
 await page.getByRole('button',{name:'注册服务',exact:true}).click();
 await expect(page.getByRole('heading',{name:'test-vla'})).toBeVisible();
 await page.getByRole('button',{name:'03 实验配置'}).click();
 await expect(page.getByLabel('模式',{exact:true})).toHaveValue('agentic');
 await page.getByLabel('Planner',{exact:true}).selectOption({label:'test-planner'});
 await page.getByLabel('Environment',{exact:true}).selectOption({label:'test-env'});
 await page.getByLabel('VLA',{exact:true}).selectOption({label:'test-vla'});
 await page.getByRole('button',{name:'校验并冻结配置'}).click();
 await expect(page.getByText('实验配置已冻结，可以加入队列。')).toBeVisible();
 await page.getByRole('button',{name:'加入队列'}).click();
 await expect(page.getByRole('button',{name:'Local Planner comparison'})).toBeVisible();
 await expect(page.getByText('0 / 3')).toBeVisible();
 await page.getByRole('button',{name:'完成当前条后暂停'}).click();
 await page.getByRole('button',{name:'05 结果比较'}).click();
 await page.getByRole('checkbox').check();
 await page.getByRole('button',{name:'配对查看'}).click();
 await expect(page.getByText('完成 0 条',{exact:false})).toBeVisible();
 await page.setViewportSize({width:390,height:844});
 await expect(page.getByRole('heading',{name:'结果比较',exact:true})).toBeVisible();
 await page.getByRole('button',{name:'退出',exact:true}).click();
 await expect(page.getByLabel('管理口令')).toBeVisible();
 expect(errors).toEqual([]);
 expect((await request.get('/api/jobs')).status()).toBe(401);
});

test('trace deep link survives login and shows model text, calls, rejection and VLA output', async ({page}) => {
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 const call={id:'call-1',function:{name:'update_memory',arguments:JSON.stringify({start_node_id:0,end_node_id:1,caption:'Crossed the doorway.'})}};
 const detail={id:'trace-test',episode:0,ordinal:1,status:'completed',output:'/run',result:{metrics:{success:1}},events:[],
  requests:[
   {seq:1,kind:'model_request',request:{messages:[{role:'user',content:'Cross the doorway'}]}},
   {seq:2,kind:'model_response',elapsed_s:1.5,response:{choices:[{message:{content:'I crossed the doorway and will record the passage.',tool_calls:[call]}}]}},
   {seq:3,kind:'tool_dispatch',call_id:'call-1',name:'update_memory',arguments:JSON.parse(call.function.arguments)},
   {seq:4,kind:'tool_result',call_id:'call-1',name:'update_memory',is_error:true,feedback:{content:JSON.stringify({protocol_error:'Call observe_map first'})+'\n'+JSON.stringify({subtask_state:{revision:1,tasks:[{id:1,instruction:'Cross the doorway',status:'in_progress'}]}})},images:[]}
  ],subtask_events:[
   {event:1,kind:'policy_context',prediction_id:1,context_key:'task-1',instruction:'Cross the doorway',frame_ids:[]},
   {event:2,kind:'policy_prediction',prediction_id:1,advice:{navigation:[[0.2,0,0]]}},
   {event:3,kind:'policy_execution_result',prediction_id:1,feedback:{step_count:1}}
  ]};
 await page.route('**/api/attempts/trace-test',route=>route.fulfill({json:detail}));
 await page.goto('/?attempt=trace-test');
 await page.getByLabel('管理口令').fill(readFileSync(process.env.NAVH_E2E_STATE+'/admin-password','utf8'));
 await page.getByRole('button',{name:'进入工作台'}).click();
 await expect(page.getByRole('heading',{name:'完整流程 · 1 次模型响应'})).toBeVisible();
 await expect(page.getByText('I crossed the doorway and will record the passage.',{exact:true})).toBeVisible();
 await expect(page.getByRole('heading',{name:'update_memory · 返回错误'})).toBeVisible();
 await expect(page.getByText('Call observe_map first',{exact:false})).toBeVisible();
 await page.getByText('模型输出：原始轨迹点与动作',{exact:true}).click();
 await expect(page.getByText('"navigation"',{exact:false})).toBeVisible();
 await page.getByRole('button',{name:'逐事件检查',exact:true}).click();
 await page.getByRole('button',{name:/003tool_dispatch/}).click();
 await expect(page.getByRole('heading',{name:'tool_dispatch',exact:true})).toBeVisible();
 await page.getByRole('button',{name:/004tool_result/}).click();
 await expect(page.getByRole('heading',{name:'该时刻的计划 · r1'})).toBeVisible();
 await page.reload();
 await expect(page.getByRole('heading',{name:'完整流程 · 1 次模型响应'})).toBeVisible();
 expect(errors).toEqual([]);
});
