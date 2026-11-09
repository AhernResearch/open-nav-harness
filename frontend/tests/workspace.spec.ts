import {test,expect} from '@playwright/test';

test('open workspace, register services, freeze experiment and queue',async({page,request})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/workbench');
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
 const frozen=(await (await request.get('/api/experiments')).json()).find((e:any)=>e.spec.name==='Local Planner comparison');
 expect(frozen.spec.config.planner.tools).toHaveLength(7);
 expect(frozen.spec.config.planner.tools).not.toContain('navigate_relative');
 await page.locator('article').filter({has:page.getByRole('heading',{name:'Local Planner comparison',exact:true})}).getByRole('button',{name:'加入队列'}).click();
 await expect(page.getByRole('button',{name:'Local Planner comparison'})).toBeVisible();
 await expect(page.getByText('0 / 3')).toBeVisible();
 await page.getByRole('button',{name:'完成在途路线后暂停'}).click();
 await page.getByRole('button',{name:'05 结果比较'}).click();
 await page.getByRole('checkbox').check();
 await page.getByRole('button',{name:'配对查看'}).click();
 await expect(page.getByText('完成 0 条',{exact:false})).toBeVisible();
 await page.setViewportSize({width:390,height:844});
 await expect(page.getByRole('heading',{name:'结果比较',exact:true})).toBeVisible();
 expect(errors).toEqual([]);
 expect((await request.get('/api/jobs')).status()).toBe(200);
 await expect(page.getByLabel('管理口令')).toHaveCount(0);
});

test('trace deep link opens without login and shows model text, calls, rejection and VLA output', async ({page}) => {
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
 await page.goto('/workbench?attempt=trace-test');
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

test('shared inference experiment selects independent execution lanes and shows concurrency', async ({page,request}) => {
 const register=async(body:any)=>(await (await request.post('/api/services',{data:body})).json()).id;
 const planner=await register({name:'pool-planner',kind:'planner',url:'http://127.0.0.1:18180/v1',planner:{model:'test',base_url:'http://127.0.0.1:18180/v1',max_concurrency:4}});
 const vla=await register({name:'shared-lightnav',kind:'vla',url:'http://127.0.0.1:18380',max_concurrency:4,metadata:{shared_inference:true}});
 for(let i=0;i<2;i++){
  await register({name:`pool-env-${i}`,kind:'environment',url:`http://127.0.0.1:${19580+i}`});
  await register({name:`pool-worker-${i}`,kind:'execution',url:`http://127.0.0.1:${19680+i}`,metadata:{environment_url:`http://127.0.0.1:${19580+i}`,inference_url:'http://127.0.0.1:18380'}});
 }
 await page.route('**/api/inference/status',route=>route.fulfill({json:[{service_id:vla,name:'shared-lightnav',ok:true,details:{sessions:4,pending_requests:3,waiting_requests:2}}]}));
 await page.goto('/workbench');
 await expect(page.getByLabel('共享推理状态')).toContainText('会话 4 · 在途 3 · 等待 2');
 await page.getByRole('button',{name:'03 实验配置'}).click();
 await page.getByLabel('实验名称').fill('shared-pool-test');
 await page.getByLabel('Planner',{exact:true}).selectOption(planner);
 await page.getByLabel('VLA',{exact:true}).selectOption(vla);
 await expect(page.getByLabel('Environment',{exact:true})).toHaveCount(0);
 await page.getByLabel('pool-worker-0').check();
 await page.getByLabel('pool-worker-1').check();
 await page.getByRole('button',{name:'校验并冻结配置'}).click();
 await expect(page.getByText('实验配置已冻结，可以加入队列。')).toBeVisible();
 const experiments=await (await request.get('/api/experiments')).json();
 const frozen=experiments.find((e:any)=>e.spec.name==='shared-pool-test');
 expect(frozen.spec.execution_pool).toHaveLength(2);
 expect(frozen.spec.environment_service).toBeNull();
});

test('clicking an experiment opens records in the viewport and returns to the list', async ({page}) => {
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 const jobs=Array.from({length:12},(_,i)=>({id:`job-${i}`,name:`Navigation run ${i}`,status:'running',completed:1,total:50,active:7,pause_requested:1}));
 await page.route('**/api/jobs',route=>route.fulfill({json:jobs}));
 let release!:()=>void;
 const gate=new Promise<void>(resolve=>{release=resolve;});
 await page.route('**/api/jobs/job-0/attempts',async route=>{
  await gate;
  await route.fulfill({json:[{attempt_id:'nav-attempt',episode:7648,attempt:1,status:'completed',success:1,ne:1.25,spl:1,tool_calls:11,wall_s:150}]});
 });
 await page.route('**/api/jobs/job-1/attempts',route=>route.fulfill({status:502,json:{detail:'temporary failure'}}));
 await page.route('**/api/attempts/nav-attempt',route=>route.fulfill({json:{id:'nav-attempt',episode:7648,ordinal:1,status:'completed',result:{metrics:{}},requests:[],events:[],subtask_events:[]}}));
 await page.setViewportSize({width:1280,height:800});
 await page.goto('/workbench');
 await page.getByRole('button',{name:'Navigation run 0',exact:true}).click();
 await expect(page.getByRole('heading',{name:'Navigation run 0 · 逐条记录'})).toBeInViewport();
 await expect(page.getByRole('status')).toHaveText('正在加载执行记录…');
 release();
 await expect(page.getByRole('button',{name:'7648 / 1',exact:true})).toBeInViewport();
 await expect(page.getByLabel('实验运行详情')).toContainText('当前路线结束后暂停');
 await page.getByRole('button',{name:'返回实验列表'}).click();
 await page.getByRole('button',{name:'Navigation run 1',exact:true}).click();
 await expect(page.getByRole('alert')).toContainText('temporary failure');
 await expect(page.getByRole('button',{name:'7648 / 1',exact:true})).toHaveCount(0);
 await page.getByRole('button',{name:'返回实验列表'}).click();
 await page.getByRole('button',{name:'Navigation run 0',exact:true}).click();
 await page.getByRole('button',{name:'7648 / 1',exact:true}).click();
 await expect(page.getByRole('heading',{name:'Episode 7648 / Attempt 1'})).toBeVisible();
 expect(errors).toEqual([]);
});

const sse=(events:object[])=>`id: 10.4\ndata: ${JSON.stringify(events)}\n\nevent: end\ndata: {"status":"completed","job_id":"j","job_status":"completed"}\n\n`;

test('legacy links open the Harness and the live view shows plan, tool rows, memory and map', async ({page}) => {
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 let streams=0;
 const state={tasks:[{id:1,instruction:'Climb the stairs',status:'completed'},{id:2,instruction:'Turn left at the landing',status:'in_progress'}]};
 const events=[
  {type:'session.started',instruction:'Walk up the stairs and turn left.',scene_id:'mp3d/S1/S1.glb'},
  {type:'turn.completed',t:1,text:'I will climb first.',usage:null,elapsed_s:1},
  {type:'item.started',t:2,item:{id:'m',type:'motion',name:'execute_subtask',arguments:{instruction:'Climb the stairs'},status:'in_progress'}},
  {type:'item.updated',item:{id:'m',pose:{position:[1,0,0],heading_rad:0}}},
  {type:'item.completed',t:3,item:{id:'m',status:'completed',result:['ok'],state,images:[]}},
  {type:'item.started',t:4,item:{id:'w',type:'memory_write',name:'update_memory',arguments:{start_node_id:0,end_node_id:1,caption:'Reached the landing.'},status:'in_progress'}},
  {type:'item.completed',t:5,item:{id:'w',status:'completed',result:[],state,images:[]}},
  {type:'item.started',t:6,item:{id:'p',type:'map',name:'observe_map',arguments:{},status:'in_progress'}},
  {type:'item.completed',t:7,item:{id:'p',status:'completed',result:[],state,images:[]}},
  {type:'item.updated',item:{id:'p',map:{artifact:'a'.repeat(64),size:[400,300],places_px:{'0':[100,100],'1':[150,100]},transform:{matrix:[50,0,100,0,50,100],ppm:50,residual_px:0,nodes:2}}}},
  {type:'attempt.status',status:'completed',job_id:'j',job_status:'completed'},
 ];
 await page.route('**/api/attempts/h-test/stream**',route=>{streams++;return route.fulfill({status:200,headers:{'content-type':'text/event-stream'},body:sse(events)});});
 await page.route('**/api/attempts/h-test/artifact/**',route=>route.fulfill({status:404}));
 await page.goto('/?attempt=h-test');
 await expect(page).toHaveURL(/\/harness\?attempt=h-test$/);
 await expect(page.getByText('Walk up the stairs and turn left.')).toBeVisible();
 await expect(page.getByText('1/2')).toBeVisible();
 await expect(page.getByText('Turn left at the landing')).toBeVisible();
 await expect(page.getByRole('button',{name:/记住 0 → 1/})).toBeVisible();
 await expect(page.getByText('Reached the landing.')).toBeVisible();
 await expect(page.getByRole('img',{name:'记忆地图'})).toBeVisible();
 await expect(page.locator('.overlay .pose')).toHaveCount(1);
 await page.waitForTimeout(1500);
 expect(streams).toBe(1);
 await page.getByRole('button',{name:/Climb the stairs/}).first().click();
 await expect(page.getByRole('button',{name:'回到实时'})).toBeVisible();
 expect(errors).toEqual([]);
});

test('reduced motion and an empty composer render without errors', async ({page}) => {
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.emulateMedia({reducedMotion:'reduce'});
 await page.goto('/harness');
 await expect(page.getByRole('heading',{name:'今天让机器人去哪里？'})).toBeVisible();
 await expect(page.getByRole('button',{name:'开始运行'})).toBeDisabled();
 await page.getByRole('link',{name:'Workbench'}).click();
 await expect(page.getByRole('heading',{name:'任务队列',exact:true})).toBeVisible();
 expect(errors).toEqual([]);
});
