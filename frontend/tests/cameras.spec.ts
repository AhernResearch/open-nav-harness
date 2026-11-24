import {test,expect} from '@playwright/test';

test('camera panel follows observe results, preserves camera identity and opens uncropped images',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 const artifacts=['a','b','c','d'].map(c=>c.repeat(64));
 const directions=['ahead','right','behind','left'];
 const events=[
  {type:'session.started',instruction:'Observe the corridor.',scene_id:'m_floor'},
  {type:'item.started',item:{id:'look-1',type:'tool',name:'observe',arguments:{views:['front','right','back','left']},status:'in_progress'}},
  {type:'item.completed',item:{id:'look-1',status:'completed',images:artifacts,image_views:artifacts.map((artifact,i)=>({artifact,caption:`Tool look-1: ${directions[i]} — clearance 1 m`}))}},
  {type:'item.started',item:{id:'move',type:'motion',name:'execute_subtask',arguments:{instruction:'Move forward'},status:'in_progress'}},
  {type:'item.updated',item:{id:'move',frame:{policy_front:'motion.png'}}},
  {type:'item.completed',item:{id:'move',status:'completed'}},
  {type:'item.started',item:{id:'look-2',type:'tool',name:'observe',arguments:{views:['left']},status:'in_progress'}},
  {type:'item.completed',item:{id:'look-2',status:'completed',images:[artifacts[3]],image_views:[{artifact:artifacts[3],caption:'Tool look-2: left — clearance 2 m'}]}},
 ];
 const body=`id: 10.4\ndata: ${JSON.stringify(events)}\n\nevent: end\ndata: {"status":"completed"}\n\n`;
 await page.route('**/api/runs/r/episodes/cameras/stream**',r=>r.fulfill({headers:{'content-type':'text/event-stream'},body}));
 await page.route('**/api/runs/r/episodes/cameras/artifact/**',r=>{
  const small=r.request().url().endsWith(artifacts[1]);const size=small?128:512;
  return r.fulfill({contentType:'image/svg+xml',body:`<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}"><rect width="100%" height="100%" fill="#9bc"/></svg>`});
 });
 await page.route('**/api/runs/r/episodes/cameras/files/motion.png',r=>r.fulfill({contentType:'image/svg+xml',body:'<svg xmlns="http://www.w3.org/2000/svg" width="480" height="270"/>'}));
 await page.setViewportSize({width:1440,height:900});
 await page.goto('/harness?run=r/cameras');
 const panel=page.getByRole('complementary',{name:'环境'});
 await expect(panel.locator('.camera img')).toHaveCount(1);
 await expect(panel.getByRole('img',{name:'左方视角',exact:true})).toHaveAttribute('src',`/api/runs/r/episodes/cameras/artifact/${artifacts[3]}`);
 await page.getByRole('button',{name:'观察四周',exact:true}).first().click();
 await expect(panel.locator('.camera img')).toHaveCount(4);
 for(let i=0;i<4;i++)await expect(panel.locator('.camera img').nth(i)).toHaveAttribute('src',`/api/runs/r/episodes/cameras/artifact/${artifacts[i]}`);
 await panel.getByRole('button',{name:'放大前方图像',exact:true}).click();
 const dialog=page.getByRole('dialog',{name:'相机图像'});
 await expect(dialog).toBeVisible();
 await expect(dialog.getByRole('img')).toHaveAttribute('src',`/api/runs/r/episodes/cameras/artifact/${artifacts[0]}`);
 await expect(dialog.getByRole('img')).toHaveJSProperty('naturalWidth',512);
 await page.keyboard.press('Escape');await expect(dialog).not.toBeVisible();
 await panel.getByRole('button',{name:'放大右方图像',exact:true}).click();
 await expect(dialog.getByText('这张历史图像分辨率较低，放大无法恢复被标签遮挡的场景。')).toBeVisible();
 await dialog.getByRole('button',{name:'关闭图像'}).click();
 await page.getByRole('button',{name:'Move forward',exact:true}).click();
 await expect(panel.getByRole('img',{name:'前方（动作相机）视角',exact:true})).toHaveAttribute('src','/api/runs/r/episodes/cameras/files/motion.png');
 await panel.getByRole('button',{name:'放大前方（动作相机）图像',exact:true}).click();
 await expect(dialog.getByRole('img')).toHaveJSProperty('naturalWidth',480);
 const ratio=await dialog.getByRole('img').evaluate(im=>{const r=im.getBoundingClientRect();return r.width/r.height;});
 expect(ratio).toBeCloseTo(480/270,2);
 expect(errors).toEqual([]);
});

test('structured route displays the final stop location in the movement row',async({page})=>{
 const events=[
  {type:'session.started',instruction:'Enter the room.',scene_id:'m_floor'},
  {type:'item.started',item:{id:'route',type:'motion',name:'execute_subtask',arguments:{task_id:1,max_distance_m:8,instruction:'Follow the hall and turn left through the doorway.',stop_at:'inside the room beside the table'},status:'in_progress'}},
  {type:'item.completed',item:{id:'route',status:'completed'}},
 ];
 const body=`id: 10.4\ndata: ${JSON.stringify(events)}\n\nevent: end\ndata: {"status":"completed"}\n\n`;
 await page.route('**/api/runs/r/episodes/r2r-stop/stream**',r=>r.fulfill({headers:{'content-type':'text/event-stream'},body}));
 await page.goto('/harness?run=r/r2r-stop');
 await expect(page.getByRole('button',{name:/Follow the hall.*Stop: inside the room beside the table/})).toBeVisible();
});
