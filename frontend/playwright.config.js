import {defineConfig} from '@playwright/test';
import {mkdtempSync} from 'node:fs';
const state=process.env.ONH_E2E_STATE||mkdtempSync('/tmp/onh-ui-');
process.env.ONH_E2E_STATE=state;
export default defineConfig({
 testDir:'./tests',workers:1,fullyParallel:false,
 use:{baseURL:'http://127.0.0.1:18878',headless:true,launchOptions:process.env.ONH_CHROME?{executablePath:process.env.ONH_CHROME}:{}},
 webServer:{command:`.venv/bin/python -m open_nav_harness.cli --state ${state} serve --port 18878`,cwd:'..',url:'http://127.0.0.1:18878/health',timeout:30000,reuseExistingServer:false,env:{PYTHONPATH:process.cwd()+'/..'}},
});
