import {defineConfig} from '@playwright/test';
import {mkdtempSync} from 'node:fs';
const state=process.env.NAVH_E2E_STATE||mkdtempSync('/tmp/navh-ui-');
process.env.NAVH_E2E_STATE=state;
export default defineConfig({
 testDir:'./tests',workers:1,fullyParallel:false,
 use:{baseURL:'http://127.0.0.1:18878',headless:true,launchOptions:process.env.NAVH_CHROME?{executablePath:process.env.NAVH_CHROME}:{}},
 webServer:{command:`.venv/bin/python -m nav_harness.cli --state ${state} serve --port 18878`,cwd:'..',url:'http://127.0.0.1:18878/health',timeout:30000,reuseExistingServer:false,env:{PYTHONPATH:process.cwd()+'/..'}},
});
