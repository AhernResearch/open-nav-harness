import { defineConfig } from 'vite';
export default defineConfig({build: {outDir: '../open_nav_harness/web', emptyOutDir:true},server:{proxy:{'/api':'http://127.0.0.1:8878'}}});
