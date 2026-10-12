// node stills.mjs <entry.tsx> <outDir> <scale> <frame,frame,...> [props.json]
// Bundles once and renders many stills in one browser (the CLI re-bundles per still).
import {bundle} from '@remotion/bundler';
import {renderStill, selectComposition, openBrowser} from '@remotion/renderer';
import fs from 'fs';
import path from 'path';

const [entry, outDir, scale, list, propsFile] = process.argv.slice(2);
const inputProps = propsFile ? JSON.parse(fs.readFileSync(propsFile, 'utf8')) : {};
const here = process.cwd();
const serveUrl = await bundle({
  entryPoint: path.resolve(entry),
  publicDir: path.join(here, 'public'),
  rootDir: here,
  webpackOverride: (c) => ({...c, resolve: {...c.resolve,
    modules: [path.join(here, 'node_modules'), ...(c.resolve?.modules ?? ['node_modules'])],
    alias: {...(c.resolve?.alias ?? {}), '@gen': path.join(here, 'gen')}}}),
});
const browser = await openBrowser('chrome');
const composition = await selectComposition({serveUrl, id: 'episode', inputProps, puppeteerInstance: browser});
for (const f of list.split(',').map(Number)) {
  await renderStill({composition, serveUrl, frame: f, inputProps, output: path.join(outDir, `${String(f).padStart(5, '0')}.png`),
    scale: Number(scale), puppeteerInstance: browser});
}
await browser.close({silent: true});
console.log('stills done', list.split(',').length);
