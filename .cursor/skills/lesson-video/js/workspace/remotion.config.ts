// Sources live in the repo (.cursor/skills/lesson-video/js and */video/*); this folder only holds deps,
// generated data (gen/) and media (public/). Resolve bare imports from here so repo files need no node_modules.
import {Config} from '@remotion/cli/config';
import path from 'path';

const here = process.cwd();
Config.overrideWebpackConfig((c) => ({
  ...c,
  resolve: {
    ...c.resolve,
    modules: [path.join(here, 'node_modules'), ...(c.resolve?.modules ?? ['node_modules'])],
    alias: {...(c.resolve?.alias ?? {}), '@gen': path.join(here, 'gen')},
  },
}));
Config.setVideoImageFormat('jpeg');
Config.setJpegQuality(92);
