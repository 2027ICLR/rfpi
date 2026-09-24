// Build index.html from src/index.template.html:
//  - pre-renders all \( .. \) and $$ .. $$ math with KaTeX (no runtime JS / CDN needed)
//  - inlines static/data/q_values.json
// Usage:  npm i katex && node tools/build.js
const fs = require('fs'), path = require('path'), katex = require('katex');
const root = path.join(__dirname, '..');
let html = fs.readFileSync(path.join(root, 'src/index.template.html'), 'utf8');
const parts = html.split(/(<script[\s\S]*?<\/script>|<style[\s\S]*?<\/style>)/);
let n = 0;
const r = (tex, display) => { n++; return katex.renderToString(tex.replace(/&amp;/g, '&'), { displayMode: display, throwOnError: true, output: 'html', strict: 'ignore' }); };
for (let i = 0; i < parts.length; i += 2) {
  parts[i] = parts[i]
    .replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => r(t, true))
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, t) => r(t, false));
}
html = parts.join('');
html = html.replace('/*QDATA*/', fs.readFileSync(path.join(root, 'static/data/q_values.json'), 'utf8').trim());
fs.writeFileSync(path.join(root, 'index.html'), html);
console.log(`rendered ${n} formulas -> index.html (${(html.length / 1024).toFixed(0)} KB)`);
