// Build the note's printable HTML. Requires katex and markdown-it.
// GAMMA_NOTE_NODE_MODULES may point to an external node_modules directory.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const dir = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.join(process.env.GAMMA_NOTE_NODE_MODULES || '/tmp/gamma-note-build/node_modules', '../package.json'));
const katex = require('katex');
const MarkdownIt = require('markdown-it');
const katexDist = path.dirname(require.resolve('katex/dist/katex.min.css'));
let mathCss = fs.readFileSync(path.join(katexDist, 'katex.min.css'), 'utf8');
mathCss = mathCss.replace(/url\(([^)]+)\)/g, (_, raw) => {
  const p = raw.replace(/["']/g, '');
  const mime = p.endsWith('.woff2') ? 'font/woff2' : p.endsWith('.woff') ? 'font/woff' : 'font/ttf';
  return `url(data:${mime};base64,${fs.readFileSync(path.join(katexDist, p)).toString('base64')})`;
});

const sourceName = process.argv[2] || 'gamma_ratio_note.md';
const htmlName = process.argv[3] || 'note.html';
let source = fs.readFileSync(path.join(dir, sourceName), 'utf8');
const math = [];
source = source.replace(/\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)/g, (_, display, inline) => {
  const i = math.length;
  const html = katex.renderToString((display ?? inline).trim(), { displayMode: display !== undefined, throwOnError: true, strict: 'error', output: 'html' });
  math.push({ html, display: display !== undefined });
  const token = `GAMMAMATH${i}END`;
  return display !== undefined ? `\n\n${token}\n\n` : token;
});
let html = new MarkdownIt({ html: true, typographer: true }).render(source);
math.forEach((m, i) => {
  const token = `GAMMAMATH${i}END`;
  html = m.display
    ? html.replace(`<p>${token}</p>`, `<div class="equation">${m.html}</div>`)
    : html.replaceAll(token, m.html);
});
if (/GAMMAMATH\d+END/.test(html)) {
  throw new Error('Unrendered mathematical placeholder remains in HTML.');
}
html = html.replace(/<p><img src="([^"]+)" alt="([^"]*)"><\/p>\s*<p><em>([\s\S]*?)<\/em><\/p>/g, (_, src, alt, caption) => {
  const img = fs.readFileSync(path.resolve(dir, decodeURI(src)));
  return `<figure><img src="data:image/png;base64,${img.toString('base64')}" alt="${alt}"><figcaption>${caption}</figcaption></figure>`;
});
html = html.replace(/href="([^"#][^"]*)"/g, (_, target) => {
  if (/^\w+:/.test(target)) return `href="${target}"`;
  return `href="file://${path.resolve(dir, target)}"`;
});
html = html.replace(/<h2>(Appendix[^<]*)<\/h2>/g, '<h2 class="appendix">$1</h2>');
html = html.replace('<h2 class="appendix">Appendix A.', '<h2 class="appendix appendix-start">Appendix A.');
html = html.replace(/<p><strong>Theorem 1/g, '<p class="theorem-title"><strong>Theorem 1');
const compact = sourceName.includes('compact');
if (compact) {
  html = html.replace(/<p><em>Compact argument[^<]*<\/em><\/p>/, '');
  const appendixStart = html.indexOf('<h2 class="appendix appendix-start">');
  if (appendixStart < 0) throw new Error('Compact note must include Appendix A.');
  html = '<section class="compact-main">' + html.slice(0, appendixStart) + '</section>' + html.slice(appendixStart);
}
const css = `
@page { size: A4; margin: 17mm 17mm 17mm; @bottom-right { content: counter(page); font: 9pt Arial; color: #777; } }
* { box-sizing: border-box; }
body { margin: 0 auto; max-width: 176mm; color: #19232b; font: 10.7pt/1.42 Georgia, 'Times New Roman', serif; }
h1,h2,h3 { font-family: Arial, Helvetica, sans-serif; color: #163342; }
h1 { font-size: 24pt; line-height: 1.15; letter-spacing: -.4px; margin: 0 0 12pt; }
h2 { font-size: 15pt; line-height: 1.23; margin: 25pt 0 11pt; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 18pt 0 8pt; break-after: avoid; }
h2.appendix { border-top: 1px solid #abc0c8; padding-top: 12pt; }
h2.appendix-start { break-before: page; }
p { margin: 8pt 0; orphans: 3; widows: 3; break-inside: avoid; }
h1 + p { color: #52646e; font: 10pt/1.4 Arial, sans-serif; margin-bottom: 18pt; }
strong { font-weight: 700; }
.equation { break-inside: avoid; padding: 0; margin: 9pt 0; }
.katex { font-size: 1.03em; color: #17232d; }
.katex-display { margin: .7em 0; overflow: visible; }
.katex-display > .katex { white-space: nowrap; }
figure { margin: 17pt 0; break-inside: avoid; }
figure img { display: block; width: 100%; height: auto; }
figcaption { font: 9pt/1.38 Arial, Helvetica, sans-serif; color: #40515b; margin-top: 8pt; }
table { width: 100%; border-collapse: collapse; margin: 13pt 0; font: 9.3pt/1.35 Arial, sans-serif; break-inside: avoid; }
thead { display: table-header-group; }
th { background: #edf2f4; color: #193746; border-bottom: 1px solid #99adb8; }
th,td { padding: 6pt 5pt; text-align: left; vertical-align: top; }
td { border-bottom: 1px solid #dfe6ea; }
th .katex,td .katex { font-size: 1.02em; }
ul { margin: 9pt 0; padding-left: 18pt; }
li { margin: 5pt 0; overflow-wrap: anywhere; }
code { font: 8.3pt 'Courier New', monospace; overflow-wrap: anywhere; }
a { color: #235c7d; text-decoration: none; overflow-wrap: anywhere; }
.theorem-title { border-left: 3px solid #568b9b; padding-left: 9pt; }
.compact-main { font-size: 10pt; line-height: 1.27; }
.compact-main h1 { font-size: 18pt; margin-bottom: 6pt; }
.compact-main h1 + p { color: inherit; font: inherit; margin: 6pt 0; }
.compact-main p { margin: 6pt 0; }
.compact-main .equation { margin: 7pt 0; }
.compact-main .katex-display { margin: .45em 0; }
.compact-main .theorem-title { break-after: avoid; }
.compact-main figure { margin: 4pt 0 0; }
.compact-main figure img { width: 94%; margin: 0 auto; }
.compact-main figcaption { font-size: 8.3pt; line-height: 1.28; margin-top: 3pt; }
@media screen { body { padding: 35px 0; } }
`;
const output = `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Why small tanh slopes make readout gradient descent slow</title><style>${mathCss}\n${css}</style></head><body>${html}</body></html>`;
fs.mkdirSync(path.join(dir, '.build'), { recursive: true });
fs.writeFileSync(path.join(dir, '.build', htmlName), output);
console.log(`Rendered ${math.length} equations to ${path.join(dir, '.build', htmlName)}`);
