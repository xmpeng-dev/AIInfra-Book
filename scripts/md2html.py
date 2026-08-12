#!/usr/bin/env python3
"""把 slab 的 markdown 笔记渲染成单文件 HTML（样式内联，可离线打开）。

用法：
    python3 scripts/md2html.py papers/hipkittens-zh.md
    python3 scripts/md2html.py papers/hipkittens-zh.md -o /tmp/out.html

相比编辑器内置预览，这里额外支持脚注、行内公式，并针对中英混排调了排版。
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from markdown_it import MarkdownIt
from mdit_py_plugins.anchors import anchors_plugin
from mdit_py_plugins.footnote import footnote_plugin
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

# KaTeX 太重且需要联网取字体，笔记里的公式都很简单，直接转成 HTML 排版。
_TEX = {
    r"\times": "×", r"\cdot": "·", r"\approx": "≈", r"\neq": "≠",
    r"\leq": "≤", r"\geq": "≥", r"\pm": "±", r"\to": "→", r"\ll": "≪",
}


def render_math(tex: str) -> str:
    tag = ""
    m = re.search(r"\\tag\{([^}]*)\}", tex)
    if m:
        tag = m.group(1)
        tex = tex[: m.start()] + tex[m.end():]
    tex = re.sub(r"\\text\{([^}]*)\}", r"\1", tex)
    for k, v in _TEX.items():
        tex = tex.replace(k, v)
    tex = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1) / (\2)", tex)
    tex = re.sub(r"\\[a-zA-Z]+", "", tex).strip()
    body = f'<span class="math-body">{html.escape(tex)}</span>'
    if tag:
        body += f'<span class="math-tag">({html.escape(tag)})</span>'
    return f'<div class="math-block">{body}</div>\n'


def extract_math(src: str) -> tuple[str, list[str]]:
    """先把 $$...$$ 摘出来占位，避免 markdown 把里面的下划线当强调。"""
    blocks: list[str] = []

    def sub(m: re.Match) -> str:
        blocks.append(render_math(m.group(1)))
        return f"\n\nMATHBLOCK{len(blocks) - 1}ENDMATH\n\n"

    return re.sub(r"\$\$(.+?)\$\$", sub, src, flags=re.S), blocks


def make_highlighter():
    formatter = HtmlFormatter(nowrap=True)

    def cb(code: str, lang: str, _attrs: str) -> str:
        try:
            lexer = get_lexer_by_name(lang) if lang else None
        except ClassNotFound:
            lexer = None
        inner = highlight(code, lexer, formatter) if lexer else html.escape(code)
        cls = f' data-lang="{html.escape(lang)}"' if lang else ""
        return f"<pre class='code'{cls}><code>{inner}</code></pre>"

    return cb


def build_toc(tokens) -> str:
    items = []
    for i, tok in enumerate(tokens):
        if tok.type != "heading_open" or tok.tag not in ("h2", "h3"):
            continue
        text = tokens[i + 1].content
        anchor = tok.attrGet("id") or ""
        items.append((tok.tag, text, anchor))
    if not items:
        return ""
    out = ['<nav class="toc"><div class="toc-title">目录</div><ul>']
    for tag, text, anchor in items:
        out.append(f'<li class="{tag}"><a href="#{html.escape(anchor)}">{html.escape(text)}</a></li>')
    out.append("</ul></nav>")
    return "".join(out)


CSS = """
:root{--fg:#1f2328;--muted:#59636e;--bg:#fff;--line:#d8dee4;--accent:#0969da;
--code-bg:#f6f8fa;--mark:#fff8c5;--th:#f6f8fa;--zebra:#fcfcfd}
@media (prefers-color-scheme:dark){:root{--fg:#e6edf3;--muted:#9198a1;--bg:#0d1117;
--line:#30363d;--accent:#4493f8;--code-bg:#161b22;--mark:#3f2e00;--th:#161b22;--zebra:#11151b}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans CJK SC","Source Han Sans SC",
"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
line-height:1.85;font-size:16px;-webkit-font-smoothing:antialiased}
.wrap{display:flex;gap:40px;max-width:1240px;margin:0 auto;padding:40px 24px 120px}
main{flex:1;min-width:0;max-width:860px}
.toc{position:sticky;top:40px;align-self:flex-start;width:230px;flex-shrink:0;
max-height:calc(100vh - 80px);overflow-y:auto;font-size:13px;line-height:1.6;
border-left:1px solid var(--line);padding-left:16px}
.toc-title{font-weight:700;margin-bottom:10px;font-size:12px;letter-spacing:.08em;
text-transform:uppercase;color:var(--muted)}
.toc ul{list-style:none;margin:0;padding:0}
.toc li{margin:5px 0}
.toc li.h3{padding-left:14px;font-size:12.5px}
.toc a{color:var(--muted);text-decoration:none}
.toc a:hover{color:var(--accent)}
@media(max-width:1040px){.toc{display:none}.wrap{padding:24px 18px 80px}}
h1,h2,h3,h4{line-height:1.4;font-weight:700;margin:1.9em 0 .7em}
h1{font-size:1.9em;margin-top:0;padding-bottom:.35em;border-bottom:1px solid var(--line)}
h2{font-size:1.45em;padding-bottom:.3em;border-bottom:1px solid var(--line)}
h3{font-size:1.18em}h4{font-size:1.02em}
h2 .header-anchor,h3 .header-anchor{opacity:0;text-decoration:none;color:var(--muted);
margin-left:.35em;font-weight:400}
h2:hover .header-anchor,h3:hover .header-anchor{opacity:.55}
p{margin:.9em 0}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
strong{font-weight:700}
ul,ol{padding-left:1.6em;margin:.9em 0}
li{margin:.4em 0}
li>p{margin:.45em 0}
blockquote{margin:1.2em 0;padding:.7em 1.1em;border-left:3px solid var(--line);
background:var(--code-bg);color:var(--muted);border-radius:0 5px 5px 0;font-size:.95em}
blockquote p{margin:.42em 0}
blockquote strong{color:var(--fg)}
code{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
font-size:.87em;background:var(--code-bg);padding:.15em .38em;border-radius:4px;
border:1px solid var(--line)}
pre.code{background:var(--code-bg);border:1px solid var(--line);border-radius:7px;
padding:14px 16px;overflow-x:auto;margin:1.1em 0;position:relative;line-height:1.6}
pre.code code{background:none;border:0;padding:0;font-size:.855em}
pre.code[data-lang]::before{content:attr(data-lang);position:absolute;top:0;right:0;
padding:2px 9px;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
color:var(--muted);background:var(--th);border-left:1px solid var(--line);
border-bottom:1px solid var(--line);border-radius:0 6px 0 6px}
table{border-collapse:collapse;width:100%;margin:1.3em 0;font-size:.9em;display:block;
overflow-x:auto}
th,td{border:1px solid var(--line);padding:8px 12px;text-align:left;vertical-align:top}
th{background:var(--th);font-weight:700;white-space:nowrap}
tbody tr:nth-child(even){background:var(--zebra)}
td code{white-space:nowrap}
hr{border:0;border-top:1px solid var(--line);margin:2.4em 0}
.math-block{margin:1.3em 0;padding:14px 18px;background:var(--code-bg);
border:1px solid var(--line);border-radius:7px;display:flex;align-items:center;
justify-content:center;gap:18px;font-size:1.02em}
.math-tag{color:var(--muted);font-size:.9em}
.footnotes{margin-top:3em;padding-top:1.2em;border-top:1px solid var(--line);
font-size:.89em;color:var(--muted)}
.footnotes ol{padding-left:1.4em}
.footnotes li{margin:.55em 0}
.footnotes p{margin:.25em 0}
.footnote-backref{text-decoration:none;margin-left:.3em}
sup.footnote-ref a{text-decoration:none;padding:0 .12em;font-weight:600}
@media print{.toc{display:none}.wrap{max-width:none;padding:0}
pre.code,table{page-break-inside:avoid}body{font-size:11pt}}
"""


def convert(src: str, title: str) -> str:
    src, math_blocks = extract_math(src)

    md = (
        MarkdownIt("commonmark", {"highlight": make_highlighter(), "linkify": True})
        .enable(["table", "strikethrough", "linkify"])
        .use(footnote_plugin)
        .use(anchors_plugin, max_level=3, permalink=True, permalinkSymbol="#")
    )
    tokens = md.parse(src)
    body = md.renderer.render(tokens, md.options, {})
    toc = build_toc(tokens)

    for i, block in enumerate(math_blocks):
        body = re.sub(rf"<p>MATHBLOCK{i}ENDMATH</p>", block, body)

    pyg = HtmlFormatter().get_style_defs(".code")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}
@media (prefers-color-scheme:light){{{pyg}}}
</style>
</head>
<body>
<div class="wrap">
{toc}
<main>
{body}
</main>
</div>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="markdown -> 单文件 HTML")
    ap.add_argument("source", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="默认与源文件同名同目录的 .html")
    args = ap.parse_args()

    src = args.source.read_text(encoding="utf-8")
    m = re.search(r"^#\s+(.+)$", src, re.M)
    title = m.group(1).strip() if m else args.source.stem

    out = args.output or args.source.with_suffix(".html")
    out.write_text(convert(src, title), encoding="utf-8")
    print(f"{args.source} -> {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
