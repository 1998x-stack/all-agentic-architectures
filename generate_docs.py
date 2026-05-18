#!/usr/bin/env python3
import json, re
from pathlib import Path
from html import escape as h_esc

DOCS_DIR = Path("docs"); IPYNB_DIR = Path("ipynb"); SRC_DIR = Path("src")
CN_DOC = Path("从0开发大模型的17种Agent架构演进详细拆解.md")
DOCS_DIR.mkdir(exist_ok=True)

with open("arch_data.json") as f:
    data = json.load(f)
PARTS = data["parts"]
ARCHS = data["archs"]
ARCH_ORDER = [f"{i:02d}" for i in range(1, 18)]
for num in ARCH_ORDER:
    ARCHS[num]["num"] = num

def hex2rgb(h):
    h = h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"

PY_KW = {"False","None","True","and","as","assert","async","await","break","class","continue","def","del","elif","else","except","finally","for","from","global","if","import","in","is","lambda","nonlocal","not","or","pass","raise","return","try","while","with","yield"}
PY_FN = {"print","len","range","int","str","float","list","dict","set","tuple","bool","type","isinstance","super","enumerate","zip","map","filter","any","all","sum","min","max","abs","round","sorted","reversed","open","input","next","iter"}

def py_hl(code):
    r=[]; i=0; n=len(code)
    while i<n:
        c=code[i]
        if c=="#" and (i==0 or code[i-1] not in "'\""):
            e=code.find("\n",i); e=n if e==-1 else e
            r.append(f'<span class="cm">{h_esc(code[i:e])}</span>'); i=e; continue
        if code[i:i+3] in ('"""',"'''"):
            q=code[i:i+3]; e=code.find(q,i+3)
            if e!=-1: e+=3
            r.append(f'<span class="str">{h_esc(code[i:e])}</span>'); i=e; continue
        if c in "\"'":
            q=c; e=i+1
            while e<n:
                if code[e]=="\\": e+=2; continue
                if code[e]==q: e+=1; break
                e+=1
            r.append(f'<span class="str">{h_esc(code[i:e])}</span>'); i=e; continue
        if c=="@":
            e=i+1
            while e<n and (code[e].isalnum() or code[e] in "_.()"): e+=1
            r.append(f'<span class="op">{h_esc(code[i:e])}</span>'); i=e; continue
        if c.isdigit() or (c=="." and i+1<n and code[i+1].isdigit()):
            e=i+1 if c!="." else i+2
            while e<n and (code[e].isdigit() or code[e] in ".eExX_abcdefABCDEF"): e+=1
            r.append(f'<span class="num">{h_esc(code[i:e])}</span>'); i=e; continue
        if c.isalpha() or c=="_":
            e=i+1
            while e<n and (code[e].isalnum() or code[e]=="_"): e+=1
            w=code[i:e]
            if w in PY_KW: r.append(f'<span class="kw">{h_esc(w)}</span>')
            elif w in PY_FN: r.append(f'<span class="fn">{h_esc(w)}</span>')
            elif e<n and code[e]=="(": r.append(f'<span class="fn">{h_esc(w)}</span>')
            else: r.append(h_esc(w))
            i=e; continue
        if c in "=+-*/%<>!&|^~:": r.append(f'<span class="op">{h_esc(c)}</span>')
        else: r.append(h_esc(c))
        i+=1
    return "".join(r)

def get_py(num):
    for f in sorted(SRC_DIR.glob(f"{num}_*.py")):
        with open(f) as fh: return fh.read()
    return ""

def get_nb_md(num):
    p = IPYNB_DIR / ARCHS[num]["ipynb"]
    if not p.exists(): return ""
    nb = json.loads(p.read_text())
    parts=[]; intro=True
    for cell in nb.get("cells",[]):
        if cell.get("cell_type")=="markdown":
            src=cell.get("source",[]); t="".join(src) if isinstance(src,list) else src
            if intro: parts.append(t)
            if cell.get("id","").startswith("intro-definition"): intro=False
        elif cell.get("cell_type")=="code": intro=False
    return "\n\n".join(parts)

def nb2html(txt):
    if not txt: return ""
    result=[]; in_ul=False
    for line in txt.strip().split("\n"):
        s=line.strip()
        if not s:
            if in_ul: result.append("</ul>"); in_ul=False
            continue
        if s.startswith("# "): result.append(f'<h3 style="font-size:16px;font-weight:600;margin:20px 0 8px;color:var(--text)">{h_esc(s[2:])}</h3>')
        elif s.startswith("## "): result.append(f'<h4 style="font-size:14px;font-weight:600;margin:16px 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em">{h_esc(s[3:])}</h4>')
        elif s.startswith("### "):
            if in_ul: result.append("</ul>"); in_ul=False
            result.append(f'<p style="font-weight:600;margin:14px 0 4px;font-size:14px">{md_inline(s[4:])}</p>')
        elif s.startswith("*   "):
            if not in_ul: result.append('<ul style="margin:4px 0 12px 18px;list-style-type:disc">'); in_ul=True
            result.append(f'<li style="margin-bottom:4px;font-size:13px;line-height:1.6;color:var(--muted)">{md_inline(s[4:])}</li>')
        elif s.startswith("* "):
            if not in_ul: result.append('<ul style="margin:4px 0 12px 18px;list-style-type:disc">'); in_ul=True
            result.append(f'<li style="margin-bottom:4px;font-size:13px;line-height:1.6;color:var(--muted)">{md_inline(s[2:])}</li>')
        elif re.match(r"^\d+\.\s",s):
            if not in_ul: result.append('<ol style="margin:4px 0 12px 18px">'); in_ul=True
            result.append(f'<li style="margin-bottom:4px;font-size:13px;line-height:1.6;color:var(--muted)">{md_inline(re.sub(r"^\d+\.\s+","",s))}</li>')
        else:
            if in_ul: result.append("</ul>"); in_ul=False
            result.append(f'<p style="margin-bottom:8px;line-height:1.7;font-size:13px;color:var(--muted)">{md_inline(s)}</p>')
    if in_ul: result.append("</ul>")
    return "\n".join(result)

def md_inline(t):
    t=h_esc(t)
    t=re.sub(r"\*\*(.+?)\*\*",r'<strong style="color:var(--text)">\1</strong>',t)
    t=re.sub(r"`([^`]+)`",r'<code style="font-family:var(--mono);font-size:11px;background:var(--bg3);padding:1px 5px;border-radius:3px">\1</code>',t)
    return t

def cn_inline(t):
    t=h_esc(t)
    t=re.sub(r"\*\*(.+?)\*\*",r'<strong style="color:var(--text)">\1</strong>',t)
    t=re.sub(r"`([^`]+)`",r'<code style="font-family:var(--mono);font-size:12px;background:var(--bg3);padding:1px 6px;border-radius:4px;color:var(--c1)">\1</code>',t)
    return t

def parse_cn():
    raw = CN_DOC.read_text()
    secs = {}
    pattern = re.compile(r'^### (\d+)\\.\s+(.+?)$', re.MULTILINE)
    ms = list(pattern.finditer(raw))
    for i, m in enumerate(ms):
        sn = int(m.group(1)); title = m.group(2).strip()
        start = m.start(); end = ms[i+1].start() if i+1<len(ms) else len(raw)
        chunk = raw[start:end]
        parsed = {"title": title, "raw": chunk}
        subs = re.split(r'^####\s+', chunk, flags=re.MULTILINE)
        for sub in subs[1:]:
            nl = sub.find('\n')
            if nl==-1: continue
            hdr = sub[:nl].strip(); body = sub[nl:].strip()
            body = re.sub(r'!\[.*?\]\(data:image/svg\+xml[^)]*\)','',body).strip()
            parsed[hdr.lower()]=body
        secs[sn]=parsed
    return secs

def cn2html(cn_sec):
    if not cn_sec: return ""
    parts = []
    fp_raw = cn_sec.get("raw","").strip()
    fp_lines = fp_raw.split("\n")
    fp = ""
    for line in fp_lines:
        s = line.strip()
        if s.startswith("###") or not s:
            continue
        fp = s
        break
    if fp and len(fp)<400:
        parts.append(f'<p style="font-size:15px;color:var(--text);line-height:1.9;margin-bottom:24px;padding:16px 20px;background:var(--bg2);border-radius:8px;border-left:3px solid var(--c1)"><strong style="color:var(--c1)">核心洞察：</strong>{cn_inline(fp)}</p>')

    qs = [
        ("要解决什么问题？","要解决什么"),
        ("State 设计","state"),
        ("拓扑结构","拓扑"),
        ("Router 机制","router"),
        ("关键代码","关键代码"),
        ("失败模式","失败模式"),
        ("何时升级","什么时候升级"),
        ("何时升级","它为什么还不够"),
        ("何时升级","什么时候用"),
        ("附加洞察","它为什么比"),
        ("新增的能力","新增的能力"),
        ("新增的能力","它真正新增"),
        ("技术核心","技术核心"),
    ]

    seen = set()
    for dtitle, mkey in qs:
        if dtitle in seen: continue
        body = None
        for k,v in cn_sec.items():
            if mkey in k: body=v; break
        if not body: continue
        seen.add(dtitle)

        cbs = []
        def sc(m):
            cbs.append(m.group(1)); return f"__CB{len(cbs)-1}__"
        bc = re.sub(r'```.*?\n(.*?)```', sc, body, flags=re.DOTALL)

        hlines=[]
        for line in bc.split("\n"):
            s=line.strip()
            if not s: hlines.append(""); continue
            for j,cb in enumerate(cbs):
                line=line.replace(f"__CB{j}__",f'<code class="pill">{h_esc(cb[:60])}...</code>')
            if s.startswith("> "): hlines.append(f'<p style="font-size:13px;color:var(--muted);margin:12px 0;padding:8px 16px;border-left:2px solid var(--border);font-style:italic">{cn_inline(s[2:])}</p>')
            elif s.startswith("- "): hlines.append(f'<li style="margin-bottom:8px;line-height:1.75">{cn_inline(s[2:])}</li>')
            else: hlines.append(f'<p style="margin-bottom:12px;line-height:1.85">{cn_inline(s)}</p>')
        bh="\n".join(hlines)
        if "<li" in bh:
            bh=re.sub(r'(<li.*?</li>\n?)+',r'<ul style="margin:0 0 16px 20px;list-style-type:disc;color:var(--text);font-size:14px">\g<0></ul>',bh)

        for j,cb in enumerate(cbs):
            lang="python"
            cc=cb.strip()
            hl=py_hl(cc)
            ch=f"""<div class="codeblock" style="margin:12px 0 20px">
  <div class="cb-head" style="font-size:11px">
    <span class="cb-dot" style="background:#f85149"></span>
    <span class="cb-dot" style="background:#e3b341"></span>
    <span class="cb-dot" style="background:#28c840"></span>
    agno 实现示例 · {cc.count(chr(10))+1} 行
  </div>
  <div class="cb-body" style="font-size:12px;line-height:1.8">{hl}</div>
</div>"""
            bh=bh.replace(f"__CB{j}__",ch)

        parts.append(f"""<div style="margin-bottom:32px">
  <h4 style="font-size:16px;font-weight:600;margin:0 0 12px;color:var(--text)">{dtitle}</h4>
  <div style="padding-left:4px">{bh}</div>
</div>""")

    return "\n".join(parts)

def page_head(title, xtra=""):
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — All Agentic Architectures</title>
<link rel="stylesheet" href="base-dark.css">
<style>
  :root{{--c1:#28c840;--c1d:#1a7a2a;--c2:#378add;--c2d:#1a5290;--c3:#a259ff;--c3d:#5a2a8a;--c4:#f0883e;--c4d:#9a4a1a;--c5:#1db0a0;--c5d:#0f6e56}}
  .hero h1 em{{background:linear-gradient(90deg,var(--c1),var(--c2),var(--c3),var(--c4),var(--c5));-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
  .prevnext{{display:flex;justify-content:space-between;padding:32px 0;border-top:1px solid var(--border);margin-top:56px}}
  .prevnext a{{color:var(--muted);font-size:14px;transition:color .15s}}
  .prevnext a:hover{{color:var(--text);text-decoration:none}}
  .prevnext .pn-label{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--faint);margin-bottom:4px;display:block}}
  .cn-section{{margin-top:48px}}
  .cn-section h3{{font-size:20px;font-weight:700;margin-bottom:20px;padding-bottom:8px;border-bottom:1px solid var(--border)}}
  .toc-link{{display:inline-block;padding:3px 8px;border-radius:4px;font-size:11px;font-family:var(--mono);color:var(--muted);border:1px solid var(--border);margin:2px 4px 2px 0;transition:all .15s}}
  .toc-link:hover{{border-color:var(--c1);color:var(--c1);text-decoration:none}}
</style>
{xtra}
</head>
<body>"""

def _nav():
    return """<nav>
  <div class="nav-inner">
    <a href="index.html" style="text-decoration:none"><span class="nav-brand">All Agentic Architectures <span>· 17 种架构</span></span></a>
    <a class="nl" href="index.html">总览</a>
    <a class="nl" href="index.html#p1">基础</a>
    <a class="nl" href="index.html#p2">多智能体</a>
    <a class="nl" href="index.html#p3">记忆推理</a>
    <a class="nl" href="index.html#p4">安全可靠</a>
    <a class="nl" href="index.html#p5">学习适应</a>
  </div>
</nav>"""

def _footer():
    return """<div class="footer"><p>Built from <a href="https://github.com/1998x-stack/all-agentic-architectures" style="color:var(--text)">all-agentic-architectures</a></p></div>"""

def build_index():
    fboxes=""
    for idx,part in enumerate(PARTS):
        cn=part["cn"]; arr=""
        if idx<len(PARTS)-1: arr=f'<div class="flow-arrow"><div class="arr-icon">→</div><div class="arr-text">演化到</div></div>'
        fboxes+=f"""<div class="flow-box" style="border-color:rgba({hex2rgb(part['color'])},0.25)">
  <div class="fb-tag" style="color:var(--{cn})">{part['name'].split(':')[0].upper()}</div>
  <div class="fb-name">{part['name_cn']}</div>
  <div class="fb-sub">{part['desc_cn']}</div>
</div>{arr}"""

    secs=""
    for part in PARTS:
        cn=part["cn"]; cd=part["colord"]; col=part["color"]
        cards=""
        for num in part["archs"]:
            a=ARCHS[num]
            cards+=f"""<div class="card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
    <span style="font-size:11px;font-weight:700;color:var(--{cn});letter-spacing:.05em">#{num}</span>
    <a href="{a['id']}.html" style="font-size:12px;color:var(--muted);border:1px solid var(--border);padding:3px 10px;border-radius:6px;transition:all .15s" onmouseover="this.style.borderColor='var(--{cn})';this.style.color='var(--{cn})'" onmouseout="this.style.borderColor='var(--border)';this.style.color='var(--muted)'">探索 &rarr;</a>
  </div>
  <div style="font-size:15px;font-weight:600;margin-bottom:4px;color:var(--text)">{h_esc(a['name_cn'])} <span style="font-size:12px;font-weight:400;color:var(--muted)">{h_esc(a['name'])}</span></div>
  <div style="font-size:11px;color:var(--muted);margin-bottom:8px;background:var(--bg3);display:inline-block;padding:2px 8px;border-radius:4px">{h_esc(a['cn_capability'])}</div>
  <div style="font-size:13px;color:var(--muted);line-height:1.65">{h_esc(a['cn_brief'])}</div>
</div>"""
        ncols="grid2" if len(part["archs"])<=4 else "grid3"
        secs+=f"""<section class="layer-sec" id="{part['id']}">
  <div class="layer-hdr">
    <span class="layer-id-badge" style="background:{cd};color:rgba(255,255,255,.9)">{part['name'].split(':')[0]}</span>
    <div class="layer-title" style="font-size:24px">{part['name_cn']}</div>
  </div>
  <p class="layer-def" style="border-color:{col}">{h_esc(part['desc_cn'])}</p>
  <div class="{ncols}">{cards}</div>
</section>"""

    return page_head("总览")+_nav()+f"""<div class="page">
  <section class="hero">
    <div class="hero-badge"><span class="dot"></span> 教育项目 · 17 种实现 · 2026</div>
    <h1>17 种 <em>Agentic 架构</em></h1>
    <p>从单次生成到自反式元认知系统——现代 AI Agent 设计的完整动手大师课。每种架构均使用 LangChain 和 LangGraph 端到端实现，配有深度解析与生产级代码。</p>
    <div class="tag-row">
      <span class="ltag" style="color:var(--c1);background:rgba(40,200,64,.08);border-color:rgba(40,200,64,.25)">基础模式</span>
      <span class="ltag" style="color:var(--c2);background:rgba(55,138,221,.08);border-color:rgba(55,138,221,.25)">多智能体</span>
      <span class="ltag" style="color:var(--c3);background:rgba(162,89,255,.08);border-color:rgba(162,89,255,.25)">记忆推理</span>
      <span class="ltag" style="color:var(--c4);background:rgba(240,136,62,.08);border-color:rgba(240,136,62,.25)">安全可靠</span>
      <span class="ltag" style="color:var(--c5);background:rgba(29,176,160,.08);border-color:rgba(29,176,160,.25)">学习适应</span>
    </div>
  </section>
  <section class="flow">
    <div class="sec-label">学习路径</div>
    <div style="font-size:20px;font-weight:600;margin-bottom:8px">从简单到自感知</div>
    <p style="font-size:14px;color:var(--muted);margin-bottom:32px;max-width:760px;line-height:1.8">每种架构都建立在前一种的局限性之上。从简单的自我批判循环，逐步演进为能规划、协作、记忆、模拟并最终理解自身边界的系统。</p>
    <div class="flow-track">{fboxes}</div>
  </section>
  {secs}
</div>"""+_footer()+"</body></html>"

def build_detail(num, cn_secs):
    a=ARCHS[num]; part=next(p for p in PARTS if p["id"]==a["part"])
    cn=part["cn"]; cd=part["colord"]; col=part["color"]

    cn_sec=cn_secs.get(a["cn_section"],{})
    cn_html=cn2html(cn_sec) if cn_sec else ""

    en_md=get_nb_md(num); en_html=nb2html(en_md) if en_md else ""

    py_src=get_py(num); code_html=""
    if py_src:
        hl=py_hl(py_src)
        code_html=f"""<div class="codeblock">
  <div class="cb-head">
    <span class="cb-dot" style="background:#f85149"></span><span class="cb-dot" style="background:#e3b341"></span><span class="cb-dot" style="background:#28c840"></span>
    src/... · LangChain + LangGraph · {py_src.count(chr(10))} 行
  </div>
  <div class="cb-body" style="font-size:12px;line-height:1.8">{hl}</div>
</div>"""

    idx=ARCH_ORDER.index(num)
    pa=ARCHS[ARCH_ORDER[idx-1]] if idx>0 else None
    na=ARCHS[ARCH_ORDER[idx+1]] if idx<len(ARCH_ORDER)-1 else None
    nav='<div class="prevnext">'
    if pa: nav+=f'<a href="{pa["id"]}.html"><span class="pn-label">&larr; 上一架构</span>#{pa["num"]} {h_esc(pa["name_cn"])}</a>'
    else: nav+="<span></span>"
    if na: nav+=f'<a href="{na["id"]}.html" style="text-align:right"><span class="pn-label">下一架构 &rarr;</span>#{na["num"]} {h_esc(na["name_cn"])}</a>'
    else: nav+="<span></span>"
    nav+="</div>"

    toc=""
    if cn_html:
        toc="""<div style="margin-bottom:28px;line-height:2">
  <span style="font-size:11px;color:var(--faint);text-transform:uppercase;letter-spacing:.06em">快速导航</span><br>
  <a href="#cn-analysis" class="toc-link">六维分析</a>
  <a href="#en-intro" class="toc-link">英文解析</a>
  <a href="#code" class="toc-link">LangGraph 实现</a>
</div>"""

    return page_head(f"#{num} {a['name_cn']}")+_nav()+f"""<div class="page">
  <section class="hero" style="padding-bottom:28px">
    <div class="hero-badge"><span class="dot" style="background:var(--{cn})"></span> #{num} · {h_esc(a['cn_stage'])} · {h_esc(a['cn_capability'])}</div>
    <h1 style="font-size:34px;color:var(--text);margin-bottom:6px">{h_esc(a['name_cn'])} <span style="font-size:18px;font-weight:400;color:var(--muted)">{h_esc(a['name'])}</span></h1>
    <p style="font-size:14px;color:var(--faint);margin-bottom:8px;font-style:italic">{h_esc(a['subtitle_cn'])}</p>
    <p style="margin-bottom:8px;font-size:14px">{h_esc(a['cn_brief'])}</p>
    <p style="font-size:13px;color:var(--muted);margin-bottom:20px">典型应用：{h_esc(a['usecase_cn'])}</p>
    <div class="tag-row"><span class="ltag" style="color:var(--{cn});background:rgba({hex2rgb(col)},.08);border-color:rgba({hex2rgb(col)},.25)">{part['name'].split(':')[0]}</span></div>
  </section>
  {toc}
  <section class="layer-sec" style="padding-top:0;border-bottom:none">
    <div class="layer-hdr">
      <span class="layer-id-badge" style="background:{cd};color:rgba(255,255,255,.9)">核心概念</span>
      <div class="layer-title" style="font-size:18px">{h_esc(a['subtitle_cn'])}</div>
    </div>
    <p class="layer-def" style="border-color:{col}">{h_esc(a['core_cn'])}</p>
  </section>
  <section class="layer-sec cn-section" id="cn-analysis">
    <h3 style="border-color:{col}">六维深度分析</h3>
    <p style="font-size:13px;color:var(--faint);margin-bottom:24px">以下分析基于统一框架的六个核心问题：问题定义 → State 建模 → 拓扑结构 → Router 机制 → 失败模式 → 升级时机</p>
    {cn_html}
  </section>
  <section class="layer-sec" id="en-intro" style="border-bottom:none">
    <h3 style="border-color:{col}">英文原版解析</h3>
    <p style="font-size:12px;color:var(--faint);margin-bottom:16px">来自 Jupyter Notebook 的原始英文概念说明</p>
    <div style="font-size:13px;line-height:1.8;color:var(--muted)">{en_html if en_html else '<p style="color:var(--muted)">内容提取中...</p>'}</div>
  </section>
  <section class="layer-sec" id="code">
    <h3 style="border-color:{col}">LangChain + LangGraph 实现代码</h3>
    <p style="font-size:12px;color:var(--faint);margin-bottom:16px">完整的端到端实现代码，可从 src/ 目录直接运行</p>
    {code_html if code_html else '<p style="color:var(--muted)">未找到 Python 源码。</p>'}
  </section>
  {nav}
</div>"""+_footer()+"</body></html>"

def main():
    print("Parsing Chinese doc...")
    cn_secs=parse_cn()
    print(f"  {len(cn_secs)} sections found")

    idx=build_index()
    (DOCS_DIR/"index.html").write_text(idx); print("  \u2713 index.html")

    for num in ARCH_ORDER:
        pg=build_detail(num,cn_secs)
        (DOCS_DIR/f"{ARCHS[num]['id']}.html").write_text(pg)
        print(f"  \u2713 {ARCHS[num]['id']}.html")

    print(f"\nGenerated {len(ARCH_ORDER)+1} files in docs/")

if __name__=="__main__":
    main()
