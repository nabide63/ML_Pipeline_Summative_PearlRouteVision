"""CLI: build a contact sheet for reviewing the collected images by eye.

Writes data/processed/contact_sheet.html (git-ignored), which shows every image in metadata.csv
from data/raw. Click an image to mark it as a reject; marks already in data/exclude.txt start
marked. "Save exclude.txt" downloads the full list: move it over data/exclude.txt, then run
scripts/clean_images.py.

Example:
    python scripts/contact_sheet.py
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from pearlroute.config import load_data_config
from pearlroute.data.cleaning import read_manual_excludes

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image Review</title>
<style>
  :root {
    --bg: #f7f7f5; --panel: #ffffff; --text: #1d1d1b; --muted: #6b6b66; --line: #deded8;
    --accent: #2b6cb0; --reject: #c53030; --reject-bg: rgba(197, 48, 48, .14);
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #161615; --panel: #20201e; --text: #ececea; --muted: #9d9d97; --line: #353532;
            --accent: #63a4e8; --reject: #f56565; --reject-bg: rgba(245, 101, 101, .18); }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { position: sticky; top: 0; z-index: 2; background: var(--panel);
           border-bottom: 1px solid var(--line); padding: 10px 16px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .row + .row { margin-top: 8px; }
  h1 { font-size: 16px; margin: 0 12px 0 0; }
  .tab { border: 1px solid var(--line); background: transparent; color: var(--text);
         padding: 5px 10px; border-radius: 6px; cursor: pointer; font: inherit; }
  .tab[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: #fff; }
  .tab .n { opacity: .75; font-variant-numeric: tabular-nums; }
  select, button.act { font: inherit; padding: 5px 10px; border-radius: 6px;
                       border: 1px solid var(--line); background: var(--panel); color: var(--text); }
  button.act { cursor: pointer; }
  button.primary { background: var(--reject); border-color: var(--reject); color: #fff; }
  .rule { color: var(--muted); }
  .spacer { flex: 1; }
  main { padding: 12px 16px 40px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 10px; }
  figure { margin: 0; background: var(--panel); border: 2px solid var(--line); border-radius: 8px;
           overflow: hidden; cursor: pointer; position: relative; }
  figure img { display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: cover; background: var(--line); }
  figcaption { padding: 5px 7px; font-size: 12px; }
  figcaption .t { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
  figcaption .q { color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  figure.rejected { border-color: var(--reject); background: var(--reject-bg); }
  figure.rejected img { opacity: .45; }
  figure.rejected::after { content: "REJECT"; position: absolute; top: 6px; left: 6px; background: var(--reject);
                           color: #fff; font-size: 11px; font-weight: 600; padding: 2px 6px; border-radius: 4px; }
  .open { position: absolute; top: 6px; right: 6px; background: rgba(0,0,0,.6); color: #fff; text-decoration: none;
          font-size: 12px; padding: 2px 7px; border-radius: 4px; }
  .empty { color: var(--muted); padding: 40px 0; text-align: center; }
</style>
</head>
<body>
<header>
  <div class="row"><h1>Image review</h1><div id="tabs" class="row"></div></div>
  <div class="row">
    <span class="rule" id="rule"></span>
  </div>
  <div class="row">
    <label>Found by <select id="query"></select></label>
    <label><input type="checkbox" id="onlyRejected"> rejects only</label>
    <span class="spacer"></span>
    <span id="status"></span>
    <button class="act" id="copy">Copy list</button>
    <button class="act primary" id="save">Save exclude.txt</button>
  </div>
</header>
<main><div id="grid" class="grid"></div></main>
<script>
const DATA = __DATA__;
const STORE_KEY = "pearlroute-review-rejects";
const rejected = new Set(DATA.initial);
try { (JSON.parse(localStorage.getItem(STORE_KEY)) || []).forEach(f => rejected.add(f)); } catch (e) {}
let current = DATA.classes[0];

const $ = id => document.getElementById(id);
function persist() { try { localStorage.setItem(STORE_KEY, JSON.stringify([...rejected])); } catch (e) {} }
function el(tag, props = {}, kids = []) {
  const n = Object.assign(document.createElement(tag), props); kids.forEach(k => n.append(k)); return n;
}

function renderTabs() {
  $("tabs").replaceChildren(...DATA.classes.map(c => {
    const total = DATA.images.filter(i => i.l === c).length;
    const rej = DATA.images.filter(i => i.l === c && rejected.has(i.f)).length;
    const b = el("button", { className: "tab", onclick: () => { current = c; $("query").value = ""; render(); } },
                 [c.replace(/_/g, " ") + " ", el("span", { className: "n", textContent: rej ? `${total} · ${rej} out` : total })]);
    b.setAttribute("aria-pressed", c === current);
    return b;
  }));
}

function renderQueries() {
  const counts = {};
  DATA.images.filter(i => i.l === current).forEach(i => counts[i.q] = (counts[i.q] || 0) + 1);
  const keep = $("query").value;
  $("query").replaceChildren(el("option", { value: "", textContent: "all searches" }),
    ...Object.entries(counts).sort((a, b) => b[1] - a[1])
      .map(([q, n]) => el("option", { value: q, textContent: `${q} (${n})` })));
  $("query").value = counts[keep] ? keep : "";
}

function render() {
  renderTabs(); renderQueries();
  $("rule").textContent = "Rule: " + DATA.rules[current];
  const q = $("query").value, only = $("onlyRejected").checked;
  const items = DATA.images.filter(i => i.l === current && (!q || i.q === q) && (!only || rejected.has(i.f)));
  $("grid").replaceChildren(...(items.length ? items.map(card) : [el("p", { className: "empty", textContent: "Nothing to show." })]));
  $("status").textContent = `${rejected.size} rejected in total`;
}

function card(i) {
  const fig = el("figure", { title: `${i.t}\\n${i.c} · ${i.q}\\n${i.f}` }, [
    el("img", { src: DATA.root + i.f, loading: "lazy", alt: i.t }),
    el("a", { className: "open", href: DATA.root + i.f, target: "_blank", textContent: "open",
              onclick: e => e.stopPropagation() }),
    el("figcaption", {}, [el("div", { className: "t", textContent: i.t || "(no title)" }),
                          el("div", { className: "q", textContent: i.q })]),
  ]);
  fig.classList.toggle("rejected", rejected.has(i.f));
  fig.onclick = () => {
    rejected.has(i.f) ? rejected.delete(i.f) : rejected.add(i.f);
    fig.classList.toggle("rejected", rejected.has(i.f));
    persist(); renderTabs(); $("status").textContent = `${rejected.size} rejected in total`;
  };
  return fig;
}

function listText() {
  const byFile = Object.fromEntries(DATA.images.map(i => [i.f, i]));
  return ["# Images rejected by eye; one filename per line. Read by scripts/clean_images.py.",
          ...[...rejected].sort().map(f => byFile[f] ? `${f}  # ${byFile[f].t.slice(0, 60)}` : f)].join("\\n") + "\\n";
}
$("save").onclick = () => {
  const a = el("a", { href: URL.createObjectURL(new Blob([listText()], { type: "text/plain" })), download: "exclude.txt" });
  document.body.append(a); a.click(); a.remove();
};
$("copy").onclick = async () => {
  try { await navigator.clipboard.writeText(listText()); $("status").textContent = "Copied"; }
  catch (e) { $("status").textContent = "Copy failed; use Save"; }
};
$("query").onchange = render;
$("onlyRejected").onchange = render;
render();
</script>
</body>
</html>
"""


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    cfg = load_data_config()
    p = cfg.paths
    df = pd.read_csv(p.metadata_csv, dtype=str, keep_default_na=False)
    out = p.processed_dir / "contact_sheet.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    root = os.path.relpath(p.raw_dir, out.parent).replace(os.sep, "/") + "/"
    data = {
        "root": root,
        "classes": cfg.class_names,
        "rules": {c: cfg.classes[c].rule for c in cfg.class_names},
        "initial": sorted(read_manual_excludes(p.manual_exclude)),
        "images": [{"f": r.filename, "l": r.label, "q": r.query, "t": r.title, "c": r.creator}
                   for r in df.sort_values(["label", "query", "filename"]).itertuples()],
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out.write_text(PAGE.replace("__DATA__", payload), encoding="utf-8")
    print(f"{len(df)} images -> {out}")


if __name__ == "__main__":
    main()
