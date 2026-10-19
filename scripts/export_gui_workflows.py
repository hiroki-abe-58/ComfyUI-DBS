"""Load the API workflows into a running ComfyUI frontend, check them, and export the UI-format workflows.

Maintainer tool (needs `playwright` and a Chromium for it; not part of the node pack):

  python scripts/export_gui_workflows.py --url http://127.0.0.1:PORT --shots DIR

For each workflows/api/<name>.json it loads the graph with the frontend's own `app.loadApiJson`, checks that every
node type is known to the frontend (no "missing nodes") and that `app.graphToPrompt()` gives back the same class
types, links and widget values, saves a screenshot, and writes the frontend's serialized graph to
workflows/gui/<name>.json with the seed set to "fixed". Nothing is queued.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SEEDED = set()
FRONTEND_OWNED = set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--api-dir", type=Path, default=Path("workflows/api"))
    ap.add_argument("--out", type=Path, default=Path("workflows/gui"))
    ap.add_argument("--shots", type=Path, required=True)
    args = ap.parse_args()
    if not args.url.startswith("http://127.0.0.1:"):
        raise SystemExit("only a local test server")
    args.shots.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1680, "height": 1000}, locale="en-US")
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(args.url)
        page.wait_for_function("() => window.app && window.app.graph && window.LiteGraph", timeout=120000)
        page.wait_for_timeout(3000)
        page.keyboard.press("Escape")  # the template browser opens on first start
        page.wait_for_timeout(500)
        for api_file in sorted(args.api_dir.glob("*.json")):
            name = api_file.stem
            api = json.loads(api_file.read_text(encoding="utf-8"))
            res = page.evaluate(
                """async ([api, name]) => {
                    await window.app.loadApiJson(api, name);
                    await new Promise(r => setTimeout(r, 1500));
                    const nodes = window.app.graph._nodes;
                    const known = Object.keys(window.LiteGraph.registered_node_types);
                    const missing = nodes.filter(n => !known.includes(n.type)).map(n => n.type);
                    const { output } = await window.app.graphToPrompt();
                    const ui = window.app.graph.serialize();
                    return { count: nodes.length, missing, output, ui };
                }""",
                [api, name],
            )
            page.keyboard.press("Escape")
            page.mouse.click(840, 560)  # focus the canvas (empty area) and fit the whole graph
            page.keyboard.press(".")
            page.wait_for_timeout(800)
            page.screenshot(path=str(args.shots / f"{name}.png"))
            back = res["output"]
            mismatch = []
            for nid, node in api.items():
                got = back.get(nid)
                if not got or got["class_type"] != node["class_type"]:
                    mismatch.append((nid, "class_type"))
                    continue
                for k, v in node["inputs"].items():
                    if k in FRONTEND_OWNED:
                        continue
                    if got["inputs"].get(k) != v:
                        mismatch.append((nid, k, v, got["inputs"].get(k)))
            report[name] = {"nodes": res["count"], "missing_node_types": res["missing"], "roundtrip_mismatches": mismatch}
            for node in res["ui"]["nodes"]:
                if node["type"] in SEEDED:  # reproducible example: keep the seed after each run
                    node["widgets_values"] = ["fixed" if v == "randomize" else v for v in node["widgets_values"]]
                    if node.get("widgets_values_named", {}).get("control_after_generate") == "randomize":  # newer frontends keep both
                        node["widgets_values_named"]["control_after_generate"] = "fixed"
            (args.out / f"{name}.json").write_text(json.dumps(res["ui"], indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        report["page_errors"] = errors
        browser.close()
    print(json.dumps(report, indent=1, ensure_ascii=False))
    bad = any(v["missing_node_types"] or v["roundtrip_mismatches"] for k, v in report.items() if k != "page_errors")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
