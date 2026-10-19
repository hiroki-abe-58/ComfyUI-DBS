"""Submit an API-format workflow to a running ComfyUI, wait for it, and save the outputs (standard library only).

    python scripts/queue_client.py workflows/api/separate_soundtrack.json --server http://127.0.0.1:8188 --set 2.model=DBS-BSRNN-DnR

The workflow runs through the normal ComfyUI queue (POST /prompt), exactly as the web UI would run it.
`--set NODE.INPUT=VALUE` overrides one widget value (VALUE is parsed as JSON when possible, else used as text).
Prints one JSON line: prompt id, status, wall seconds, saved files and the text outputs (e.g. the node reports).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


def _get(url: str, timeout: float = 30) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"POST /prompt rejected ({e.code}): {e.read().decode(errors='replace')[:2000]}")


def apply_overrides(workflow: dict, overrides: list[str]) -> None:
    for item in overrides:
        target, _, raw = item.partition("=")
        node, _, name = target.partition(".")
        if node not in workflow or not name:
            raise SystemExit(f"--set {item!r}: unknown node or missing input name")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        workflow[node]["inputs"][name] = value


def run(server: str, workflow: dict, out_dir: Path | None, timeout: float) -> dict:
    server = server.rstrip("/")
    client_id = str(uuid.uuid4())
    started = time.time()
    queued = _post(f"{server}/prompt", {"prompt": workflow, "client_id": client_id})
    prompt_id = queued["prompt_id"]
    entry = None
    while time.time() - started < timeout:
        history = json.loads(_get(f"{server}/history/{prompt_id}"))
        if prompt_id in history and history[prompt_id].get("status", {}).get("completed") is not None:
            entry = history[prompt_id]
            if entry["status"].get("completed") or entry["status"].get("status_str") == "error":
                break
        time.sleep(0.5)
    wall = time.time() - started
    if entry is None:
        return {"prompt_id": prompt_id, "status": "timeout", "wall_seconds": round(wall, 2)}
    status = entry["status"]
    result = {"prompt_id": prompt_id, "status": status.get("status_str"), "wall_seconds": round(wall, 2), "files": [], "text": {}}
    for message in status.get("messages", []):
        if message[0] == "execution_error":
            result["error"] = {k: message[1].get(k) for k in ("node_type", "exception_message")}
        elif message[0] == "execution_interrupted":
            result["status"] = "interrupted"
    for node_id, output in entry.get("outputs", {}).items():
        for kind in ("audio", "images", "gifs"):
            for f in output.get(kind, []):
                query = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")})
                if out_dir is not None:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    target = out_dir / f["filename"]
                    target.write_bytes(_get(f"{server}/view?{query}", timeout=120))
                    result["files"].append(str(target))
                else:
                    result["files"].append(f["filename"])
        if "text" in output:
            result["text"][node_id] = output["text"]
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--out", type=Path, default=None, help="download outputs into this folder")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--set", action="append", default=[], metavar="NODE.INPUT=VALUE")
    args = parser.parse_args(argv)
    workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
    apply_overrides(workflow, args.set)
    result = run(args.server, workflow, args.out, args.timeout)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
