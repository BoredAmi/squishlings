#!/usr/bin/env python3
"""Ollama-compatible /api/chat endpoint backed by headless Claude Code (`claude -p`), for squishlings' LLM operator.

squishlings/llm.py POSTs {"model", "messages": [system, user], "options": {...}} to <url>/api/chat and reads
["message"]["content"]. This answers that with Claude, using the Claude Code login (Pro plan) instead of an API key.

Important: llm.py silently falls back to random mutation when a request fails, so this bridge never returns an
error for a transient problem (usage limit, network): it waits and retries until it gets an answer or until
MAX_WAIT has passed. Keep the job's llm.timeout above MAX_WAIT.

    python3 bridge.py --model opus --port 11500      # logs to bridge.log next to this file
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "bridge.log"
WORKDIR = HERE / "empty"          # no CLAUDE.md, no project files: Claude sees only the prompt
MAX_WAIT = 5.5 * 3600             # longest wait for one answer (covers a 5-hour usage window)
CALL_TIMEOUT = 600                # one claude -p call


def log(msg: str) -> None:
    with LOG.open("a") as f:
        f.write(f"{time.strftime('%F %T')} {msg}\n")


def claude_bin() -> str:
    """The newest Claude Code binary from the VS Code extension (the folder name changes on updates)."""
    env = os.environ.get("CLAUDE_BIN")
    if env:
        return env
    found = sorted(glob.glob(os.path.expanduser(
        "~/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude")))
    if not found:
        raise FileNotFoundError("no claude binary; set CLAUDE_BIN")
    return found[-1]


def ask(model: str, system: str, user: str) -> dict:
    """One headless call. Returns the parsed --output-format json result (raises on failure)."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(system)
        sysfile = f.name
    try:
        cmd = [claude_bin(), "-p", "--model", model, "--output-format", "json",
               "--system-prompt-file", sysfile, "--tools", "", "--strict-mcp-config",
               "--no-session-persistence", "--max-turns", "1"]
        p = subprocess.run(cmd, input=user, capture_output=True, text=True, cwd=WORKDIR, timeout=CALL_TIMEOUT)
    finally:
        os.unlink(sysfile)
    try:
        out = json.loads(p.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"exit {p.returncode}: {(p.stdout + p.stderr).strip()[:400]}")
    if out.get("is_error") or not out.get("result"):
        raise RuntimeError(f"claude error: {json.dumps(out)[:400]}")
    return out


class Handler(BaseHTTPRequestHandler):
    model = "opus"

    def log_message(self, *args) -> None:  # quiet http.server
        pass

    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/api/tags":
            self._send(200, {"models": [{"name": f"claude-{self.model}", "bridge": True}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            return self._send(404, {"error": "not found"})
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        msgs = req.get("messages", [])
        system = "\n\n".join(m["content"] for m in msgs if m.get("role") == "system")
        user = "\n\n".join(m["content"] for m in msgs if m.get("role") == "user")
        t0, delay, attempt = time.time(), 60, 0
        while True:
            attempt += 1
            try:
                out = ask(self.model, system, user)
                u = out.get("usage", {})
                log(f"ok model={self.model} attempt={attempt} secs={time.time() - t0:.1f} "
                    f"in={u.get('input_tokens')} cache_read={u.get('cache_read_input_tokens')} "
                    f"out={u.get('output_tokens')} cost_usd={out.get('total_cost_usd')}")
                return self._send(200, {"model": req.get("model"), "done": True,
                                        "message": {"role": "assistant", "content": out["result"]}})
            except Exception as exc:  # usage limit, network, timeout: wait and retry
                waited = time.time() - t0
                log(f"retry attempt={attempt} waited={waited:.0f}s next_in={delay}s: {exc}")
                if waited + delay > MAX_WAIT:
                    log(f"GIVING UP after {waited:.0f}s")
                    return self._send(503, {"error": str(exc)})
                time.sleep(delay)
                delay = min(delay * 2, 900)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="opus")
    ap.add_argument("--port", type=int, default=11500)
    a = ap.parse_args()
    WORKDIR.mkdir(exist_ok=True)
    Handler.model = a.model
    log(f"bridge start model={a.model} port={a.port} bin={claude_bin()}")
    HTTPServer(("127.0.0.1", a.port), Handler).serve_forever()   # one request at a time, like ollama here


if __name__ == "__main__":
    main()
