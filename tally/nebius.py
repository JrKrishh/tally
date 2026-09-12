"""Nebius Token Factory client. One place for the key, retries, and the chat call."""
import os
import sys
import time
from pathlib import Path

import requests

API = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
DEFAULT_MODEL = os.environ.get("TALLY_MODEL", "nvidia/nemotron-3-super-120b-a12b")


def key():
    k = os.environ.get("NEBIUS_API_KEY")
    if not k:
        for p in (Path(__file__).resolve().parent.parent / ".nebius_key",
                  Path.home() / ".claude" / "session-rag" / ".nebius_key"):
            if p.exists():
                k = p.read_text(encoding="utf-8").strip()
                break
    if not k:
        sys.exit("NEBIUS_API_KEY not set.\n  PowerShell:  $env:NEBIUS_API_KEY = '<key>'\n"
                 "  or write it to " + str(Path(__file__).resolve().parent.parent / ".nebius_key"))
    return k


def _post(path, payload, tries=4):
    h = {"Authorization": "Bearer " + key(), "Content-Type": "application/json"}
    for i in range(tries):
        r = requests.post(API + path, headers=h, json=payload, timeout=180)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503) and i < tries - 1:
            time.sleep(2 ** i)
            continue
        sys.exit("HTTP %d from %s\n%s" % (r.status_code, path, r.text[:400]))
    raise RuntimeError("unreachable")


def models():
    r = requests.get(API + "/models", headers={"Authorization": "Bearer " + key()}, timeout=60)
    if r.status_code != 200:
        sys.exit("HTTP %d listing models\n%s" % (r.status_code, r.text[:400]))
    return sorted(m["id"] for m in r.json().get("data", []))


def chat(model, prompt, max_tokens=300, temperature=0.0):
    """-> (text, usage dict). Usage is kept so the receipt can be printed."""
    out = _post("/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    })
    return out["choices"][0]["message"]["content"], out.get("usage") or {}
