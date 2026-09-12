"""Pull the agentic coding collections from the Every Eval Ever datastore.

snapshot_download does not work here: the repo is large enough that the Hub
returns an empty siblings list, so it silently matches nothing. The per-collection
tree API lists every file (terminalbench 520, swebenchpro 324 -- both under the
1000-entry cap), and per-file hf_hub_download resumes.

Two phases: aggregates are ~400 small JSON files (seconds); samples are ~7.4 GB.
Re-running skips any file already present at the listed size.

  python -m tally.pull --phase aggregates
  python -m tally.pull --phase samples
  python -m tally.pull                      # both, in that order
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

REPO = "evaleval/EEE_datastore"
COLLECTIONS = ["terminalbench", "swebenchpro"]
DATA = Path(os.environ.get("TALLY_DATA", Path(__file__).resolve().parent.parent / "data" / "eee"))
TREE = "https://huggingface.co/api/datasets/%s/tree/main/data/%s?recursive=true"
SUFFIX = {"aggregates": ".json", "samples": "_samples.jsonl"}


def listing(col):
    r = requests.get(TREE % (REPO, col), timeout=120)
    r.raise_for_status()
    entries = [(e["path"], e.get("size", 0)) for e in r.json() if e.get("type") == "file"]
    if len(r.json()) >= 1000:
        sys.exit("%s: tree listing hit the 1000-entry cap; paginate before trusting it" % col)
    return entries


def wanted(entries, phase):
    suf = SUFFIX[phase]
    return [(p, s) for p, s in entries if p.endswith(suf)]


def progress(msg, i, n):
    """\\r-overwrite on a terminal; every 25th line when captured to a file."""
    if sys.stdout.isatty():
        sys.stdout.write("\r" + msg)
        sys.stdout.flush()
    elif i == n or i % 25 == 0:
        print(msg)


def fetch(path, size):
    local = DATA / path
    if local.exists() and local.stat().st_size == size:
        return path, 0
    hf_hub_download(REPO, filename=path, repo_type="dataset", local_dir=str(DATA))
    return path, size


def pull(phase, workers):
    t0 = time.time()
    todo = []
    for col in COLLECTIONS:
        todo += wanted(listing(col), phase)
    total_mb = sum(s for _, s in todo) / 1e6
    print("# %s: %d files, %.0f MB listed -> %s" % (phase, len(todo), total_mb, DATA))
    done, got_mb, skipped = 0, 0.0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch, p, s) for p, s in todo]
        for f in as_completed(futs):
            path, size = f.result()
            done += 1
            if size == 0:
                skipped += 1
            got_mb += size / 1e6
            progress("  %d/%d files, %.0f MB fetched, %d already present, %.0fs"
                     % (done, len(todo), got_mb, skipped, time.time() - t0), done, len(todo))
    print("\n# %s done" % phase)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["aggregates", "samples"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    for phase in ([args.phase] if args.phase else ["aggregates", "samples"]):
        pull(phase, args.workers)


if __name__ == "__main__":
    main()
