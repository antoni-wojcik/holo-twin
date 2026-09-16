"""
Shared orchestration for the studies/ scripts.

Each pipeline call (acquire / train / cgh) is run in its own subprocess via
studies/worker.py, instead of in-process like the notebook did. That's the
actual fix for the notebook slowing down over a long session: PyTorch/CUDA
state (model instances, autograd graphs, the CUDA caching allocator's
fragmented reservations) accumulates in one long-lived interpreter no
matter how carefully you `del` things, because a stray reference anywhere
(a traceback, a plot, a closure) is enough to keep a whole model's GPU
tensors alive. A fresh subprocess has none of that history -- when it
exits, the OS reclaims everything unconditionally, the same guarantee you'd
get from restarting the notebook kernel between every run, just automatic.

Also keeps a manifest (studies/_runs/manifest.jsonl) of every run's
outcome, so:
  - a study (or run_main_sequence.py) can be safely re-launched after a
    crash without redoing already-successful runs,
  - a downstream step (e.g. study 6 needing study 4's 150-sample twin
    path) can look up a prior run's output path on disk, instead of it
    only living in a notebook variable that dies with the kernel.

Each study script (study_XX_*.py) only needs to build a dict of
{run_id: config} and hand it to run_study() (or run_isolated() directly,
for chained/one-off runs) -- all the subprocess/logging/resume machinery
lives here so the study scripts themselves stay just "define the configs
relative to DEFAULT_CONFIG, that's it."
"""
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from src.config import REPO_ROOT
from src.io.base_config import BaseConfig

RUNS_DIR = REPO_ROOT / "studies" / "_runs"
MANIFEST_PATH = RUNS_DIR / "manifest.jsonl"


class Manifest:
    """
    Append-only JSONL log of every (study, run_id) attempt, keyed to the
    latest attempt for each. Lets a study be safely re-run: successful
    runs are skipped (see `resume` on run_isolated), not repeated.
    """

    def __init__(self, path: Path = MANIFEST_PATH):
        self.path = path
        self._latest = {}
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self._latest[(record["study"], record["run_id"])] = record

    def get(self, study: str, run_id: str) -> Optional[dict]:
        return self._latest.get((study, run_id))

    def record(self, entry: dict) -> None:
        self._latest[(entry["study"], entry["run_id"])] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")


def run_isolated(
    module: str,
    cfg: BaseConfig,
    run_id: str,
    study: str,
    manifest: Optional[Manifest] = None,
    resume: bool = True,
) -> Optional[str]:
    """
    Run `cfg` through `module`'s `run(cfg)` (e.g. "pipelines.exp.run_training")
    in a fresh subprocess. Returns the resulting DATA_ROOT-relative path on
    success, or None on failure (already printed and logged -- callers
    decide whether to continue to the next config or stop, same as the
    try/except-and-continue loops the notebook used to hand-roll).

    If `resume` (default) and this exact (study, run_id) already succeeded
    per the manifest, skips straight to returning the cached path -- lets a
    crashed unattended sequence be re-launched without redoing finished work.
    Pass resume=False to force a fresh run regardless (e.g. while iterating
    on a config).
    """
    manifest = manifest or Manifest()

    if resume:
        cached = manifest.get(study, run_id)
        if cached is not None and cached["status"] == "ok":
            print(f"[{study}/{run_id}] already completed -> {cached['path']} (skipping)")
            return cached["path"]

    run_dir = RUNS_DIR / study / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    result_path = run_dir / "result.json"
    log_path = run_dir / "log.txt"
    cfg.save(config_path)

    print(f"[{study}/{run_id}] starting {module} (log: {log_path})")
    start = time.time()

    # A fresh `python -m studies.worker` process per call is the whole point:
    # it gets its own CUDA context, so nothing from a previous run (this one
    # or any other) can still be resident when it starts.
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "studies.worker",
            "--module", module,
            "--config", str(config_path),
            "--result", str(result_path),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    # Tee the worker's output live to both the console and a per-run log
    # file, so an unattended run can be watched (or tailed) in progress.
    with open(log_path, "w") as log_file:
        for line in proc.stdout:
            log_file.write(line)
            print(f"  [{study}/{run_id}] {line}", end="")
    proc.wait()

    duration = time.time() - start

    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        # Worker died before it could write a result (e.g. hard crash, OOM kill).
        result = {"status": "error", "error": f"worker exited {proc.returncode} without writing a result file"}

    entry = {
        "study": study, "run_id": run_id, "module": module,
        "status": result["status"], "path": result.get("path"),
        "error": result.get("error"), "duration_s": round(duration, 1),
        "log": str(log_path),
    }
    manifest.record(entry)

    if result["status"] == "ok":
        print(f"[{study}/{run_id}] done in {duration:.0f}s -> {result['path']}")
        return result["path"]
    else:
        print(f"[{study}/{run_id}] FAILED after {duration:.0f}s: {result.get('error')} (see {log_path})")
        return None


def run_study(module: str, study: str, configs: dict, manifest: Optional[Manifest] = None) -> dict:
    """
    Run every {run_id: cfg} in `configs` through `module`, one subprocess
    at a time, continuing past individual failures. Prints a final
    success/fail summary, same shape as the notebook's sweep cells did.
    Returns {run_id: path_or_None}.
    """
    manifest = manifest or Manifest()
    results = {
        run_id: run_isolated(module, cfg, run_id, study, manifest)
        for run_id, cfg in configs.items()
    }

    failed = [run_id for run_id, path in results.items() if path is None]
    succeeded = len(results) - len(failed)
    print(f"\n{study}: {succeeded}/{len(results)} runs succeeded." + (f" Failed: {failed}" if failed else ""))
    return results
