"""Multi-slot variant of run_queue.py: identical stages, commands and outputs, several GPU jobs.

Scheduling rules
  * at most OSFP4_SLOTS stages at once (default 2), picked in the serial queue's order;
  * at most OSFP4_MAX_HEAVY long-generation stages (GPQA, AIME, IFBench) at once
    (default 1; raise it only on a GPU with memory for two 8192-token batches);
  * a stage that fails with CUDA OOM is retried once, alone (stages resume from
    saved batches / the lm-eval request cache, so a retry repeats at most a batch);
  * any other failure is recorded and left for a later rerun, like run_queue.py.
Holds the same lock, so it can never run next to run_queue.py.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_queue as q  # noqa: E402

HEAVY = {"gpqa_diamond", "aime2025", "ifbench"}
SLOTS = int(os.environ.get("OSFP4_SLOTS", "2"))
MAX_HEAVY = int(os.environ.get("OSFP4_MAX_HEAVY", "1"))
ENV = {**q.ENV, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
status_lock = threading.Lock()


def pending_stages() -> list[dict]:
    protocol = json.loads(q.PROTOCOL.read_text())
    stages = []
    for method in q.METHODS:
        mid, ckpt = method[0], method[1]
        if ckpt is not None and not q.calibration_passed(ckpt):
            continue
        out_root = q.R / f"benchmarks-{q.PROFILE}" / mid
        wiki = q.R / f"{ckpt or 'bf16'}-wikitext2.json"
        if not wiki.exists():
            stages.append({"name": f"{mid}-wikitext2", "task": "wikitext2",
                           "cmd": [q.EVAL_PY, "-u", str(q.H / "eval_wikitext.py"),
                                   "--model-path", str(q.checkpoint(method)), "--out", str(wiki)]})
        for task in protocol["order"]:
            cfg = protocol["tasks"][task]
            out = out_root / ("lmeval" if cfg["runner"] == "lmeval" else "generate") / task
            res = out / "results.json"
            if res.exists() and json.loads(res.read_text()).get("limit") is None:
                continue
            if cfg["runner"] == "lmeval":
                cmd = [q.EVAL_PY, "-u", str(q.H / "eval_lmeval.py"), "--model-path", str(q.checkpoint(method)),
                       "--task", task, "--protocol", str(q.PROTOCOL), "--out", str(out),
                       "--cache", str(q.R / "lmeval-request-cache" / mid / task)]
            else:
                cmd = [q.EVAL_PY, "-u", str(q.H / "eval_generate.py"), "--model-path", str(q.checkpoint(method)),
                       "--task", task, "--protocol", str(q.PROTOCOL), "--out", str(out)]
            stages.append({"name": f"{mid}-{task}", "task": task, "cmd": cmd})
    return stages


def update_status(running, completed, failed):
    with status_lock:
        q.write_status(status="running", mode=f"parallel-{SLOTS}", stage=", ".join(sorted(running)) or "idle",
                       running_stages=sorted(running), completed_stages=completed, failed_stages=failed)


def run(stage: dict, result: dict):
    log = q.R / "benchmark-logs" / f"{stage['name']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    tag = f"[{stage['name']}] "
    print(f"\n===== {time.strftime('%H:%M:%S')} START {stage['name']}", flush=True)
    with log.open("a") as f:
        f.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%S')} (parallel) {' '.join(stage['cmd'])}\n")
        f.flush()
        proc = subprocess.Popen(stage["cmd"], env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        oom = False
        for line in proc.stdout:
            f.write(line)
            oom |= "OutOfMemoryError" in line or "CUDA out of memory" in line
            # Keep the shared progress log readable: only stage milestones and errors.
            if any(k in line for k in ("_COMPLETE", "Traceback", "Error", "] seed=")):
                sys.stdout.write(tag + line)
                sys.stdout.flush()
        rc = proc.wait()
    print(f"===== {time.strftime('%H:%M:%S')} END {stage['name']} rc={rc}{' (OOM)' if oom else ''}", flush=True)
    result.update(rc=rc, oom=oom)


def main():
    lock = open(q.LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another queue holds {q.LOCK}; not starting a second one")

    todo = pending_stages()
    running: dict[str, tuple[threading.Thread, dict, dict]] = {}
    completed, failed, exclusive, retried = [], [], set(), set()
    update_status(running, completed, failed)
    while todo or running:
        for name, (thread, stage, result) in list(running.items()):
            if thread.is_alive():
                continue
            del running[name]
            if result["rc"] == 0:
                completed.append(name)
            elif result["oom"] and name not in retried:
                retried.add(name)
                exclusive.add(name)
                todo.insert(0, stage)
                print(f"===== {name} hit CUDA OOM; requeued to run alone", flush=True)
            else:
                failed.append(name)
            update_status(running, completed, failed)

        started = False
        if len(running) < SLOTS and not any(n in exclusive for n in running):
            heavy_running = sum(running[n][1]["task"] in HEAVY for n in running)
            for stage in todo:
                if stage["name"] in exclusive and running:
                    continue
                if stage["task"] in HEAVY and heavy_running >= MAX_HEAVY:
                    continue
                todo.remove(stage)
                result: dict = {}
                thread = threading.Thread(target=run, args=(stage, result), daemon=True)
                running[stage["name"]] = (thread, stage, result)
                thread.start()
                started = True
                update_status(running, completed, failed)
                time.sleep(90)  # stagger model loading
                break
        if not started:
            time.sleep(5)

    ok = q.run_stage("aggregate", [q.EVAL_PY, "-u", str(q.H / "aggregate.py")])
    final = json.loads((q.R / "benchmark-results.json").read_text()) if (q.R / "benchmark-results.json").exists() else {}
    if ok and final.get("status") == "selected_9_evaluation_matrix_complete" and not failed:
        q.write_status(status="benchmark_matrix_complete", stage="awaiting_user_review_before_next_model", running_stages=[])
        print("OLMO3_BENCHMARK_MATRIX_COMPLETE; STOPPED_BEFORE_ANY_NEXT_MODEL")
    else:
        q.write_status(status="incomplete", stage="queue_finished_with_missing_stages", running_stages=[])
        print(f"OLMO3_QUEUE_FINISHED_INCOMPLETE failed={failed}")


if __name__ == "__main__":
    main()
