"""Olmo-3-7B-Instruct OSFP4 queue: 4 calibrations, then 5 methods x (WikiText-2 + 8 tasks).

Restart-safe: a stage is skipped when its validated output exists. A failed stage is
recorded and the queue moves on; rerun this script to retry only what is missing.
Holds /tmp/osfp4-olmo3.lock for the whole matrix (one GPU job at a time).
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

H = Path("/workspace/osfp4-olmo3-harness")
R = Path("/workspace/osfp4-runs/olmo3-7b-instruct")
MODEL = Path("/workspace/osfp4-cache/hub/models--allenai--Olmo-3-7B-Instruct/snapshots/"
             "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc")
REVISION = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"
CAL_PY = "/workspace/venvs/osfp4/bin/python"
EVAL_PY = "/workspace/venvs/osfp4-eval/bin/python"
PROTOCOL = H / "benchmark_protocol.json"
PROFILE = "selected9-olmo3-v1"
LOCK = "/tmp/osfp4-olmo3.lock"

METHODS = [  # (method id, checkpoint dir, scheme, mode)
    ("bf16", None, None, None),
    ("osfp4_rtn_w4a16", "nvfp4a16-rtn", "NVFP4A16", "rtn"),
    ("osfp4_sic_w4a16", "nvfp4a16-sic", "NVFP4A16", "sic"),
    ("osfp4_rtn_w4a4", "nvfp4-rtn", "NVFP4", "rtn"),
    ("osfp4_sic_w4a4", "nvfp4-sic", "NVFP4", "sic"),
]
STATUS = R / "benchmarks-status.json"
ENV = {**os.environ, "HF_HOME": "/workspace/osfp4-cache", "TOKENIZERS_PARALLELISM": "false",
       "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"}


def write_status(**fields):
    state = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    state.update(fields, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, STATUS)


def run_stage(name: str, cmd: list[str]) -> bool:
    log = R / "benchmark-logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    write_status(status="running", stage=name, command=" ".join(cmd))
    print(f"\n===== {time.strftime('%H:%M:%S')} START {name}", flush=True)
    with log.open("a") as f:
        f.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(cmd, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            f.write(line)
            sys.stdout.write(line)
        rc = proc.wait()
    print(f"===== {time.strftime('%H:%M:%S')} END {name} rc={rc}", flush=True)
    return rc == 0


def checkpoint(method):
    _, ckpt, _, _ = method
    return MODEL if ckpt is None else R / "checkpoints" / ckpt


def calibration_passed(ckpt: str) -> bool:
    manifest = R / "checkpoints" / f"{ckpt}-calibration.json"
    return (R / "checkpoints" / ckpt).is_dir() and manifest.exists() and \
        json.loads(manifest.read_text()).get("status") == "passed"


def main():
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another queue holds {LOCK}; not starting a second one")
    protocol = json.loads(PROTOCOL.read_text())
    failed, completed = [], []
    write_status(status="running", profile=PROFILE, model=str(MODEL), revision=REVISION,
                 failed_stages=[], completed_stages=[])

    for method in METHODS[1:]:
        mid, ckpt, scheme, mode = method
        name = f"{mid}-calibration"
        if calibration_passed(ckpt):
            continue
        ok = run_stage(name, [CAL_PY, "-u", str(H / "calibrate.py"), "--model", str(MODEL),
                              "--revision", REVISION, "--scheme", scheme, "--mode", mode,
                              "--out", str(R / "checkpoints" / ckpt)])
        (completed if ok and calibration_passed(ckpt) else failed).append(name)
        write_status(failed_stages=failed, completed_stages=completed)

    for method in METHODS:
        mid, ckpt = method[0], method[1]
        if ckpt is not None and not calibration_passed(ckpt):
            failed.append(f"{mid}-*: no validated checkpoint")
            continue
        out_root = R / f"benchmarks-{PROFILE}" / mid
        wiki = R / f"{ckpt or 'bf16'}-wikitext2.json"
        stages = []
        if not wiki.exists():
            stages.append((f"{mid}-wikitext2", [EVAL_PY, "-u", str(H / "eval_wikitext.py"),
                                                "--model-path", str(checkpoint(method)), "--out", str(wiki)]))
        for task in protocol["order"]:
            cfg = protocol["tasks"][task]
            out = out_root / ("lmeval" if cfg["runner"] == "lmeval" else "generate") / task
            if (out / "results.json").exists() and json.loads((out / "results.json").read_text()).get("limit") is None:
                continue
            if cfg["runner"] == "lmeval":
                cmd = [EVAL_PY, "-u", str(H / "eval_lmeval.py"), "--model-path", str(checkpoint(method)),
                       "--task", task, "--protocol", str(PROTOCOL), "--out", str(out),
                       "--cache", str(R / "lmeval-request-cache" / mid / task)]
            else:
                cmd = [EVAL_PY, "-u", str(H / "eval_generate.py"), "--model-path", str(checkpoint(method)),
                       "--task", task, "--protocol", str(PROTOCOL), "--out", str(out)]
            stages.append((f"{mid}-{task}", cmd))
        for name, cmd in stages:
            ok = run_stage(name, cmd)
            (completed if ok else failed).append(name)
            write_status(failed_stages=failed, completed_stages=completed)

    ok = run_stage("aggregate", [EVAL_PY, "-u", str(H / "aggregate.py")])
    final = json.loads((R / "benchmark-results.json").read_text()) if (R / "benchmark-results.json").exists() else {}
    if ok and final.get("status") == "selected_9_evaluation_matrix_complete" and not failed:
        write_status(status="benchmark_matrix_complete", stage="awaiting_user_review_before_next_model")
        print("OLMO3_BENCHMARK_MATRIX_COMPLETE; STOPPED_BEFORE_ANY_NEXT_MODEL")
    else:
        write_status(status="incomplete", stage="queue_finished_with_missing_stages")
        print(f"OLMO3_QUEUE_FINISHED_INCOMPLETE failed={failed}")


if __name__ == "__main__":
    main()
