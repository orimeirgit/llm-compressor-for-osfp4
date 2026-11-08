"""Collect the 5-method x (8 tasks + WikiText-2) matrix into benchmark-results.{json,csv}."""

import csv
import json
import statistics
from pathlib import Path

R = Path("/workspace/osfp4-runs/olmo3-7b-instruct")
PROFILE = "selected9-olmo3-v1"
protocol = json.loads(Path("/workspace/osfp4-olmo3-harness/benchmark_protocol.json").read_text())
METHODS = [("bf16", "bf16"), ("osfp4_rtn_w4a16", "nvfp4a16-rtn"), ("osfp4_sic_w4a16", "nvfp4a16-sic"),
           ("osfp4_rtn_w4a4", "nvfp4-rtn"), ("osfp4_sic_w4a4", "nvfp4-sic")]

table, missing = {}, []
for mid, ckpt in METHODS:
    row = {}
    for task in protocol["order"]:
        runner = "lmeval" if protocol["tasks"][task]["runner"] == "lmeval" else "generate"
        path = R / f"benchmarks-{PROFILE}" / mid / runner / task / "results.json"
        res = json.loads(path.read_text()) if path.exists() else None
        if res is None or res.get("limit") is not None:
            missing.append(f"{mid}/{task}")
            continue
        row[task] = {"score_pct": 100 * res["score"], "cap_hits": res.get("cap_hits")}
    wiki = R / f"{ckpt}-wikitext2.json"
    if wiki.exists() and json.loads(wiki.read_text()).get("limit_windows") is None:
        row["wikitext2_ppl"] = json.loads(wiki.read_text())["perplexity"]
    else:
        missing.append(f"{mid}/wikitext2")
    table[mid] = row

for mid, row in table.items():
    for task in protocol["order"]:
        if task in row and task in table["bf16"]:
            row[task]["delta_pp_vs_bf16"] = row[task]["score_pct"] - table["bf16"][task]["score_pct"]
    scores = [row[t]["score_pct"] for t in protocol["order"] if t in row]
    row["mean_8_tasks_pct"] = statistics.mean(scores) if len(scores) == len(protocol["order"]) else None

status = "selected_9_evaluation_matrix_complete" if not missing else "incomplete"
(R / "benchmark-results.json").write_text(json.dumps(
    {"status": status, "profile": PROFILE, "protocol": protocol["protocol"], "missing": missing,
     "results": table}, indent=2))
with (R / "benchmark-results.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "task", "score_pct", "delta_pp_vs_bf16", "cap_hits"])
    for mid, row in table.items():
        for task in protocol["order"]:
            if task in row:
                w.writerow([mid, task, round(row[task]["score_pct"], 3),
                            None if row[task].get("delta_pp_vs_bf16") is None else round(row[task]["delta_pp_vs_bf16"], 3),
                            row[task]["cap_hits"]])
        w.writerow([mid, "wikitext2_ppl", row.get("wikitext2_ppl"), None, None])
print(f"AGGREGATE status={status} missing={len(missing)}")
