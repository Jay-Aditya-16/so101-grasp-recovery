"""Score a predeclared paired Phase II / frozen Phase III simulator evaluation.

The manifest and before/after SHA-256 values must be collected by the runner;
this tool checks consistency but cannot establish that a manifest is truthful.
No automatic selection of runs or model parameters is performed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from placement_bandit import PlacementBandit

CONDITIONS = ("phase2", "phase3_frozen")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(manifest, memory_path):
    if manifest.get("schema") != "phase3-paired-evaluation-v1":
        raise ValueError("unknown manifest schema")
    seeds = manifest.get("heldout_seeds")
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("heldout_seeds must be unique integers")
    bandit = PlacementBandit(memory_path)
    if any(type(record.get("seed")) is not int for record in bandit.records):
        raise ValueError("memory contains records with unknown training seeds")
    training_seeds = {r["seed"] for r in bandit.records}
    if training_seeds.intersection(seeds):
        raise ValueError("held-out seeds overlap with memory training seeds")
    memory_hash = sha256(memory_path)
    expected_checkpoint = manifest.get("checkpoint_sha256")
    seconds = manifest.get("horizon_s")
    if not isinstance(expected_checkpoint, str) or len(expected_checkpoint) != 64 or any(c not in '0123456789abcdef' for c in expected_checkpoint):
        raise ValueError("invalid checkpoint SHA-256")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("invalid horizon")
    by_key = {}
    for t in manifest.get("trials", []):
        condition, seed = t.get("condition"), t.get("seed")
        key = (condition, seed)
        if condition not in CONDITIONS or seed not in seeds or key in by_key:
            raise ValueError("unexpected or duplicate trial")
        if t.get("scenario") != "cube" or t.get("checkpoint_sha256") != expected_checkpoint or t.get("horizon_s") != seconds:
            raise ValueError("scenario, checkpoint, or horizon mismatch")
        if t.get("interrupted") is not False or type(t.get("task_success")) is not bool:
            raise ValueError("interrupted or unscored trial")
        pos = t.get("start_xy")
        if not isinstance(pos, list) or len(pos) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in pos):
            raise ValueError("invalid starting position")
        if condition == "phase3_frozen" and (t.get("frozen") is not True or t.get("memory_sha256_before") != memory_hash or t.get("memory_sha256_after") != memory_hash):
            raise ValueError("memory is not proven frozen by manifest checksums")
        by_key[key] = t
    if len(by_key) != len(seeds) * len(CONDITIONS):
        raise ValueError("missing paired trial")
    counts = {"both_succeed": 0, "only_phase2": 0, "only_phase3": 0, "both_fail": 0}
    for seed in seeds:
        a, b = (by_key[(condition, seed)] for condition in CONDITIONS)
        if any(abs(x - y) > 1e-6 for x, y in zip(a["start_xy"], b["start_xy"])):
            raise ValueError("paired starting positions differ")
        key = {(True, True): "both_succeed", (True, False): "only_phase2",
               (False, True): "only_phase3", (False, False): "both_fail"}[(a["task_success"], b["task_success"])]
        counts[key] += 1
    n = len(seeds)
    discordant = counts["only_phase2"] + counts["only_phase3"]
    lo = min(counts["only_phase2"], counts["only_phase3"])
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(lo + 1)) / (2 ** discordant)) if discordant else 1.
    return {"paired_trials": n, **counts,
            "phase2_successes": counts["both_succeed"] + counts["only_phase2"],
            "phase3_successes": counts["both_succeed"] + counts["only_phase3"],
            "success_difference": (counts["only_phase3"] - counts["only_phase2"]) / n,
            "mcnemar_exact_two_sided_p": p, "memory_sha256": memory_hash,
            "limitations": "Checks manifest consistency, not authenticity or causal validity. Small samples have low power."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("memory", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(json.loads(args.manifest.read_text()), args.memory), indent=2))


if __name__ == "__main__":
    main()
