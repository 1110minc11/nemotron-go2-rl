"""Turn a per-case evaluation (scripts/evaluate.py) into findings for the next Nemotron round."""

from __future__ import annotations

import json


def findings_from_eval(eval_path: str, policy: str, threshold: float = 0.9) -> list[str]:
    """One sentence per case whose success rate is below ``threshold``, with the measured gap."""
    with open(eval_path, encoding="utf-8") as f:
        results = json.load(f)
    if policy not in results["policies"]:
        raise KeyError(f"{policy!r} not in {eval_path}; available: {list(results['policies'])}")
    rows = results["policies"][policy]["cases"]
    findings = []
    for r in rows:
        if r["success_rate"] >= threshold:
            continue
        cmd = ", ".join(f"{x:+.2f}" for x in r["cmd_mean"])
        if r["vel_mean"] is None:
            findings.append(f"Case {r['case']}: every robot fell (command [{cmd}]).")
            continue
        vel = ", ".join(f"{x:+.2f}" for x in r["vel_mean"])
        msg = (
            f"Case {r['case']}: only {r['success_rate'] * 100:.0f}% of robots tracked the command within 20%; "
            f"mean command [{cmd}] -> measured [{vel}] (vx, vy, wz)"
        )
        if r["fall_rate"] > 0:
            msg += f", {r['fall_rate'] * 100:.0f}% fell"
        findings.append(msg + ".")
    passed = [r["case"] for r in rows if r["success_rate"] >= threshold]
    if passed:
        findings.append(f"These cases already pass (>= {threshold * 100:.0f}% success): {', '.join(passed)}.")
    return findings
