"""Charts and a markdown table from evaluation JSON and training logs (no Isaac Sim needed).

    python scripts/report.py --eval outputs/eval/stage1.json \
        --run ours=logs/rsl_rl/nemotron_go2_walk_flat/<run> --run uniform=logs/.../<run> \
        --out docs/results
"""

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# validated categorical slots in fixed order (adjacent-pair safe for grouped bars), text and surface tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=22)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8.5)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load_scalars(run_dir, tags):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not files:
        raise FileNotFoundError(f"no TensorBoard events in {run_dir}")
    ea = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in tags:
        if tag in ea.Tags()["scalars"]:
            ev = ea.Scalars(tag)
            out[tag] = ([e.step for e in ev], [e.value for e in ev])
    return out


def smooth(values, k=10):
    out, acc = [], []
    for v in values:
        acc.append(v)
        if len(acc) > k:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def training_figure(runs, out_png):
    panels = [
        ("Train/mean_reward", "평균 에피소드 보상"),
        ("Train/mean_episode_length", "평균 에피소드 길이 (step, 최대 1000)"),
        ("Episode_Reward/track_lin_vel_xy", "선속도 추종 보상 (에피소드 평균)"),
        ("Episode_Reward/track_ang_vel_z", "회전속도 추종 보상 (에피소드 평균)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), dpi=150, facecolor=SURFACE)
    data = {label: load_scalars(path, [t for t, _ in panels]) for label, path in runs}
    for ax, (tag, title) in zip(axes.flat, panels):
        style(ax, title)
        for i, (label, _) in enumerate(runs):
            if tag not in data[label]:
                continue
            steps, values = data[label][tag]
            ax.plot(steps, smooth(values), color=SERIES[i], linewidth=2, label=label)
        ax.set_xlabel("PPO iteration", color=TEXT_2, fontsize=8.5)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9, ncol=len(runs))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)


def success_figure(results, out_png):
    labels = list(results["policies"])
    cases = [r["case"] for r in results["policies"][labels[0]]["cases"]]
    n = len(labels)
    height = 0.8 / n
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(cases) + 1.4), dpi=150, facecolor=SURFACE)
    style(ax, "경우별 성공률 (평균 속도 오차 20% 이내 + 넘어짐 없음)")
    for i, label in enumerate(labels):
        rates = {r["case"]: r["success_rate"] for r in results["policies"][label]["cases"]}
        ys = [k + (i - (n - 1) / 2) * height for k in range(len(cases))]
        ax.barh(ys, [rates[c] * 100 for c in cases], height=height - 0.04, color=SERIES[i], label=label)
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels(cases, fontsize=8.5, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("성공률 (%)", color=TEXT_2, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=n, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)


def markdown_table(results, catalog_cases):
    labels = list(results["policies"])
    by_case = {c["id"]: c for c in catalog_cases}
    head = "| 경우 | 분류 | 난이도 | 명령 평균 [vx, vy, wz] | " + " | ".join(f"{lb} 성공 / 실제 속도" for lb in labels) + " |"
    sep = "|" + "---|" * (4 + len(labels))
    rows = [head, sep]
    first = results["policies"][labels[0]]["cases"]
    for k, r in enumerate(first):
        c = by_case.get(r["case"], {})
        cmd = ", ".join(f"{x:+.2f}" for x in r["cmd_mean"])
        cells = []
        for lb in labels:
            rr = results["policies"][lb]["cases"][k]
            vel = ", ".join(f"{x:+.2f}" for x in rr["vel_mean"]) if rr["vel_mean"] else "넘어짐"
            fall = f", 넘어짐 {rr['fall_rate']*100:.0f}%" if rr["fall_rate"] > 0 else ""
            cells.append(f"{rr['success_rate']*100:.0f}% / [{vel}]{fall}")
        rows.append(f"| `{r['case']}` | {c.get('category', '')} | {c.get('difficulty', '')} | [{cmd}] | " + " | ".join(cells) + " |")
    rows.append("")
    rows.append("| 정책 | 경우 평균 성공률 | 가중 성공률 | 평균 넘어짐 |")
    rows.append("|---|---|---|---|")
    for lb in labels:
        o = results["policies"][lb]["overall"]
        rows.append(f"| {lb} | {o['success_rate']*100:.1f}% | {o['success_rate_weighted']*100:.1f}% | {o['fall_rate']*100:.1f}% |")
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", required=True)
    p.add_argument("--run", action="append", default=[], help="label=run_dir (repeatable)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    for font in ("NanumGothic", "Noto Sans CJK KR"):
        if any(font in f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

    os.makedirs(args.out, exist_ok=True)
    with open(args.eval, encoding="utf-8") as f:
        results = json.load(f)
    with open(results["catalog"], encoding="utf-8") as f:
        catalog = json.load(f)

    success_figure(results, os.path.join(args.out, "success_by_case.png"))
    if args.run:
        training_figure([tuple(r.split("=", 1)) for r in args.run], os.path.join(args.out, "training_curves.png"))
    table = markdown_table(results, catalog["cases"])
    with open(os.path.join(args.out, "results_table.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(table)
    print(f"\n[report] wrote charts and table to {args.out}")


if __name__ == "__main__":
    main()
