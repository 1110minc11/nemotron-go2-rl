"""Charts and a markdown table for the obstacle experiment (no Isaac Sim needed).

    python scripts/report_avoid.py --eval outputs/eval/stage2.json --run "ours=<run>" --run "control=<run>" --out docs/results_stage2
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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


def scalars(run_dir, tags):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    ea = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in tags:
        if tag in ea.Tags()["scalars"]:
            ev = ea.Scalars(tag)
            out[tag] = ([e.step for e in ev], [e.value for e in ev])
    return out


def smooth(v, k=8):
    out, acc = [], []
    for x in v:
        acc.append(x)
        acc = acc[-k:]
        out.append(sum(acc) / len(acc))
    return out


def training_figure(runs, path):
    panels = [("Train/mean_reward", "평균 에피소드 보상"), ("Episode_Termination/goal_reached", "목표 도달로 끝난 에피소드 비율"),
              ("Episode_Termination/obstacle_collision", "장애물 충돌로 끝난 비율"), ("Metrics/goal_pose/curriculum_level", "커리큘럼 난이도 (Nemotron 학습만 단계 상승)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), dpi=150, facecolor=SURFACE)
    data = {lb: scalars(p, [t for t, _ in panels]) for lb, p in runs}
    for ax, (tag, title) in zip(axes.flat, panels):
        style(ax, title)
        for i, (lb, _) in enumerate(runs):
            if tag in data[lb]:
                x, y = data[lb][tag]
                ax.plot(x, smooth(y), color=SERIES[i], linewidth=2, label=lb)
        ax.set_xlabel("PPO iteration", color=TEXT_2, fontsize=8.5)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", frameon=False, fontsize=9, ncol=len(runs))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def success_figure(res, path):
    labels = list(res["policies"])
    cases = [r["case"] for r in res["policies"][labels[0]]["cases"]]
    n, h = len(labels), 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(9, 0.46 * len(cases) + 1.5), dpi=150, facecolor=SURFACE)
    style(ax, "시나리오별 성공률 (충돌 없이 목표 0.4 m 이내)")
    for i, lb in enumerate(labels):
        rate = {r["case"]: r["success_rate"] for r in res["policies"][lb]["cases"]}
        ys = [k + (i - (n - 1) / 2) * h for k in range(len(cases))]
        ax.barh(ys, [rate[c] * 100 for c in cases], height=h - 0.04, color=SERIES[i], label=lb)
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels(cases, fontsize=8.5, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("성공률 (%)", color=TEXT_2, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=n, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def table(res, cases):
    labels = list(res["policies"])
    by = {c["id"]: c for c in cases}
    rows = ["| 시나리오 | 유형 | 난이도 | " + " | ".join(f"{lb} (성공 / 충돌 / 시간초과)" for lb in labels) + " |", "|" + "---|" * (3 + len(labels))]
    for k, r in enumerate(res["policies"][labels[0]]["cases"]):
        cells = []
        for lb in labels:
            rr = res["policies"][lb]["cases"][k]
            cells.append(f"{rr['success_rate'] * 100:.0f}% / {rr['collision_rate'] * 100:.0f}% / {rr['timeout_rate'] * 100:.0f}%")
        c = by.get(r["case"], {})
        rows.append(f"| `{r['case']}` | {c.get('family', '')} | {c.get('difficulty', '')} | " + " | ".join(cells) + " |")
    rows += ["", "| 정책 | 성공률(평균) | 성공률(가중) | 충돌 | 넘어짐 | 시간 초과 |", "|---|---|---|---|---|---|"]
    for lb in labels:
        o = res["policies"][lb]["overall"]
        rows.append(f"| {lb} | {o['success_rate'] * 100:.1f}% | {o['success_rate_weighted'] * 100:.1f}% | {o['collision_rate'] * 100:.1f}% | "
                    f"{o['fall_rate'] * 100:.1f}% | {o['timeout_rate'] * 100:.1f}% |")
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", required=True)
    p.add_argument("--run", action="append", default=[])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    for font in ("NanumGothic", "Noto Sans CJK KR"):
        if any(font in f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(a.out, exist_ok=True)
    res = json.load(open(a.eval, encoding="utf-8"))
    cases = json.load(open(res["catalog"], encoding="utf-8"))["cases"]
    success_figure(res, os.path.join(a.out, "success_by_scenario.png"))
    if a.run:
        training_figure([tuple(r.split("=", 1)) for r in a.run], os.path.join(a.out, "training_curves.png"))
    t = table(res, cases)
    open(os.path.join(a.out, "results_table.md"), "w", encoding="utf-8").write(t + "\n")
    print(t)


if __name__ == "__main__":
    main()
