"""Record one Go2 touring every walking case of a catalog, with a following camera and overlay.

    python scripts/record.py --headless --enable_cameras \
        --policy "Nemotron cases (ours)=logs/rsl_rl/nemotron_go2_walk_flat/<run>/model_1499.pt" \
        --out outputs/videos/ours.mp4
"""

import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.stdout.reconfigure(line_buffering=True)  # Isaac Sim exits without flushing stdout

from isaaclab.app import AppLauncher  # noqa: E402

from nemotron_go2_rl import DEFAULT_CATALOG  # noqa: E402

parser = argparse.ArgumentParser(description="Scripted case tour video.")
parser.add_argument("--policy", required=True, help="label=path/to/model.pt")
parser.add_argument("--catalog", default=DEFAULT_CATALOG)
parser.add_argument("--out", default=os.path.join(REPO, "outputs", "videos", "tour.mp4"))
parser.add_argument("--segment_s", type=float, default=3.0, help="seconds per showcase command")
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--cases", default=None, help="comma-separated case ids (default: all, by difficulty)")
parser.add_argument("--title_s", type=float, default=2.5, help="title card length (0 to skip)")
parser.add_argument("--font", default="/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
parser.add_argument("--font_bold", default="/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
os.environ["NEMOTRON_GO2_CATALOG"] = os.path.abspath(args_cli.catalog)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import nemotron_go2_rl.tasks  # noqa: E402, F401
from nemotron_go2_rl.tasks.runtime import case_command_term, load_policy, showcase_commands  # noqa: E402
from nemotron_go2_rl.tasks.walk_env_cfg import apply_catalog  # noqa: E402

TASK = "Nemotron-Go2-Walk-Flat-Record"
AXES = (("vx", "m/s", 1.0), ("vy", "m/s", 0.6), ("wz", "rad/s", 1.5))


class Overlay:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.f_big = ImageFont.truetype(args_cli.font_bold, 34)
        self.f_mid = ImageFont.truetype(args_cli.font, 22)
        self.f_small = ImageFont.truetype(args_cli.font, 18)

    def title_card(self, lines):
        img = Image.new("RGB", (self.w, self.h), (18, 20, 24))
        d = ImageDraw.Draw(img)
        y = self.h // 2 - 30 * len(lines)
        for i, text in enumerate(lines):
            font = self.f_big if i == 0 else self.f_mid
            tw = d.textlength(text, font=font)
            d.text(((self.w - tw) / 2, y), text, font=font, fill=(235, 235, 235) if i == 0 else (170, 178, 190))
            y += 56 if i == 0 else 36
        return np.asarray(img)

    def draw(self, frame, case, idx, total, label, cmd, vel, t):
        img = Image.fromarray(frame[..., :3]).convert("RGBA")
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle((16, 16, 560, 252), radius=12, fill=(10, 12, 16, 170))
        d.text((32, 26), case["title_ko"], font=self.f_big, fill=(255, 255, 255, 255))
        sub = f"{case['title']}  ·  {case['category']}  ·  난이도 {case['difficulty']}"
        d.text((32, 72), sub, font=self.f_small, fill=(190, 198, 210, 255))
        for k, (name, unit, scale) in enumerate(AXES):
            y = 106 + k * 38
            d.text((32, y), name, font=self.f_mid, fill=(230, 230, 230, 255))
            x0, x1, yc = 80, 330, y + 14
            d.rectangle((x0, yc - 8, x1, yc + 8), fill=(60, 64, 72, 230))
            mid = (x0 + x1) / 2
            half = (x1 - x0) / 2
            cx = mid + half * max(-1.0, min(1.0, cmd[k] / scale))
            mx = mid + half * max(-1.0, min(1.0, vel[k] / scale))
            d.rectangle((min(mid, mx), yc - 6, max(mid, mx), yc + 6), fill=(80, 160, 255, 255))
            d.line((cx, yc - 13, cx, yc + 13), fill=(120, 230, 120, 255), width=4)
            d.line((mid, yc - 10, mid, yc + 10), fill=(200, 200, 200, 255), width=1)
            d.text((345, y), f"명령 {cmd[k]:+.2f} / 실제 {vel[k]:+.2f} {unit}", font=self.f_small, fill=(230, 230, 230, 255))
        d.text((32, 222), "초록 선 = 명령, 파랑 막대 = 실제 속도 (몸체 기준)", font=self.f_small, fill=(160, 168, 180, 255))

        tw = d.textlength(label, font=self.f_mid)
        d.rounded_rectangle((self.w - tw - 48, 16, self.w - 16, 92), radius=12, fill=(10, 12, 16, 170))
        d.text((self.w - tw - 32, 24), label, font=self.f_mid, fill=(255, 255, 255, 255))
        d.text((self.w - tw - 32, 58), f"case {idx}/{total}   t = {t:5.1f} s", font=self.f_small, fill=(190, 198, 210, 255))
        foot = "Nemotron이 만든 보행 경우의 수  >  Isaac Lab PPO  ·  Unitree Go2  ·  빈 평지"
        d.text((24, self.h - 36), foot, font=self.f_small, fill=(235, 235, 235, 230))
        return np.asarray(Image.alpha_composite(img, layer).convert("RGB"))


def main():
    label, path = args_cli.policy.rsplit("=", 1)
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    catalog = apply_catalog(env_cfg, args_cli.catalog, randomize=False)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.device = env_cfg.sim.device
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg, render_mode="rgb_array"), clip_actions=agent_cfg.clip_actions)
    policy = load_policy(env, agent_cfg, os.path.abspath(path))
    term = case_command_term(env)
    robot = env.unwrapped.scene["robot"]
    dev = env.unwrapped.device

    cases = catalog["cases"]
    if args_cli.cases:
        wanted = args_cli.cases.split(",")
        cases = [c for c in cases if c["id"] in wanted]
    else:
        order = {c: i for i, c in enumerate(("stand", "forward", "backward", "turn_in_place", "arc", "lateral",
                                             "diagonal", "omni"))}
        cases = sorted(cases, key=lambda c: (c["difficulty"], order.get(c["category"], 9)))
    script = [(c, cmd) for c in cases for cmd in showcase_commands(c)]

    step_dt = env.unwrapped.step_dt
    seg_steps = int(round(args_cli.segment_s / step_dt))
    stride = max(1, int(round(1.0 / (args_cli.fps * step_dt))))
    width, height = env_cfg.viewer.resolution
    overlay = Overlay(width, height)
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    writer = imageio.get_writer(args_cli.out, fps=args_cli.fps, codec="libx264", quality=7, macro_block_size=1,
                                ffmpeg_log_level="error")
    if args_cli.title_s > 0:
        card = overlay.title_card([
            "Stage 1 · 빈 평지에서 걷기",
            f"{label}",
            f"{len(cases)}개 경우의 수 · {catalog['generator']['model']}",
        ])
        for _ in range(int(args_cli.title_s * args_cli.fps)):
            writer.append_data(card)

    case_pos = {c["id"]: i + 1 for i, c in enumerate(cases)}
    obs, _ = env.reset()
    vel_ema = torch.zeros(3, device=dev)
    t = 0.0
    with torch.inference_mode():
        for case, cmd in script:
            term.set_command(cmd.to(dev).unsqueeze(0), case_index=term.case_names.index(case["id"]))
            for s in range(seg_steps):
                obs, _, _, _ = env.step(policy(obs))
                t += step_dt
                v = torch.cat([robot.data.root_lin_vel_b[0, :2], robot.data.root_ang_vel_b[0, 2:3]])
                vel_ema = 0.9 * vel_ema + 0.1 * v
                if s % stride == 0:
                    frame = env.unwrapped.render()
                    writer.append_data(
                        overlay.draw(frame, case, case_pos[case["id"]], len(cases), label, cmd.tolist(),
                                     vel_ema.tolist(), t)
                    )
    writer.close()
    print(f"[INFO] wrote {args_cli.out} ({t:.1f} s of simulation, {len(script)} segments)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
