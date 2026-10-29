"""Record one Go2 working through the obstacle scenarios (top-down camera, scenario / status overlay).

    python scripts/record_avoid.py --headless --enable_cameras --policy "label=<ckpt>" --episodes 12 --out outputs/videos/avoid.mp4
"""

import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--policy", required=True, help="label=path/to/model.pt")
parser.add_argument("--catalog", default=None)
parser.add_argument("--locomotion", default=None)
parser.add_argument("--episodes", type=int, default=12)
parser.add_argument("--case_offset", type=int, default=0, help="first scenario index of the tour")
parser.add_argument("--case", default=None, help="repeat this scenario id for every episode instead of touring all")
parser.add_argument("--max_steps", type=int, default=150, help="5 Hz steps per episode")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--out", default=os.path.join(REPO, "outputs", "videos", "avoid.mp4"))
parser.add_argument("--font", default="/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
parser.add_argument("--font_bold", default="/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
if args_cli.catalog:
    os.environ["NEMOTRON_GO2_OBSTACLES"] = os.path.abspath(args_cli.catalog)
if args_cli.locomotion:
    os.environ["NEMOTRON_GO2_LOCOMOTION"] = os.path.abspath(args_cli.locomotion)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import nemotron_go2_rl.tasks  # noqa: E402, F401

TASK = "Nemotron-Go2-Obstacles-Record"
FAIL_TERMS = {"obstacle_collision": "충돌", "non_foot_contact": "몸체 접촉", "base_contact": "넘어짐", "bad_orientation": "전도"}


def main():
    label, path = args_cli.policy.rsplit("=", 1)
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    cfg.seed = args_cli.seed
    cfg.commands.goal_pose.seed = args_cli.seed
    cfg.commands.goal_pose.case_offset = args_cli.case_offset
    if args_cli.case:
        from nemotron_go2_rl.obstacles.schema import load_catalog
        from nemotron_go2_rl.tasks.field_env_cfg import resolve_catalog_path

        ids = [c["id"] for c in load_catalog(resolve_catalog_path())["cases"]]
        cfg.commands.goal_pose.assignment = "round_robin"
        cfg.commands.goal_pose.case_offset = ids.index(args_cli.case)
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.device = cfg.sim.device
    base = gym.make(TASK, cfg=cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.path.abspath(path), load_optimizer=False)
    policy = runner.get_inference_policy(device=base.unwrapped.device)
    term = base.unwrapped.command_manager.get_term("goal_pose")
    catalog = term.catalog
    titles = {c["id"]: c for c in catalog["cases"]}
    robot = base.unwrapped.scene["robot"]
    tm = base.unwrapped.termination_manager
    width, height = cfg.viewer.resolution
    f_big, f_mid, f_small = (ImageFont.truetype(args_cli.font_bold, 30), ImageFont.truetype(args_cli.font, 22),
                             ImageFont.truetype(args_cli.font, 17))
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    writer = imageio.get_writer(args_cli.out, fps=5, codec="libx264", quality=7, macro_block_size=1, ffmpeg_log_level="error")

    def draw(frame, case, ep, step, dist, status, color):
        img = Image.fromarray(frame[..., :3]).convert("RGBA")
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle((16, 16, 640, 150), radius=12, fill=(10, 12, 16, 175))
        d.text((32, 24), case["title_ko"], font=f_big, fill=(255, 255, 255, 255))
        d.text((32, 66), f"{case['title']} · {case['family']} · 난이도 {case['difficulty']}", font=f_small, fill=(190, 198, 210, 255))
        d.text((32, 96), f"목표까지 {dist:4.2f} m · {step * 0.2:5.1f} s", font=f_mid, fill=(230, 230, 230, 255))
        d.text((32, 122), status, font=f_mid, fill=color)
        tw = d.textlength(label, font=f_mid)
        d.rounded_rectangle((width - tw - 48, 16, width - 16, 64), radius=12, fill=(10, 12, 16, 175))
        d.text((width - tw - 32, 26), label, font=f_mid, fill=(255, 255, 255, 255))
        d.text((24, height - 34), f"에피소드 {ep}/{args_cli.episodes} · Nemotron이 설계한 장애물 경우 → Isaac Lab PPO · Unitree Go2",
               font=f_small, fill=(235, 235, 235, 230))
        return np.asarray(Image.alpha_composite(img, layer).convert("RGB"))

    results = []
    with torch.inference_mode():
        obs, _ = env.reset()
        for ep in range(1, args_cli.episodes + 1):
            case = titles[term.case_names[int(term.case_index[0])]]
            outcome, steps = "시간 초과", 0
            for step in range(args_cli.max_steps):
                dist = float((term.goal_pos_w[0] - robot.data.root_pos_w[0, :2]).norm())
                frame = base.unwrapped.render()
                writer.append_data(draw(frame, case, ep, step, dist, "주행 중", (255, 255, 255, 255)))
                obs, _, dones, _ = env.step(policy(obs))
                steps = step + 1
                if bool(dones[0]):
                    if bool(tm.get_term("goal_reached")[0]):
                        outcome = "목표 도달"
                    elif bool(tm.get_term("time_out")[0]):
                        outcome = "시간 초과"
                    else:
                        outcome = next((v for k, v in FAIL_TERMS.items() if bool(tm.get_term(k)[0])), "종료")
                    break
            ok = outcome == "목표 도달"
            color = (120, 235, 140, 255) if ok else ((255, 140, 100, 255) if outcome != "시간 초과" else (255, 210, 120, 255))
            last = draw(frame, case, ep, steps, 0.0 if ok else dist, outcome, color)
            for _ in range(8):
                writer.append_data(last)
            results.append({"episode": ep, "case": case["id"], "outcome": outcome, "steps": steps})
            print(f"[REC] episode {ep}: {case['id']} -> {outcome} ({steps} steps)", flush=True)
    writer.close()
    print(f"[INFO] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
