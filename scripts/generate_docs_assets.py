#!/usr/bin/env python3
"""
Regenerate documentation figures and scheduling GIFs from evaluation artifacts.

Does not retrain models or modify experimental results. Reads pickled evaluation
outputs under files/data/ and writes assets under docs/images and docs/gifs.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "files" / "data"
IMG = ROOT / "docs" / "images"
GIF = ROOT / "docs" / "gifs"

COLORS = {
    "DQN": "#1f77b4",
    "A2C": "#ff7f0e",
    "EDD": "#9467bd",
    "FIFO": "#d62728",
    "Heuristic": "#2ca02c",
    "HDR": "#2ca02c",
}


def load(path: Path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return (
        np.asarray(d["rewards"], float),
        np.asarray(d["tardiness"], float),
        np.asarray(d["jot"], float),
        np.asarray(d["jnot"], float),
    )


def metrics(path: Path):
    rewards, _tardiness, jot, jnot = load(path)
    tot = jot + jnot
    with np.errstate(invalid="ignore", divide="ignore"):
        late = np.where(tot > 0, jnot / tot * 100.0, np.nan)
    return {
        "reward_mean": float(np.mean(rewards)),
        "jot_mean": float(np.mean(jot)),
        "jnot_mean": float(np.mean(jnot)),
        "tot_mean": float(np.mean(tot)),
        "late_mean": float(np.nanmean(late)),
        "rewards": rewards,
        "jot": jot,
        "jnot": jnot,
        "late": late,
    }


def smooth(y, w=40):
    y = np.asarray(y, float)
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def write_summary(sets: dict[str, dict]):
    lines = ["# Auto-generated benchmark summary\n"]
    for name, mset in sets.items():
        lines.append(f"## {name}\n")
        lines.append(
            "| Agent | Mean Reward | On-time Jobs | Late Jobs | Total Jobs | Late Rate (%) |\n"
        )
        lines.append("|---|---:|---:|---:|---:|---:|\n")
        for agent, m in mset.items():
            lines.append(
                f"| {agent} | {m['reward_mean']:.2f} | {m['jot_mean']:.2f} | "
                f"{m['jnot_mean']:.2f} | {m['tot_mean']:.2f} | {m['late_mean']:.4f} |\n"
            )
        lines.append("\n")
    (IMG / "benchmark_summary.md").write_text("".join(lines))


def generate_figures():
    IMG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfd",
            "axes.edgecolor": "#c9cdd6",
            "axes.grid": True,
            "grid.color": "#e6e8ef",
            "grid.linewidth": 0.8,
            "font.size": 11,
        }
    )

    primary = {
        "DQN": DATA / "dqn_data_seco_2m_2r_3b_0.6g_1000.pkl",
        "EDD": DATA / "edd_data_seco_2m_2r_3b_1000.pkl",
        "FIFO": DATA / "fifo_data_seco_2m_2r_3b_1000.pkl",
        "Heuristic": DATA / "heuristic_data_seco_2m_2r_3b_1000.pkl",
        "A2C": DATA / "a2c_data_seco_2m_2r_3b_seco_recipes_1000.pkl",
    }
    all_purpose = {
        "A2C": DATA / "a2c_data_seco_2m_2r_3b_all_purpose_1000.pkl",
        "DQN": DATA / "dqn_data_seco_2m_2r_3b_0.6g_all_purpose_1000.pkl",
        "EDD": DATA / "edd_data_seco_2m_2r_3b_step_ptd_1000.pkl",
        "Heuristic": DATA / "heuristic_data_seco_2m_2r_3b_step_ptd_1000.pkl",
        "FIFO": DATA / "fifo_data_seco_2m_2r_3b_step_ptd_1000.pkl",
    }
    rat = {
        "DQN (RAT)": DATA
        / "dqn_data_seco_2m_2r_2b_0.9g_norm_recipes_ma_obs_20M_x3_RAT_1000.pkl",
        "EDD": DATA / "edd_data_seco_2m_2r_2b_seco_recipes_j_q_ma_obs_RAT_1000.pkl",
        "FIFO": DATA / "fifo_data_seco_2m_2r_2b_seco_recipes_j_q_ma_obs_RAT_1000.pkl",
        "Heuristic": DATA
        / "heuristic_data_seco_2m_2r_2b_seco_recipes_ma_obs_RAT_1000.pkl",
    }

    primary_m = {k: metrics(v) for k, v in primary.items()}
    ap_m = {k: metrics(v) for k, v in all_purpose.items()}
    rat_m = {k: metrics(v) for k, v in rat.items()}
    write_summary(
        {"primary_2m2r3b": primary_m, "all_purpose": ap_m, "rat_2m2r2b": rat_m}
    )

    agents = list(primary_m.keys())
    late_vals = [primary_m[a]["late_mean"] for a in agents]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bars = ax.bar(
        agents, late_vals, color=[COLORS.get(a, "#555") for a in agents], width=0.62
    )
    ax.set_ylabel("Late job rate (%)")
    ax.set_title(
        "Benchmark: Late Job Rate (2 machines · 2 recipes · buffer 3 · 1000 episodes)"
    )
    for bar, val in zip(bars, late_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, max(late_vals) * 1.25)
    fig.tight_layout()
    fig.savefig(IMG / "benchmark_late_rate_primary.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(agents))
    jot = [primary_m[a]["jot_mean"] for a in agents]
    jnot = [primary_m[a]["jnot_mean"] for a in agents]
    ax.bar(x, jot, 0.62, label="Completed on time", color="#1b7f5a")
    ax.bar(x, jnot, 0.62, bottom=jot, label="Completed late", color="#c23b22")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("Mean jobs completed / episode")
    ax.set_title("Benchmark: Job Completion (same configuration)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / "benchmark_job_completion_primary.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    rew = [primary_m[a]["reward_mean"] for a in agents]
    bars = ax.bar(
        agents, rew, color=[COLORS.get(a, "#555") for a in agents], width=0.62
    )
    ax.axhline(0, color="#222", linewidth=0.8)
    ax.set_ylabel("Mean episodic reward")
    ax.set_title("Benchmark: Mean Episodic Reward")
    span = max(rew) - min(rew)
    for bar, val in zip(bars, rew):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + span * 0.02 if val >= 0 else val - span * 0.04,
            f"{val:,.0f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(IMG / "benchmark_reward_primary.png", dpi=180)
    plt.close(fig)

    agents_ap = ["DQN", "A2C", "EDD", "Heuristic", "FIFO"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(agents_ap))
    w = 0.36
    ax.bar(
        x - w / 2,
        [ap_m[a]["jot_mean"] for a in agents_ap],
        w,
        label="On time",
        color="#1b7f5a",
    )
    ax.bar(
        x + w / 2,
        [ap_m[a]["jnot_mean"] for a in agents_ap],
        w,
        label="Late",
        color="#c23b22",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(agents_ap)
    ax.set_ylabel("Mean jobs / episode")
    ax.set_title("All-purpose machines: throughput vs tardiness trade-off")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / "benchmark_all_purpose_jobs.png", dpi=180)
    plt.close(fig)

    agents_r = list(rat_m.keys())
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    vals = [rat_m[a]["late_mean"] for a in agents_r]
    bars = ax.bar(
        agents_r, vals, color=["#1f77b4", "#9467bd", "#d62728", "#2ca02c"], width=0.62
    )
    ax.set_ylabel("Late job rate (%)")
    ax.set_title("Refresh-arrival-time setting: late rate (2m · 2r · buffer 2)")
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(IMG / "benchmark_late_rate_rat.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for agent in ["DQN", "EDD", "FIFO", "Heuristic"]:
        axes[0].plot(
            smooth(primary_m[agent]["late"]),
            label=agent,
            color=COLORS[agent],
            linewidth=1.8,
        )
        axes[1].plot(
            smooth(primary_m[agent]["jot"]),
            label=agent,
            color=COLORS[agent],
            linewidth=1.8,
        )
    axes[0].set_title("Evaluation late-rate trajectory (smoothed)")
    axes[0].set_xlabel("Episode (smoothed window)")
    axes[0].set_ylabel("Late rate (%)")
    axes[0].legend(frameon=False)
    axes[1].set_title("On-time completions trajectory (smoothed)")
    axes[1].set_xlabel("Episode (smoothed window)")
    axes[1].set_ylabel("Jobs completed on time")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / "evaluation_learning_style_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    boxes = [
        (0.2, 1.2, "Orders &\nRecipes"),
        (2.2, 1.2, "Factory\nEnvironment"),
        (4.2, 1.2, "State\nRepresentation"),
        (6.2, 1.2, "RL Policy\n(DQN / A2C / PPO)"),
        (8.4, 1.2, "Action\n(Job, Machine)"),
    ]
    for x0, y0, label in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                1.7,
                1.3,
                boxstyle="round,pad=0.02,rounding_size=0.15",
                facecolor="#eef3fb",
                edgecolor="#3b6ea5",
                linewidth=1.5,
            )
        )
        ax.text(x0 + 0.85, y0 + 0.65, label, ha="center", va="center", fontsize=10)
    for i in range(len(boxes) - 1):
        ax.annotate(
            "",
            xy=(boxes[i + 1][0], 1.85),
            xytext=(boxes[i][0] + 1.7, 1.85),
            arrowprops={"arrowstyle": "->", "color": "#3b6ea5", "lw": 1.6},
        )
    ax.text(
        5.5,
        3.1,
        "Industrial Scheduling Control Loop",
        ha="center",
        fontsize=14,
        color="#1f2a44",
    )
    ax.text(
        5.5,
        0.45,
        "Reward: on-time completions (+) · lateness / illegal actions / idle under-utilization (−)",
        ha="center",
        fontsize=9,
        color="#445",
    )
    fig.tight_layout()
    fig.savefig(IMG / "system_control_loop.png", dpi=180)
    plt.close(fig)
    print(f"Wrote figures to {IMG}")


def snapshot_frame(env, step_idx, reward, info, out_path, title_prefix):
    machines = env.get_machines()
    pending = env.get_pending_jobs()
    n_m = len(machines)
    fig, axes = plt.subplots(
        1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [2.2, 1.2]}
    )
    ax = axes[0]
    ax.set_xlim(0, 1.25)
    ax.set_ylim(-0.5, n_m - 0.5)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels([f"M{i}" for i in range(n_m)])
    ax.set_xlabel("Tray capacity utilization")
    ax.set_title("Factory floor state")
    recipe_colors = {
        0: "#3b82f6",
        1: "#f59e0b",
        2: "#10b981",
        3: "#ef4444",
        4: "#8b5cf6",
    }

    for i, m in enumerate(machines):
        ax.add_patch(
            Rectangle((0, i - 0.32), 1, 0.64, facecolor="#eef1f6", edgecolor="#c5ccd8")
        )
        tray = max(m.get_tray_capacity(), 1)
        x = 0.0
        for job in m.get_pending_jobs():
            w = job.get_tray_capacity() / tray
            rid = job.get_recipes()[0].get_id() if job.get_recipes() else 0
            ax.add_patch(
                Rectangle(
                    (x, i - 0.32),
                    w,
                    0.64,
                    facecolor=recipe_colors.get(int(rid) % 5, "#64748b"),
                    edgecolor="white",
                    linewidth=0.6,
                    alpha=0.55,
                )
            )
            x += w
        x = 0.0
        for job in m.get_active_jobs():
            w = job.get_tray_capacity() / tray
            rid = job.get_recipes()[0].get_id() if job.get_recipes() else 0
            ax.add_patch(
                Rectangle(
                    (x, i - 0.22),
                    w,
                    0.44,
                    facecolor=recipe_colors.get(int(rid) % 5, "#64748b"),
                    edgecolor="#111827",
                    linewidth=0.8,
                    alpha=0.95,
                )
            )
            x += w
        ax.text(
            1.02,
            i,
            "BUSY" if not m.is_available() else "IDLE",
            va="center",
            fontsize=9,
            color="#334155",
        )

    ax = axes[1]
    ax.set_title("Pending buffer urgency")
    ratios = [j.get_process_time_deadline_ratio() for j in pending] or [0]
    deadlines = [j.get_steps_to_deadline() for j in pending] or [0]
    y = np.arange(len(pending))
    colors = ["#dc2626" if d < 0 else "#2563eb" for d in deadlines]
    ax.barh(y, ratios, color=colors, height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels([f"J{i}" for i in range(len(pending))])
    ax.set_xlabel("process-time / deadline")
    ax.invert_yaxis()

    jot = info.get("JOBS_COMPLETED_ON_TIME", env.get_jobs_completed_on_time())
    jnot = info.get("JOBS_NOT_COMPLETED_ON_TIME", env.get_jobs_completed_not_on_time())
    fig.suptitle(
        f"{title_prefix} · step {step_idx} · t={info.get('CURRENT_TIME', env.factory_time):.0f} · "
        f"reward={reward:.1f} · on-time={jot} late={jnot}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def generate_gifs(n_steps: int = 90, every: int = 2, seed: int = 7):
    from custom_environment.environment_factory import init_custom_factory_env
    from heuristic_agent import get_heuristic_action

    GIF.mkdir(parents=True, exist_ok=True)
    tmp_dir = GIF / "_frames"
    tmp_dir.mkdir(exist_ok=True)
    np.random.seed(seed)

    def run_policy(policy_name: str, choose_action, outfile: Path):
        env = init_custom_factory_env(
            is_verbose=False,
            max_steps=5_000,
            buffer_size=3,
            n_recipes=2,
            n_machines=2,
            is_evaluation=True,
            job_deadline_ratio=0.3,
        )
        _obs, info = env.reset()
        frames = []
        reward = 0.0
        info = {
            "CURRENT_TIME": 0,
            "JOBS_COMPLETED_ON_TIME": 0,
            "JOBS_NOT_COMPLETED_ON_TIME": 0,
        }
        for step in range(n_steps):
            if step % every == 0:
                fp = tmp_dir / f"{policy_name}_{step:04d}.png"
                snapshot_frame(env, step, reward, info, fp, policy_name)
                frames.append(Image.open(fp).convert("P", palette=Image.ADAPTIVE))
            action = choose_action(env)
            _obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        if frames:
            frames[0].save(
                outfile,
                save_all=True,
                append_images=frames[1:],
                duration=180,
                loop=0,
                optimize=True,
            )
        print(f"Wrote {outfile} ({len(frames)} frames)")

    run_policy(
        "Heuristic scheduling",
        get_heuristic_action,
        GIF / "scheduling_episode_heuristic.gif",
    )
    run_policy(
        "Random policy",
        lambda env: env.action_space.sample(),
        GIF / "scheduling_episode_random.gif",
    )
    for p in tmp_dir.glob("*.png"):
        p.unlink()
    tmp_dir.rmdir()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--gifs-only", action="store_true")
    args = parser.parse_args()
    if args.gifs_only:
        generate_gifs()
    elif args.figures_only:
        generate_figures()
    else:
        generate_figures()
        generate_gifs()


if __name__ == "__main__":
    main()
