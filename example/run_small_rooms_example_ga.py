import os
import gc
import random
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from example.small_rooms_env import SmallRoomsEnv
from primitive_option import PrimitiveOption
from example.Options.storeOption import StoreOption
from example.Options.DeliverOption import DeliverOption
from example.Options.pickupOption import PickupOption
from example.Options.PickupRipeOption import PickupRipeOption
from example.Options.GAStorageSelectOption import GAStorageSelectOption
from options_agent import DQNAgent
from example.helper.tools import flat


# Clear experiment name so saved folders/files show what is being learned
exp_id = f"learned_GA_selector_diagpairs_seeds3-4_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
out_dir = os.path.join("./results", exp_id)

LOG_DIR   = os.path.join(out_dir, "logs")
NPZ_DIR   = os.path.join(out_dir, "datalogs")
MODEL_DIR = os.path.join(out_dir, "models")
PLOT_DIR  = os.path.join(out_dir, "plots")

GA_ASSIGNMENT_PATH = "./results/ga_20260311-093907-20 Gen/best_assignment.pkl"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(NPZ_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("Writing outputs to:", out_dir)


def moving_average(data, window_size):
    return pd.Series(data).rolling(
        window=window_size,
        center=True,
        min_periods=1
    ).mean().values


def save_run_logs(run_logs, out_dir, exp_name, seed, lam, mu):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{exp_name}_lam{lam}_mu{mu}_seed{seed}_logs.json"

    def to_jsonable(x):
        if isinstance(x, dict):
            return {k: to_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [to_jsonable(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    clean = to_jsonable(run_logs)

    with open(path, "w") as f:
        json.dump(clean, f, indent=2)

    print(f"[saved logs] {path}")


def save_run_npz(run_logs, out_dir, exp_name, seed, lam, mu):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{exp_name}_lam{lam}_mu{mu}_seed{seed}_logs.npz")

    np.savez_compressed(
        path,
        episode_returns=np.asarray(run_logs["episode_returns"], dtype=np.float32),
        episode_avg_error=np.asarray(
            [np.nan if v is None else v for v in run_logs["episode_avg_error"]],
            dtype=np.float32
        ),
        episode_success=np.asarray(run_logs["episode_success"], dtype=np.float32),
        manager_losses=np.asarray(run_logs["manager_losses"], dtype=np.float32),
        worker_losses=np.asarray(run_logs["worker_losses"], dtype=np.float32),
    )

    print(f"[saved npz] {path}")


def save_dual(fig, path_no_ext: str, dpi: int = 300):
    fig.savefig(f"{path_no_ext}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{path_no_ext}.eps", format="eps", dpi=dpi, bbox_inches="tight")


def make_setting_plot(all_run_logs, exp_name, lam, mu, plot_window, plot_dir):
    ret = np.mean(
        [np.asarray(log["episode_returns"], dtype=np.float32) for log in all_run_logs],
        axis=0
    )

    err = np.nanmean(
        [
            np.asarray(
                [np.nan if v is None else v for v in log["episode_avg_error"]],
                dtype=np.float32
            )
            for log in all_run_logs
        ],
        axis=0
    )

    ret_s = moving_average(ret, plot_window)
    err_s = moving_average(err, plot_window)

    fig, ax1 = plt.subplots(figsize=(13, 7))

    ax1.plot(ret_s, label=f"{exp_name} Return", linewidth=2)
    ax1.set_xlabel("Episode", color="black")
    ax1.set_ylabel("Return", color="black")
    ax1.tick_params(axis="x", colors="black")
    ax1.tick_params(axis="y", colors="black")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    all_ret = ret_s[np.isfinite(ret_s)]
    r_min = float(np.nanmin(all_ret)) if all_ret.size else 0.0
    r_max = float(np.nanmax(all_ret)) if all_ret.size else 1.0
    r_pad = max(1.0, 0.03 * (r_max - r_min))
    ax1.set_ylim(r_min - r_pad, r_max + r_pad)

    ax2 = ax1.twinx()
    n = len(err_s)
    step = max(1, n // 120)

    ax2.plot(
        err_s,
        linestyle=":",
        linewidth=1.2,
        marker="o",
        markersize=2.0,
        markevery=range(0, n, step),
        alpha=0.9,
        label=f"{exp_name} Error",
    )
    ax2.set_ylabel("Mean |Delivery Error|", color="black")
    ax2.tick_params(axis="y", colors="black")

    all_err = err_s[np.isfinite(err_s)]
    e_min = float(np.nanmin(all_err)) if all_err.size else 0.0
    e_max = float(np.nanmax(all_err)) if all_err.size else 1.0
    e_pad = max(0.5, 0.05 * (e_max - e_min))
    ax2.set_ylim(e_min - e_pad, e_max + e_pad)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
        frameon=True,
    )

    for t in leg.get_texts():
        t.set_color("black")

    plt.title(f"{exp_name} | λ={lam}, μ={mu}", color="black")
    plt.tight_layout()

    safe_exp = exp_name.replace(" ", "_").replace("/", "_")
    plot_path = os.path.join(
        plot_dir,
        f"{safe_exp}_lam{lam}_mu{mu}_combined_return_and_error"
    )
    save_dual(fig, plot_path)
    plt.close()

    print(f"[saved plot] {plot_path}.[png/eps]")


if __name__ == "__main__":
    is_quick_test = True

    if is_quick_test:
        print("--- RUNNING IN QUICK TEST MODE ---")
        n_episodes = 2500
        max_steps = 4000
        plot_window = 100
    else:
        n_episodes = 2000
        max_steps = 4000
        plot_window = 100

    # Requested seeds and diagonal (lambda, mu) pairs only
    seeds = [3, 4]
    settings = [
        (0.2, 20),
        (0.5, 50),
        (1.0, 80),
    ]

    exp_name = "Learned GA Selector"

    OptionClasses = [
        PickupOption,
        PickupRipeOption,
        DeliverOption,
        StoreOption,
        GAStorageSelectOption,
    ]

    exp_data = defaultdict(list)

    for lam, mu in settings:
        print("\n==============================")
        print(f"Setting: lambda={lam}, mu={mu}")
        print("==============================")

        for run_idx, seed in enumerate(seeds, start=1):
            print(f"--- Run {run_idx}/{len(seeds)} | seed={seed} | λ={lam}, μ={mu} ---")

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            env_kwargs = {
                "choose_storage": False,
                "arrival_rate": lam,
                "proc_mean": mu,
            }

            env = SmallRoomsEnv(**env_kwargs)

            for a in env.get_action_space():
                env.options.add(PrimitiveOption(a, env))

            for C in OptionClasses:
                if C.__name__ == "GAStorageSelectOption":
                    env.options.add(C(env, assignment_path=GA_ASSIGNMENT_PATH))
                else:
                    env.options.add(C(env))

            safe_exp = exp_name.replace(" ", "_").replace("/", "_")
            state_dim = len(flat(env.reset()))
            prim_size = len([o for o in env.options if o.is_primitive])

            agent = DQNAgent(
                env=env,
                state_size=state_dim,
                action_size=prim_size,
                n_episodes=n_episodes,
                n_steps=max_steps,
                batch_size=128,
                lr_manager=5e-5,
                lr_worker=3e-5,
                gamma=0.99,
                tau_soft=1e-3,
                update_every=100,
                grad_clip=5.0,
                tb_logdir=os.path.join(out_dir, "tb"),
                tb_run_name=f"{safe_exp}_lam{lam}_mu{mu}_seed{seed}",
            )

            run_logs = agent.run(
                n_episodes=n_episodes,
                max_steps=max_steps,
                npz_dir=NPZ_DIR,
                model_dir=MODEL_DIR,
            )

            save_run_logs(
                run_logs=run_logs,
                out_dir=LOG_DIR,
                exp_name=safe_exp,
                seed=seed,
                lam=lam,
                mu=mu,
            )

            save_run_npz(
                run_logs=run_logs,
                out_dir=NPZ_DIR,
                exp_name=safe_exp,
                seed=seed,
                lam=lam,
                mu=mu,
            )

            exp_data[(lam, mu)].append(run_logs)
            gc.collect()

        make_setting_plot(
            all_run_logs=exp_data[(lam, mu)],
            exp_name=exp_name,
            lam=lam,
            mu=mu,
            plot_window=plot_window,
            plot_dir=PLOT_DIR,
        )

    print("\nAll learned GA selector sensitivity runs completed.")