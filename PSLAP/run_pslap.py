import numpy as np
from collections import deque
from PSLAP.PSLAPPolicy import run_pslap_episode

def run_pslap_experiment(env, n_episodes, max_steps):
    logs = {
        "episode_returns": [],
        "episode_avg_error": [],
        "episode_success": [],
        "manager_losses": [],
        "worker_losses": [],
    }

    for ep in range(1, n_episodes + 1):
        ep_ret, steps, avg_err, success = run_pslap_episode(env, max_steps=max_steps)

        logs["episode_returns"].append(float(ep_ret))
        logs["episode_avg_error"].append(float(avg_err) if np.isfinite(avg_err) else np.nan)
        logs["episode_success"].append(float(success))

        mean_ret = np.mean(deque(logs["episode_returns"], maxlen=100))
        recent_errs = np.asarray(deque(logs["episode_avg_error"], maxlen=100), dtype=float)
        mean_err = np.nan if np.all(np.isnan(recent_errs)) else np.nanmean(recent_errs)
        mean_succ = np.mean(deque(logs["episode_success"], maxlen=100))

        print(
            f"[PSLaP] Ep {ep:4d} | "
            f"AvgR {mean_ret:7.2f} | "
            f"AvgErr {mean_err:7.2f} | "
            f"Succ {mean_succ:.3f}"
        )

    return logs