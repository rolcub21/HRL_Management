import os
import re
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w=100):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    # simple centered-ish MA (causal)
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = np.nanmean(x[lo:i+1])
    return out

def extract_ep(fname):
    m = re.search(r"_partial_ep(\d+)\.npz$", fname)
    return int(m.group(1)) if m else -1

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    # keys you saved:
    # episode_returns, episode_avg_error, episode_success, manager_losses, worker_losses
    return {k: d[k] for k in d.files}

def plot_from_npzs(datalog_dir="results/datalogs", pattern="partial", ma_window=100):
    files = [f for f in os.listdir(datalog_dir) if f.endswith(".npz") and pattern in f]
    if not files:
        raise FileNotFoundError(f"No npz files matching pattern='{pattern}' in {datalog_dir}")

    files = sorted(files, key=extract_ep)
    print("Found files (sorted):")
    for f in files:
        print(" ", f)

    # load the latest (largest ep) for “most complete” curves
    latest = files[-1]
    data = load_npz(os.path.join(datalog_dir, latest))

    rets = data.get("episode_returns", np.array([]))
    errs = data.get("episode_avg_error", np.array([]))
    succ = data.get("episode_success", np.array([]))
    mgrL = data.get("manager_losses", np.array([]))
    wrkL = data.get("worker_losses", np.array([]))

    x = np.arange(1, len(rets) + 1)

    # --- Return ---
    plt.figure(figsize=(12, 5))
    plt.plot(x, rets, label="Return")
    plt.plot(x, moving_average(rets, ma_window), label=f"Return MA{ma_window}", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"Episode Return ({latest})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Avg Delivery Error ---
    plt.figure(figsize=(12, 5))
    plt.plot(x, errs, label="Avg delivery error")
    plt.plot(x, moving_average(errs, ma_window), label=f"Error MA{ma_window}", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Mean |Delivery Error|")
    plt.title(f"Average Delivery Error ({latest})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Success ---
    plt.figure(figsize=(12, 5))
    plt.plot(x, succ, label="Success (all delivered)")
    plt.plot(x, moving_average(succ, ma_window), label=f"Success MA{ma_window}", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Success")
    plt.ylim(-0.05, 1.05)
    plt.title(f"Episode Success ({latest})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Losses (note: these are logged per *update*, not per episode) ---
    if mgrL.size or wrkL.size:
        plt.figure(figsize=(12, 5))
        if mgrL.size:
            plt.plot(mgrL, label="Manager loss")
        if wrkL.size:
            plt.plot(wrkL, label="Worker loss")
        plt.xlabel("Update step (logged)")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves ({latest})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    plot_from_npzs(datalog_dir="results/datalogs", pattern="partial", ma_window=100)
