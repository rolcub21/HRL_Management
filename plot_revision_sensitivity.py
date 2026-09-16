import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================
HRL_DIR   = "results/HRL_sensitivity_20260314-182749/datalogs"
GA_DIR    = "results/GA_sensitivity_20260314-182806/datalogs"
PSLAP_DIR = "results/PSLAP_sensitivity_20260314-182828/datalogs"

OUT_DIR = "results/revision_sensitivity_plots"
os.makedirs(OUT_DIR, exist_ok=True)

SMOOTH_WINDOW = 100
FINAL_WINDOW = 100   # last 100 episodes for HRL / HRL-GA summaries

# choose 3 representative settings for learning curves
REP_SETTINGS = [(0.2, 20), (0.5, 50), (1.0, 80)]


# ============================================================
# HELPERS
# ============================================================
def moving_average(x, w=100):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    w = min(max(1, w), len(x))
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def parse_setting(path):
    """
    Example:
      HRL-Learning_selector_lam0.2_mu20_seed0_logs.npz
      HRL-GA_selector_lam0.2_mu20_seed1_logs.npz
      PSLaP_lam0.2_mu20_seed2_logs.npz
    """
    name = os.path.basename(path)
    m = re.search(r"lam([0-9.]+)_mu([0-9.]+)_seed(\d+)_logs\.npz$", name)
    if not m:
        return None
    lam = float(m.group(1))
    mu = float(m.group(2))
    seed = int(m.group(3))
    return lam, mu, seed


def finite_mean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.mean(x))


def finite_std(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.std(x, ddof=1)) if x.size > 1 else 0.0


def load_group(folder):
    """
    Returns:
      data[(lam, mu)] = {
          "returns": [arr_seed0, arr_seed1, ...],
          "errors":  [...],
          "success": [...]
      }
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(folder, "*_logs.npz"))):
        parsed = parse_setting(f)
        if parsed is None:
            continue
        lam, mu, seed = parsed

        z = np.load(f, allow_pickle=True)
        returns = z["episode_returns"].astype(np.float64)
        errors  = z["episode_avg_error"].astype(np.float64)
        success = z["episode_success"].astype(np.float64)

        key = (lam, mu)
        if key not in out:
            out[key] = {"returns": [], "errors": [], "success": []}

        out[key]["returns"].append(returns)
        out[key]["errors"].append(errors)
        out[key]["success"].append(success)

    return out


def mean_curve(curves):
    if not curves:
        return None
    L = min(len(c) for c in curves)
    arr = np.stack([c[:L] for c in curves], axis=0)
    return np.mean(arr, axis=0)


def std_curve(curves):
    if not curves:
        return None
    L = min(len(c) for c in curves)
    arr = np.stack([c[:L] for c in curves], axis=0)
    return np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(L)


def final_window_mean(arr, w=100):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    w = min(w, arr.size)
    return float(np.mean(arr[-w:]))


def all_episode_mean(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def summarize_method(data, method_name):
    rows = []
    for (lam, mu), vals in sorted(data.items()):
        if method_name in ("HRL-learning", "HRL-GA"):
            ret_per_seed = [final_window_mean(x, FINAL_WINDOW) for x in vals["returns"]]
            err_per_seed = [final_window_mean(x, FINAL_WINDOW) for x in vals["errors"]]
            suc_per_seed = [final_window_mean(x, FINAL_WINDOW) for x in vals["success"]]
        else:
            # PSLAP is treated as repeated evaluation, not training
            ret_per_seed = [all_episode_mean(x) for x in vals["returns"]]
            err_per_seed = [all_episode_mean(x) for x in vals["errors"]]
            suc_per_seed = [all_episode_mean(x) for x in vals["success"]]

        rows.append({
            "method": method_name,
            "lam": lam,
            "mu": mu,
            "return_mean": finite_mean(ret_per_seed),
            "return_std": finite_std(ret_per_seed),
            "error_mean": finite_mean(err_per_seed),
            "error_std": finite_std(err_per_seed),
            "success_mean": finite_mean(suc_per_seed),
            "success_std": finite_std(suc_per_seed),
            "n_seeds": len(ret_per_seed),
        })
    return rows


# ============================================================
# LOAD DATA
# ============================================================
hrl_data   = load_group(HRL_DIR)
ga_data    = load_group(GA_DIR)
pslap_data = load_group(PSLAP_DIR)

all_settings = sorted(set(hrl_data.keys()) | set(ga_data.keys()) | set(pslap_data.keys()))

print("Found settings:")
for s in all_settings:
    print(" ", s)

summary_rows = []
summary_rows += summarize_method(hrl_data, "HRL-learning")
summary_rows += summarize_method(ga_data, "HRL-GA")
summary_rows += summarize_method(pslap_data, "PSLAP")

summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUT_DIR, "sensitivity_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print("Saved:", summary_csv)


# ============================================================
# FIGURE 1: SUMMARY RETURN ACROSS SETTINGS
# ============================================================
settings_sorted = sorted(all_settings, key=lambda x: (x[0], x[1]))
xlabels = [rf"$\lambda$={lam}, $\mu$={mu}" for lam, mu in settings_sorted]
x = np.arange(len(settings_sorted))

def get_series(df, method, metric):
    vals = []
    errs = []
    for lam, mu in settings_sorted:
        sub = df[(df["method"] == method) & (df["lam"] == lam) & (df["mu"] == mu)]
        if len(sub) == 0:
            vals.append(np.nan)
            errs.append(np.nan)
        else:
            vals.append(float(sub.iloc[0][f"{metric}_mean"]))
            errs.append(float(sub.iloc[0][f"{metric}_std"]))
    return np.array(vals, dtype=float), np.array(errs, dtype=float)

fig, ax = plt.subplots(figsize=(13, 6))

for method, marker in [("HRL-learning", "o"), ("HRL-GA", "s"), ("PSLAP", "^")]:
    y, yerr = get_series(summary_df, method, "return")
    ax.errorbar(x, y, yerr=yerr, marker=marker, linewidth=2, capsize=4, label=method)

ax.set_xticks(x)
ax.set_xticklabels(xlabels, rotation=35, ha="right")
ax.set_ylabel("Mean return")
ax.set_title("Sensitivity analysis across stochastic arrival and storage-time settings")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
plt.tight_layout()
f1 = os.path.join(OUT_DIR, "figure_return_summary.png")
plt.savefig(f1, dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved:", f1)


# ============================================================
# FIGURE 2: SUMMARY ERROR ACROSS SETTINGS
# ============================================================
fig, ax = plt.subplots(figsize=(13, 6))

for method, marker in [("HRL-learning", "o"), ("HRL-GA", "s"), ("PSLAP", "^")]:
    y, yerr = get_series(summary_df, method, "error")
    ax.errorbar(x, y, yerr=yerr, marker=marker, linewidth=2, capsize=4, label=method)

ax.set_xticks(x)
ax.set_xticklabels(xlabels, rotation=35, ha="right")
ax.set_ylabel("Mean delivery error")
ax.set_title("Sensitivity analysis: delivery error across settings")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
plt.tight_layout()
f2 = os.path.join(OUT_DIR, "figure_error_summary.png")
plt.savefig(f2, dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved:", f2)


# ============================================================
# FIGURE 3: OPTIONAL SUCCESS SUMMARY
# ============================================================
fig, ax = plt.subplots(figsize=(13, 6))

for method, marker in [("HRL-learning", "o"), ("HRL-GA", "s"), ("PSLAP", "^")]:
    y, yerr = get_series(summary_df, method, "success")
    ax.errorbar(x, y, yerr=yerr, marker=marker, linewidth=2, capsize=4, label=method)

ax.set_xticks(x)
ax.set_xticklabels(xlabels, rotation=35, ha="right")
ax.set_ylabel("Mean success rate")
ax.set_title("Sensitivity analysis: success rate across settings")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
plt.tight_layout()
f3 = os.path.join(OUT_DIR, "figure_success_summary.png")
plt.savefig(f3, dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved:", f3)


# ============================================================
# FIGURE 4: REPRESENTATIVE LEARNING CURVES
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

for ax, (lam, mu) in zip(axes, REP_SETTINGS):
    # HRL-learning
    if (lam, mu) in hrl_data:
        mean_ret = mean_curve(hrl_data[(lam, mu)]["returns"])
        std_ret  = std_curve(hrl_data[(lam, mu)]["returns"])
        if mean_ret is not None:
            m = moving_average(mean_ret, SMOOTH_WINDOW)
            s = moving_average(std_ret, SMOOTH_WINDOW)
            xx = np.arange(len(m))
            ax.plot(xx, m, linewidth=2, label="HRL-learning")
            ax.fill_between(xx, m - s, m + s, alpha=0.15)

    # HRL-GA
    if (lam, mu) in ga_data:
        mean_ret = mean_curve(ga_data[(lam, mu)]["returns"])
        std_ret  = std_curve(ga_data[(lam, mu)]["returns"])
        if mean_ret is not None:
            m = moving_average(mean_ret, SMOOTH_WINDOW)
            s = moving_average(std_ret, SMOOTH_WINDOW)
            xx = np.arange(len(m))
            ax.plot(xx, m, linewidth=2, label="HRL-GA")
            ax.fill_between(xx, m - s, m + s, alpha=0.15)

    # PSLAP mean baseline
    if (lam, mu) in pslap_data:
        pslap_vals = []
        for arr in pslap_data[(lam, mu)]["returns"]:
            pslap_vals.extend(arr.tolist())
        p_mean = finite_mean(pslap_vals)
        p_std  = finite_std(pslap_vals)
        if np.isfinite(p_mean):
            ax.axhline(p_mean, linestyle="--", linewidth=2, label="PSLAP mean")
            if np.isfinite(p_std):
                ax.axhspan(p_mean - p_std, p_mean + p_std, alpha=0.10)

    ax.set_title(rf"$\lambda$={lam}, $\mu$={mu}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.grid(True, linestyle="--", alpha=0.5)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)
fig.suptitle("Representative learning curves under easy, medium, and hard settings", y=1.03)
plt.tight_layout()
f4 = os.path.join(OUT_DIR, "figure_representative_learning_curves.png")
plt.savefig(f4, dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved:", f4)


# ============================================================
# FIGURE 5: HEATMAP STYLE TABLE PLOT FOR RETURN
# ============================================================
# rows = lambda, cols = mu, values = method-specific means
lam_vals = sorted(set(l for l, _ in settings_sorted))
mu_vals  = sorted(set(m for _, m in settings_sorted))

method_order = ["HRL-learning", "HRL-GA", "PSLAP"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, method in zip(axes, method_order):
    M = np.full((len(lam_vals), len(mu_vals)), np.nan)
    for i, lam in enumerate(lam_vals):
        for j, mu in enumerate(mu_vals):
            sub = summary_df[
                (summary_df["method"] == method) &
                (summary_df["lam"] == lam) &
                (summary_df["mu"] == mu)
            ]
            if len(sub):
                M[i, j] = float(sub.iloc[0]["return_mean"])

    im = ax.imshow(M, aspect="auto")
    ax.set_xticks(np.arange(len(mu_vals)))
    ax.set_xticklabels(mu_vals)
    ax.set_yticks(np.arange(len(lam_vals)))
    ax.set_yticklabels(lam_vals)
    ax.set_xlabel(r"$\mu$")
    ax.set_title(method)

    for i in range(len(lam_vals)):
        for j in range(len(mu_vals)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", fontsize=9)

axes[0].set_ylabel(r"$\lambda$")
fig.suptitle("Return heatmaps across stochastic settings", y=1.02)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
cbar.set_label("Mean return")
plt.tight_layout()
f5 = os.path.join(OUT_DIR, "figure_return_heatmaps.png")
plt.savefig(f5, dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved:", f5)

print("\nAll revision plots saved in:", OUT_DIR)