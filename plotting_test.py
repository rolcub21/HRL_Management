import numpy as np
import matplotlib.pyplot as plt

filename = "combined_experiments_rewards_heuristic_test.npz"
data = np.load(filename, allow_pickle=True)

# Retrieve the stored curves
heuristic_curves = data["heuristic"]

plt.figure(figsize=(8,4))
for i, rc in enumerate(heuristic_curves):
    plt.plot(range(len(rc)), rc, label=f"Episode {i}")
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.title("Heuristic Policy - Loaded Reward Curves")
plt.legend()
plt.tight_layout()
plt.show()