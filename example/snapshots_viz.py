import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import glob

# ----------------------------------------------------------------------------
#  CONFIGURATION
# ----------------------------------------------------------------------------

opt_names = [
    'PrimitiveOption-UP','PrimitiveOption-DOWN','PrimitiveOption-LEFT','PrimitiveOption-RIGHT',
    'PickupOption','StoreOption','DeliverOption','StorageSelectOption'
]

base_colors = {
    'PrimitiveOption-UP'   : '#bbbbbb',
    'PrimitiveOption-DOWN' : '#bbbbbb',
    'PrimitiveOption-LEFT' : '#bbbbbb',
    'PrimitiveOption-RIGHT': '#bbbbbb',
    'PickupOption'         : '#377eb8',
    'StoreOption'          : '#4daf4a',
    'DeliverOption'        : '#984ea3',
    'StorageSelectOption'  : '#e41a1c',
}

SNAPSHOT_DIR = "example/results/snapshots"
OUT_DIR      = "example/results/snapshots"
os.makedirs(OUT_DIR, exist_ok=True)

# which cells to average for the confidence curve
CRITICAL_CELLS = [(0,5), (6,4), (6,5), (6,6)]

# ----------------------------------------------------------------------------
#  UTILITIES
# ----------------------------------------------------------------------------

def parse_episode(fname):
    """
    Given a filename like '.../qmap_ep0123.npy', return integer 123.
    """
    base = os.path.basename(fname)
    num = base.replace("qmap_ep", "").replace(".npy", "")
    return int(num)

def sorted_snapshots():
    """
    Return a list of (episode, full_path) sorted by episode.
    """
    pattern = os.path.join(SNAPSHOT_DIR, "qmap_ep*.npy")
    files   = glob.glob(pattern)
    pairs   = [(parse_episode(f), f) for f in files]
    return sorted(pairs, key=lambda x: x[0])

def show_qmap(np_file):
    """
    Load one .npy, build the RGB heat‐map, and save it as PNG.
    """
    cube     = np.load(np_file)          # rows×cols×num_options
    best_idx = cube.argmax(axis=2)       # rows×cols
    best_q   = cube.max(axis=2)

    # mean over primitives
    prim_inds = [i for i,name in enumerate(opt_names) if name.startswith("PrimitiveOption")]
    prim_mean = cube[..., prim_inds].mean(axis=2)

    boost = np.clip(best_q - prim_mean, 0, None)
    boost = boost / (boost.max() + 1e-6)

    rows, cols = best_idx.shape
    rgb = np.ones((rows, cols, 3), dtype=float)

    for i, name in enumerate(opt_names):
        mask = (best_idx == i)
        rgb[mask] = mpl.colors.to_rgb(base_colors[name])

    rgb = rgb * boost[...,None] + (1-boost)[...,None]

    plt.figure(figsize=(4,4))
    plt.imshow(rgb, origin='upper')
    plt.axis('off')
    stem = os.path.splitext(os.path.basename(np_file))[0]
    plt.title(stem)
    out_png = os.path.join(OUT_DIR, f"{stem}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def load_confidence(option_name, snapshots):
    """
    For each (ep,path) in snapshots, load the cube and average Q over CRITICAL_CELLS.
    """
    idx = opt_names.index(option_name)
    vals = []
    for ep, path in snapshots:
        cube = np.load(path)
        rs, cs = zip(*CRITICAL_CELLS)
        vals.append(cube[rs, cs, idx].mean())
    return vals

# ----------------------------------------------------------------------------
#  MAIN
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    snaps = sorted_snapshots()
    if not snaps:
        raise RuntimeError(f"No qmap_ep*.npy found in {SNAPSHOT_DIR}")

    # 1) render each snapshot
    for ep, path in snaps:
        show_qmap(path)
    print(f"Wrote {len(snaps)} PNGs to {OUT_DIR}")

    # 2) plot confidence‐curve
    plt.figure(figsize=(6,4))
    episodes = [ep for ep,_ in snaps]

    for name in ['StorageSelectOption','DeliverOption','PickupOption']:
        y = load_confidence(name, snaps)
        plt.plot(episodes, y, marker='o', label=name)

    plt.xlabel("Episode")
    plt.ylabel("mean Q on critical cells")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    curve_out = os.path.join(OUT_DIR, "confidence_curve.png")
    plt.savefig(curve_out, dpi=300)
    plt.close()
    print("Wrote confidence curve to", curve_out)
