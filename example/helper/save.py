import os, torch, json, time
from pathlib import Path
from example.Options.selector import StorageSelectOption

def save_checkpoint(self, ep: int, kind: str = "latest"):
    ckpt_dir = Path("/app/.../saved_models")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fname = ckpt_dir / f"{kind}_epi{ep:05d}.pth"

    # build base data
    data = {
        "episode":     ep,
        "env_steps":   self.step_count,
        "macro_state": {k: v.cpu() for k, v in self.Q_macro_local.state_dict().items()},
        "opt_state":   self.optimizer.state_dict(),
        "epsilon":     self.epsilon,
    }
    # only save scheduler if present
    if hasattr(self, "scheduler"):
        data["sched_state"] = self.scheduler.state_dict()

    # try to save selector if it exists
    selectors = [o for o in self.env.options if isinstance(o, StorageSelectOption)]
    if selectors:
        selector = selectors[0]
        data["selector_state"] = {k: v.cpu() for k, v in selector.q.state_dict().items()}

    torch.save(data, fname)


def load_checkpoint(self, path: str):
    ckpt = torch.load(path, map_location=self.device)
    self.Q_macro_local.load_state_dict(ckpt["macro_state"])
    self.Q_macro_target.load_state_dict(ckpt["macro_state"])
    self.optimizer.load_state_dict(ckpt["opt_state"])
    if "sched_state" in ckpt and hasattr(self, "scheduler"):
        self.scheduler.load_state_dict(ckpt["sched_state"])
    self.epsilon    = ckpt.get("epsilon", self.epsilon)
    self.step_count = ckpt.get("env_steps", self.step_count)

    # only load selector if both ckpt and env have it
    selectors = [o for o in self.env.options if isinstance(o, StorageSelectOption)]
    if selectors and "selector_state" in ckpt:
        selector = selectors[0]
        selector.q.load_state_dict(ckpt["selector_state"])
        selector.q.eval()

    print(f"[ckpt] restored from {path} (episode {ckpt['episode']})")
