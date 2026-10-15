"""Create one compact console line when an in-flight isolated episode ends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def _line(report: dict) -> str:
    result = report["result"]
    episode = result["episode"]
    return (
        f"Ep {int(report['episode_number']):4d}"
        f" | DenseR {float(episode['dense_rescored_return']):9.2f}"
        f" | Done {int(episode['delivery_count']):2d}/64"
        f" | Dec {int(episode['macro_decisions']):4d}"
        f" | MAE {float(episode['mean_absolute_error']):.3f}"
        f" | Time {float(result['wall_seconds']) / 60.0:.1f}m"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    output = args.output.resolve()
    compact = output / "episode-console.log"
    report_path = output / "report.json"
    if compact.exists():
        raise RuntimeError(f"refusing to overwrite: {compact}")
    compact.touch(exist_ok=False)
    while True:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            break
        except (OSError, json.JSONDecodeError):
            time.sleep(args.poll_seconds)
    line = _line(report)
    with compact.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


if __name__ == "__main__":
    main()
