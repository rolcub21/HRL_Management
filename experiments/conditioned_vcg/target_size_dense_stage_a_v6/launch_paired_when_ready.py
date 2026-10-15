"""Snapshot one committed live checkpoint and launch its paired v6 episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-output", type=Path, required=True)
    parser.add_argument("--snapshot-output", type=Path, required=True)
    parser.add_argument("--benchmark-output", type=Path, required=True)
    parser.add_argument("--expected-through-episode", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--cuda-visible-devices", default="1")
    parser.add_argument("--stop-pid", type=int)
    parser.add_argument("--stop-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--main-root", type=Path, default=Path("/home/bukuj/hrl-eval"))
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/bukuj/hrl-e85-audit/.venv/bin/python"),
    )
    args = parser.parse_args()
    live = args.live_output.resolve()
    snapshot = args.snapshot_output.resolve()
    benchmark = args.benchmark_output.resolve()
    if snapshot.exists() or benchmark.exists():
        raise RuntimeError("refusing to overwrite snapshot or benchmark output")
    if args.poll_seconds <= 0.0:
        raise ValueError("poll seconds must be positive")

    summary_path = live / "training-summary.json"
    checkpoint = live / "latest.pth"
    while True:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            completed = int(summary["through_episode"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(f"waiting: summary unavailable ({error})", flush=True)
            time.sleep(args.poll_seconds)
            continue
        if completed > args.expected_through_episode:
            raise RuntimeError(
                f"live checkpoint advanced to {completed}; required exact "
                f"episode {args.expected_through_episode} was missed"
            )
        if completed < args.expected_through_episode:
            print(
                f"waiting: live checkpoint through {completed}; need "
                f"{args.expected_through_episode}",
                flush=True,
            )
            time.sleep(args.poll_seconds)
            continue
        expected_sha = str(summary["latest_checkpoint_sha256"])
        observed_sha = _sha256(checkpoint)
        if observed_sha != expected_sha:
            print("waiting: checkpoint and summary commit are not aligned", flush=True)
            time.sleep(args.poll_seconds)
            continue
        break

    if args.stop_pid is not None:
        process_path = Path(f"/proc/{args.stop_pid}")
        if not process_path.exists():
            raise RuntimeError(f"process {args.stop_pid} is not running")
        print(
            f"checkpoint committed; stopping v5 process {args.stop_pid} "
            "before v6 launch",
            flush=True,
        )
        os.kill(args.stop_pid, signal.SIGINT)
        deadline = time.monotonic() + args.stop_timeout_seconds
        while process_path.exists() and time.monotonic() < deadline:
            time.sleep(0.25)
        if process_path.exists():
            raise RuntimeError(
                f"v5 process {args.stop_pid} did not stop after SIGINT"
            )
        committed = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            int(committed["through_episode"])
            != args.expected_through_episode
            or committed["latest_checkpoint_sha256"] != expected_sha
            or _sha256(checkpoint) != expected_sha
        ):
            raise RuntimeError("v5 checkpoint changed while stopping process")
        print("v5 stopped; authenticated checkpoint is unchanged", flush=True)

    snapshot.mkdir(parents=True)
    for name in ("latest.pth", "contract.json", "training-summary.json"):
        shutil.copy2(live / name, snapshot / name)
    if _sha256(snapshot / "latest.pth") != expected_sha:
        raise RuntimeError("snapshot checkpoint hash differs after copy")
    copied_summary = json.loads(
        (snapshot / "training-summary.json").read_text(encoding="utf-8")
    )
    if (
        int(copied_summary["through_episode"]) != args.expected_through_episode
        or copied_summary["latest_checkpoint_sha256"] != expected_sha
    ):
        raise RuntimeError("copied summary does not authenticate snapshot")
    print(
        f"snapshot ready through episode {args.expected_through_episode}: "
        f"{expected_sha}",
        flush=True,
    )

    runner = Path(__file__).resolve().with_name("benchmark_one_episode.py")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    environment["PYTHONPATH"] = str(args.main_root.resolve())
    command = (
        str(args.python.absolute()),
        str(runner),
        "--source-output",
        str(snapshot),
        "--output",
        str(benchmark),
        "--expected-through-episode",
        str(args.expected_through_episode),
        "--device",
        "cuda",
    )
    subprocess.run(
        command,
        cwd=args.main_root.resolve(),
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
