"""Monitor GRPO training logs and GPU/process status.

Examples:
  python monitor_grpo_training.py --exp executor_grpo_bs16_2026_05_08 --watch 30
  python monitor_grpo_training.py --log experiments_grpo/my_exp.out
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


BATCH_RE = re.compile(
    r"Epoch \[(?P<epoch>\d+)\] Batch \[(?P<batch>\d+)/(?P<total>\d+)\] "
    r"Loss: (?P<loss>-?\d+(?:\.\d+)?), Reward: (?P<reward>-?\d+(?:\.\d+)?), "
    r"LogProb: (?P<logprob>-?\d+(?:\.\d+)?), "
    r"KL: (?P<kl>-?\d+(?:\.\d+)?), SFT_KL: (?P<sft_kl>-?\d+(?:\.\d+)?), "
    r"Ratio: (?P<ratio>-?\d+(?:\.\d+)?), "
    r"ClipFrac: (?P<clip>-?\d+(?:\.\d+)?), "
    r"InnerK: (?P<inner_k>\d+), "
    r"LR: (?P<lr>-?\d+(?:\.\d+)?), "
    r"PosSim: (?P<pos>-?\d+(?:\.\d+)?), NegSim: (?P<neg>-?\d+(?:\.\d+)?), "
    r"Phys: (?P<phys>-?\d+(?:\.\d+)?), Num: (?P<num>-?\d+(?:\.\d+)?), "
    r"Kin: (?P<kin>-?\d+(?:\.\d+)?), Exec: (?P<exec>-?\d+(?:\.\d+)?)"
)

SUMMARY_RE = re.compile(r"Epoch (?P<epoch>\d+) Summary: (?P<body>.*)")


def _read_tail(path: Path, max_bytes: int) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - max_bytes), os.SEEK_SET)
        return fh.read().decode("utf-8", "replace")


def _parse_batches(text: str) -> List[Dict[str, float]]:
    batches: List[Dict[str, float]] = []
    for match in BATCH_RE.finditer(text):
        item: Dict[str, float] = {}
        for key, value in match.groupdict().items():
            item[key] = float(value)
        batches.append(item)
    return batches


def _parse_summaries(text: str) -> List[str]:
    return [m.group(0) for m in SUMMARY_RE.finditer(text)]


def _run(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5)
        return out.strip()
    except Exception:
        return None


def _gpu_status() -> str:
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader",
    ])
    return out or "GPU status unavailable"


def _process_status(pid_file: Path) -> str:
    if not pid_file.exists():
        return f"PID file not found: {pid_file}"
    pid = pid_file.read_text(encoding="utf-8", errors="replace").strip()
    if not pid:
        return f"PID file is empty: {pid_file}"
    if os.name == "nt":
        out = _run(["powershell", "-NoProfile", "-Command", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object Id,CPU,StartTime,ProcessName | Format-List"])
    else:
        out = _run(["ps", "-p", pid, "-o", "pid,etime,%cpu,%mem,cmd"])
    return out or f"Process {pid} is not running"


def _fmt_delta(first: Dict[str, float], last: Dict[str, float], key: str) -> str:
    return f"{last[key]:.4f} ({last[key] - first[key]:+.4f})"


def render_once(log_path: Path, pid_file: Path, tail_bytes: int, recent: int) -> str:
    text = _read_tail(log_path, tail_bytes)
    batches = _parse_batches(text)
    summaries = _parse_summaries(text)
    lines: List[str] = []

    lines.append("=" * 88)
    lines.append(f"Log: {log_path}")
    lines.append(f"PID: {pid_file}")
    lines.append("")
    lines.append("[Process]")
    lines.append(_process_status(pid_file))
    lines.append("")
    lines.append("[GPU]")
    lines.append(_gpu_status())
    lines.append("")

    if not batches:
        lines.append("[Training]")
        lines.append("No batch metrics found yet.")
        if text:
            lines.append("")
            lines.append("[Log Tail]")
            lines.extend(text.splitlines()[-20:])
        return "\n".join(lines)

    last = batches[-1]
    first_window = batches[0]
    epoch = int(last["epoch"])
    batch = int(last["batch"])
    total = int(last["total"])
    pct = 100.0 * batch / max(total, 1)

    lines.append("[Latest]")
    lines.append(
        f"epoch={epoch} batch={batch}/{total} ({pct:.2f}%) "
        f"loss={last['loss']:.4f} reward={last['reward']:.4f} "
        f"sft_kl={last['sft_kl']:.4f} ratio={last['ratio']:.3f} "
        f"clip={last['clip']:.3f} exec={last['exec']:.4f}"
    )

    window = batches[-min(recent, len(batches)):]
    avg = {
        key: sum(item[key] for item in window) / len(window)
        for key in ["loss", "reward", "sft_kl", "ratio", "clip", "phys", "num", "kin", "exec"]
    }
    lines.append("")
    lines.append(f"[Averages: last {len(window)} logged batches]")
    lines.append(
        " ".join([
            f"loss={avg['loss']:.4f}",
            f"reward={avg['reward']:.4f}",
            f"sft_kl={avg['sft_kl']:.4f}",
            f"ratio={avg['ratio']:.3f}",
            f"clip={avg['clip']:.3f}",
            f"phys={avg['phys']:.4f}",
            f"num={avg['num']:.4f}",
            f"kin={avg['kin']:.4f}",
            f"exec={avg['exec']:.4f}",
        ])
    )

    lines.append("")
    lines.append("[Change In Loaded Tail]")
    lines.append(
        " ".join([
            f"reward={_fmt_delta(first_window, last, 'reward')}",
            f"sft_kl={_fmt_delta(first_window, last, 'sft_kl')}",
            f"ratio={_fmt_delta(first_window, last, 'ratio')}",
            f"exec={_fmt_delta(first_window, last, 'exec')}",
        ])
    )

    if summaries:
        lines.append("")
        lines.append("[Summaries]")
        lines.extend(summaries[-3:])

    lines.append("")
    lines.append("[Recent Batches]")
    for item in batches[-min(recent, len(batches)):]:
        lines.append(
            f"E{int(item['epoch'])} B{int(item['batch'])}/{int(item['total'])} "
            f"loss={item['loss']:.4f} reward={item['reward']:.4f} "
            f"kl={item['sft_kl']:.4f} ratio={item['ratio']:.3f} "
            f"clip={item['clip']:.3f} exec={item['exec']:.4f}"
        )

    return "\n".join(lines)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.log:
        log_path = Path(args.log)
        if args.pid_file:
            pid_file = Path(args.pid_file)
        else:
            pid_file = log_path.with_suffix(".pid")
        return log_path, pid_file

    exp = args.exp
    root = Path(args.root)
    log_path = root / f"{exp}.out"
    pid_file = root / f"{exp}.pid"
    return log_path, pid_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor GRPO training progress.")
    parser.add_argument("--exp", default="executor_grpo_bs16_2026_05_08", help="Experiment name under experiments_grpo")
    parser.add_argument("--root", default="experiments_grpo", help="Experiment root directory")
    parser.add_argument("--log", default="", help="Explicit log path")
    parser.add_argument("--pid-file", default="", help="Explicit PID file path")
    parser.add_argument("--watch", type=float, default=0.0, help="Refresh interval in seconds; 0 prints once")
    parser.add_argument("--tail-bytes", type=int, default=2_000_000, help="How many bytes of log tail to parse")
    parser.add_argument("--recent", type=int, default=8, help="Number of recent logged batches to show")
    args = parser.parse_args()

    log_path, pid_file = resolve_paths(args)
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print(render_once(log_path, pid_file, args.tail_bytes, args.recent))
        if args.watch <= 0:
            break
        time.sleep(args.watch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
