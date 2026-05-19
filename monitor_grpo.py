"""
GRPO Training Monitor
Usage: python monitor_grpo.py [log_path] [--interval 30]

Tails the log file and prints a live summary table.
Default log: experiments_grpo/reward_test/run_grpo.log
"""

import re
import time
import sys
import os
import argparse
from collections import deque

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('log_path', nargs='?', default='experiments_grpo/reward_test/run_grpo.log')
parser.add_argument('--interval', type=float, default=10.0, help='refresh interval in seconds')
parser.add_argument('--window', type=int, default=20, help='rolling average window size')
args = parser.parse_args()

LOG_RE = re.compile(
    r'Epoch \[(\d+)\] Batch \[(\d+)/(\d+)\] '
    r'Loss: ([\d\.\-]+), Reward: ([\d\.\-]+), '
    r'LogProb: ([\d\.\-]+), KL: ([\d\.\-]+), SFT_KL: ([\d\.\-]+), '
    r'Ratio: ([\d\.\-]+), ClipFrac: ([\d\.\-]+), InnerK: (\d+), '
    r'LR: ([\d\.e\-]+), PosSim: ([\d\.\-]+), NegSim: ([\d\.\-]+), '
    r'Phys: ([\d\.\-]+), Num: ([\d\.\-]+), Kin: ([\d\.\-]+)'
)

def parse_line(line):
    m = LOG_RE.search(line)
    if not m:
        return None
    return {
        'epoch': int(m.group(1)),
        'batch': int(m.group(2)),
        'total_batches': int(m.group(3)),
        'loss': float(m.group(4)),
        'reward': float(m.group(5)),
        'logprob': float(m.group(6)),
        'kl': float(m.group(7)),
        'sft_kl': float(m.group(8)),
        'ratio': float(m.group(9)),
        'clip_frac': float(m.group(10)),
        'inner_k': int(m.group(11)),
        'lr': float(m.group(12)),
        'pos_sim': float(m.group(13)),
        'neg_sim': float(m.group(14)),
        'phys': float(m.group(15)),
        'num': float(m.group(16)),
        'kin': float(m.group(17)),
    }

def avg(q):
    return sum(q) / len(q) if q else 0.0

def bar(value, min_val, max_val, width=20):
    if max_val == min_val:
        frac = 0.5
    else:
        frac = max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
    filled = int(frac * width)
    return '[' + '█' * filled + '░' * (width - filled) + ']'

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def read_all_entries(path):
    entries = []
    try:
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                e = parse_line(line)
                if e:
                    entries.append(e)
    except FileNotFoundError:
        pass
    return entries

def monitor():
    window = args.window
    print(f"Monitoring: {args.log_path}")
    print(f"Refresh: {args.interval}s | Rolling window: {window} batches")
    print("Press Ctrl+C to stop.\n")

    reward_hist = deque(maxlen=window)
    num_hist = deque(maxlen=window)
    phys_hist = deque(maxlen=window)
    kin_hist = deque(maxlen=window)
    pos_sim_hist = deque(maxlen=window)
    sft_kl_hist = deque(maxlen=window)
    loss_hist = deque(maxlen=window)

    last_batch = -1

    while True:
        entries = read_all_entries(args.log_path)

        # Only process new entries
        new_entries = [e for e in entries if e['batch'] > last_batch]
        for e in new_entries:
            reward_hist.append(e['reward'])
            num_hist.append(e['num'])
            phys_hist.append(e['phys'])
            kin_hist.append(e.get('kin', 0.0))
            pos_sim_hist.append(e['pos_sim'])
            sft_kl_hist.append(e['sft_kl'])
            loss_hist.append(e['loss'])
            last_batch = max(last_batch, e['batch'])

        clear()
        print("=" * 60)
        print("  GRPO Training Monitor")
        print("=" * 60)

        if not entries:
            print(f"\n  Waiting for log: {args.log_path}")
        else:
            latest = entries[-1]
            progress = latest['batch'] / latest['total_batches']
            pct = progress * 100

            print(f"\n  Epoch {latest['epoch']}  |  Batch {latest['batch']}/{latest['total_batches']}  ({pct:.1f}%)")
            prog_bar = '[' + '█' * int(progress * 40) + '░' * (40 - int(progress * 40)) + ']'
            print(f"  {prog_bar}")
            print(f"  LR: {latest['lr']:.2e}  |  InnerK: {latest['inner_k']}")

            print(f"\n  {'Metric':<14} {'Latest':>8}  {'Avg({:d})'.format(window):>10}  {'Trend':>22}")
            print("  " + "-" * 56)

            def row(name, latest_val, hist, lo, hi):
                a = avg(hist)
                b = bar(a, lo, hi)
                print(f"  {name:<14} {latest_val:>8.4f}  {a:>10.4f}  {b}")

            row("Reward",     latest['reward'],          reward_hist,  0.5, 1.5)
            row("Num score",  latest['num'],              num_hist,     0.0, 1.0)
            row("Kin score",  latest.get('kin', 0.0),    kin_hist,     0.0, 1.0)
            row("Phys score", latest['phys'],             phys_hist,    0.0, 1.0)
            row("PosSim",     latest['pos_sim'],          pos_sim_hist, 0.5, 1.0)
            row("SFT_KL",     latest['sft_kl'],           sft_kl_hist,  0.0, 5.0)
            row("Loss",       latest['loss'],             loss_hist,   -0.1, 0.1)

            # Reward variance in window (key discriminability metric)
            if len(reward_hist) >= 2:
                import statistics
                r_std = statistics.stdev(reward_hist)
                n_std = statistics.stdev(num_hist) if len(num_hist) >= 2 else 0.0
                print(f"\n  Reward std (window): {r_std:.4f}  |  Num std: {n_std:.4f}")
                if r_std < 0.01:
                    print("  ⚠  Low reward variance — poor discrimination")
                elif r_std > 0.05:
                    print("  ✓  Good reward variance")

            # Num score fraction
            num_nonzero = sum(1 for v in num_hist if v > 0.01)
            print(f"  Num score > 0: {num_nonzero}/{len(num_hist)} batches in window")

        print(f"\n  Log: {args.log_path}")
        print(f"  Last update: {time.strftime('%H:%M:%S')}")
        print("=" * 60)

        time.sleep(args.interval)

if __name__ == '__main__':
    try:
        monitor()
    except KeyboardInterrupt:
        print("\nStopped.")
