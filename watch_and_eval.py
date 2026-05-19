"""
watch_and_eval.py — Auto-eval watcher for GRPO training

Watches the checkpoint directory for new model saves and runs eval_grpo.py
every N batches. Results are appended to a CSV for easy trend tracking.

Usage:
    python watch_and_eval.py --exp-dir experiments_grpo/grpo_kinematic --every 100
"""

import os
import re
import time
import csv
import argparse
import subprocess
import sys
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--exp-dir', type=str, default='experiments_grpo/grpo_kinematic')
parser.add_argument('--every', type=int, default=100, help='eval every N batches')
parser.add_argument('--split', type=str, default='val')
parser.add_argument('--interval', type=float, default=60.0, help='poll interval seconds')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

LOG_PATH = os.path.join(args.exp_dir, 'run_grpo.log')
MODEL_PATH = os.path.join(args.exp_dir, 'grpo_model.pth')
RESULTS_CSV = os.path.join(args.exp_dir, 'eval_results.csv')

PYTHON = sys.executable

BATCH_RE = re.compile(r'Epoch \[(\d+)\] Batch \[(\d+)/(\d+)\]')

def get_latest_batch(log_path):
    """Return (epoch, batch) of the last logged line."""
    last_epoch, last_batch = 0, 0
    try:
        with open(log_path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = BATCH_RE.search(line)
                if m:
                    last_epoch = int(m.group(1))
                    last_batch = int(m.group(2))
    except FileNotFoundError:
        pass
    return last_epoch, last_batch

def run_eval(batch_idx):
    """Run eval_grpo.py and return metrics dict or None on failure."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running eval at batch {batch_idx}...")
    cmd = [
        PYTHON, 'eval_grpo.py',
        '--checkpoint', MODEL_PATH,
        '--split', args.split,
        '--device', args.device,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr
        print(output[-2000:])  # print last 2000 chars

        metrics = {}
        for line in output.splitlines():
            for key in ['FID', 'Div', 'Top1', 'Top2', 'Top3', 'Matching', 'Multi']:
                if line.strip().startswith(key + ':'):
                    try:
                        metrics[key] = float(line.split(':')[1].strip())
                    except ValueError:
                        pass
        return metrics if metrics else None
    except subprocess.TimeoutExpired:
        print("  [WARN] Eval timed out after 10 minutes")
        return None
    except Exception as e:
        print(f"  [ERROR] Eval failed: {e}")
        return None

def append_csv(batch_idx, metrics):
    """Append one row to the results CSV."""
    fieldnames = ['timestamp', 'batch', 'FID', 'Div', 'Top1', 'Top2', 'Top3', 'Matching', 'Multi']
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'batch': batch_idx}
        row.update(metrics)
        writer.writerow(row)
    print(f"  Saved to {RESULTS_CSV}")

def print_csv_summary():
    """Print all eval results so far."""
    if not os.path.exists(RESULTS_CSV):
        return
    print("\n  ===== Eval History =====")
    with open(RESULTS_CSV, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return
    header = f"  {'Batch':>6}  {'FID':>7}  {'Top1':>6}  {'Top3':>6}  {'Match':>7}  {'Div':>6}"
    print(header)
    print("  " + "-" * 50)
    for row in rows:
        print(f"  {row['batch']:>6}  {row.get('FID','?'):>7}  {row.get('Top1','?'):>6}  "
              f"{row.get('Top3','?'):>6}  {row.get('Matching','?'):>7}  {row.get('Div','?'):>6}")

def main():
    print(f"Watching: {LOG_PATH}")
    print(f"Eval every {args.every} batches | Poll interval: {args.interval}s")
    print(f"Results: {RESULTS_CSV}")
    print("Press Ctrl+C to stop.\n")

    last_eval_batch = -1

    while True:
        epoch, batch = get_latest_batch(LOG_PATH)

        # Determine next eval milestone
        if batch == 0:
            print(f"  Waiting for training to start...")
        else:
            next_milestone = ((last_eval_batch // args.every) + 1) * args.every
            if batch >= next_milestone:
                # Check model file exists and is recent
                if os.path.exists(MODEL_PATH):
                    metrics = run_eval(batch)
                    if metrics:
                        append_csv(batch, metrics)
                        print_csv_summary()
                    last_eval_batch = batch
                else:
                    print(f"  [WARN] Model file not found: {MODEL_PATH}")
            else:
                remaining = next_milestone - batch
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"Epoch {epoch} Batch {batch} — next eval at batch {next_milestone} "
                      f"({remaining} batches away)")

        time.sleep(args.interval)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        print_csv_summary()
