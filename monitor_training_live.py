#!/usr/bin/env python3
"""
Real-time GRPO training monitor with live updates
"""
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os
import argparse

def parse_log(log_file):
    """Parse training log and extract metrics"""
    metrics = defaultdict(list)

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Match batch log lines
                match = re.search(
                    r'Epoch \[(\d+)\] Batch \[(\d+)/(\d+)\] '
                    r'Loss: ([-\d.]+), Reward: ([-\d.]+), LogProb: ([-\d.]+), '
                    r'KL: ([-\d.]+), SFT_KL: ([-\d.]+), Ratio: ([-\d.]+), '
                    r'ClipFrac: ([-\d.]+), InnerK: (\d+), LR: ([-\d.e]+), '
                    r'PosSim: ([-\d.]+), NegSim: ([-\d.]+), Phys: ([-\d.]+), '
                    r'Num: ([-\d.]+), Kin: ([-\d.]+)',
                    line
                )

                if match:
                    epoch = int(match.group(1))
                    batch = int(match.group(2))
                    total_batches = int(match.group(3))
                    step = epoch * total_batches + batch

                    metrics['step'].append(step)
                    metrics['epoch'].append(epoch)
                    metrics['batch'].append(batch)
                    metrics['total_batches'] = total_batches
                    metrics['loss'].append(float(match.group(4)))
                    metrics['reward'].append(float(match.group(5)))
                    metrics['logprob'].append(float(match.group(6)))
                    metrics['kl'].append(float(match.group(7)))
                    metrics['sft_kl'].append(float(match.group(8)))
                    metrics['ratio'].append(float(match.group(9)))
                    metrics['clip_frac'].append(float(match.group(10)))
                    metrics['inner_k'].append(int(match.group(11)))
                    metrics['lr'].append(float(match.group(12)))
                    metrics['pos_sim'].append(float(match.group(13)))
                    metrics['neg_sim'].append(float(match.group(14)))
                    metrics['phys'].append(float(match.group(15)))
                    metrics['num'].append(float(match.group(16)))
                    metrics['kin'].append(float(match.group(17)))
    except FileNotFoundError:
        pass

    return metrics

def plot_metrics(metrics, output_file='training_metrics.png'):
    """Plot training metrics"""
    if not metrics['step']:
        return False

    plt.clf()
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    # Get current progress
    current_epoch = metrics['epoch'][-1]
    current_batch = metrics['batch'][-1]
    total_batches = metrics.get('total_batches', 291)
    progress = (current_batch / total_batches) * 100

    fig.suptitle(f'GRPO Training - Epoch {current_epoch+1}, Batch {current_batch}/{total_batches} ({progress:.1f}%)',
                 fontsize=16, fontweight='bold')

    steps = metrics['step']

    # Row 1: Main metrics
    axes[0, 0].plot(steps, metrics['reward'], 'b-', linewidth=2)
    axes[0, 0].set_title(f'Reward (Current: {metrics["reward"][-1]:.4f})', fontweight='bold')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, metrics['loss'], 'r-', linewidth=2)
    axes[0, 1].set_title(f'Loss (Current: {metrics["loss"][-1]:.4f})', fontweight='bold')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(steps, metrics['logprob'], 'g-', linewidth=2)
    axes[0, 2].set_title(f'Log Probability (Current: {metrics["logprob"][-1]:.4f})', fontweight='bold')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: KL and ratio
    axes[1, 0].plot(steps, metrics['kl'], 'purple', label='KL', linewidth=2)
    axes[1, 0].plot(steps, metrics['sft_kl'], 'orange', label='SFT KL', linewidth=2)
    axes[1, 0].set_title(f'KL Divergence (KL: {metrics["kl"][-1]:.4f}, SFT: {metrics["sft_kl"][-1]:.4f})',
                         fontweight='bold')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(steps, metrics['ratio'], 'brown', linewidth=2)
    axes[1, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_title(f'Importance Ratio (Current: {metrics["ratio"][-1]:.4f})', fontweight='bold')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(steps, metrics['clip_frac'], 'cyan', linewidth=2)
    axes[1, 2].set_title(f'Clip Fraction (Current: {metrics["clip_frac"][-1]:.4f})', fontweight='bold')
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].grid(True, alpha=0.3)

    # Row 3: Reward components
    axes[2, 0].plot(steps, metrics['pos_sim'], 'green', label='Pos Sim', linewidth=2)
    axes[2, 0].plot(steps, metrics['neg_sim'], 'red', label='Neg Sim', linewidth=2)
    axes[2, 0].set_title(f'Text-Motion Similarity (Pos: {metrics["pos_sim"][-1]:.4f}, Neg: {metrics["neg_sim"][-1]:.4f})',
                         fontweight='bold')
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(steps, metrics['phys'], 'blue', linewidth=2)
    axes[2, 1].set_title(f'Physical Reward (Current: {metrics["phys"][-1]:.4f})', fontweight='bold')
    axes[2, 1].set_xlabel('Step')
    axes[2, 1].grid(True, alpha=0.3)

    axes[2, 2].plot(steps, metrics['num'], 'magenta', label='Numerical', linewidth=2)
    axes[2, 2].plot(steps, metrics['kin'], 'olive', label='Kinematic', linewidth=2)
    axes[2, 2].set_title(f'Num & Kin Rewards (Num: {metrics["num"][-1]:.4f}, Kin: {metrics["kin"][-1]:.4f})',
                         fontweight='bold')
    axes[2, 2].set_xlabel('Step')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return True

def print_summary(metrics):
    """Print summary statistics"""
    if not metrics['step']:
        print("No metrics found yet...")
        return

    steps = metrics['step']
    current_epoch = metrics['epoch'][-1]
    current_batch = metrics['batch'][-1]
    total_batches = metrics.get('total_batches', 291)

    os.system('clear' if os.name != 'nt' else 'cls')

    print("=" * 80)
    print(f"{'GRPO TRAINING MONITOR':^80}")
    print("=" * 80)
    print(f"\nProgress: Epoch {current_epoch + 1}, Batch {current_batch}/{total_batches} ({current_batch/total_batches*100:.1f}%)")
    print(f"Total steps completed: {len(steps)}")

    print("\n" + "-" * 80)
    print(f"{'LATEST METRICS (Step ' + str(steps[-1]) + ')':^80}")
    print("-" * 80)

    print(f"\n{'Main Metrics:':<30}")
    print(f"  {'Reward:':<20} {metrics['reward'][-1]:>10.4f}  {'(avg: ' + f'{sum(metrics['reward'])/len(metrics['reward']):.4f})'}")
    print(f"  {'Loss:':<20} {metrics['loss'][-1]:>10.4f}  {'(avg: ' + f'{sum(metrics['loss'])/len(metrics['loss']):.4f})'}")
    print(f"  {'LogProb:':<20} {metrics['logprob'][-1]:>10.4f}  {'(avg: ' + f'{sum(metrics['logprob'])/len(metrics['logprob']):.4f})'}")

    print(f"\n{'Policy Metrics:':<30}")
    print(f"  {'KL:':<20} {metrics['kl'][-1]:>10.4f}")
    print(f"  {'SFT KL:':<20} {metrics['sft_kl'][-1]:>10.4f}")
    print(f"  {'Ratio:':<20} {metrics['ratio'][-1]:>10.4f}")
    print(f"  {'Clip Fraction:':<20} {metrics['clip_frac'][-1]:>10.4f}")
    print(f"  {'Learning Rate:':<20} {metrics['lr'][-1]:>10.2e}")

    print(f"\n{'Reward Components:':<30}")
    print(f"  {'Positive Similarity:':<20} {metrics['pos_sim'][-1]:>10.4f}")
    print(f"  {'Negative Similarity:':<20} {metrics['neg_sim'][-1]:>10.4f}")
    print(f"  {'Physical:':<20} {metrics['phys'][-1]:>10.4f}")
    print(f"  {'Numerical:':<20} {metrics['num'][-1]:>10.4f}")
    print(f"  {'Kinematic:':<20} {metrics['kin'][-1]:>10.4f}")

    # Show trend (last 5 batches)
    if len(metrics['reward']) >= 5:
        recent_rewards = metrics['reward'][-5:]
        trend = "↑" if recent_rewards[-1] > recent_rewards[0] else "↓"
        print(f"\n{'Recent Trend (last 5 batches):':<30}")
        print(f"  Reward: {trend} {recent_rewards[0]:.4f} → {recent_rewards[-1]:.4f}")

    print("\n" + "=" * 80)
    print(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Real-time GRPO training monitor')
    parser.add_argument('--log', type=str, required=True, help='Path to training log file')
    parser.add_argument('--output', type=str, default='training_metrics.png',
                        help='Output plot filename')
    parser.add_argument('--interval', type=int, default=30,
                        help='Update interval in seconds (default: 30)')
    args = parser.parse_args()

    print(f"Monitoring {args.log}")
    print(f"Updating every {args.interval} seconds...")
    print(f"Plot will be saved to {args.output}")
    print("\nPress Ctrl+C to stop\n")

    try:
        while True:
            metrics = parse_log(args.log)

            if metrics['step']:
                # Update plot
                plot_metrics(metrics, args.output)

                # Print summary
                print_summary(metrics)

                print(f"\n[Plot updated: {args.output}]")
            else:
                print(f"Waiting for training data... ({time.strftime('%H:%M:%S')})")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        if metrics['step']:
            print(f"Final plot saved to {args.output}")

if __name__ == '__main__':
    main()
