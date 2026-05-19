#!/usr/bin/env python3
"""
Monitor GRPO training progress and visualize metrics
"""
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

def parse_log(log_file):
    """Parse training log and extract metrics"""
    metrics = defaultdict(list)

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Match batch log lines
            # Example: Epoch [0] Batch [1/291] Loss: -0.0038, Reward: 1.0441, LogProb: -0.4694, KL: 0.0020, ...
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

                # Global step
                step = epoch * total_batches + batch

                metrics['step'].append(step)
                metrics['epoch'].append(epoch)
                metrics['batch'].append(batch)
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

    return metrics

def plot_metrics(metrics, output_file='training_metrics.png'):
    """Plot training metrics"""
    if not metrics['step']:
        print("No metrics found in log file!")
        return

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('GRPO Training Metrics', fontsize=16)

    steps = metrics['step']

    # Row 1: Main metrics
    axes[0, 0].plot(steps, metrics['reward'], 'b-', linewidth=2)
    axes[0, 0].set_title('Reward')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, metrics['loss'], 'r-', linewidth=2)
    axes[0, 1].set_title('Loss')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(steps, metrics['logprob'], 'g-', linewidth=2)
    axes[0, 2].set_title('Log Probability')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: KL and ratio
    axes[1, 0].plot(steps, metrics['kl'], 'purple', label='KL', linewidth=2)
    axes[1, 0].plot(steps, metrics['sft_kl'], 'orange', label='SFT KL', linewidth=2)
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(steps, metrics['ratio'], 'brown', linewidth=2)
    axes[1, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Importance Ratio')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(steps, metrics['clip_frac'], 'cyan', linewidth=2)
    axes[1, 2].set_title('Clip Fraction')
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].grid(True, alpha=0.3)

    # Row 3: Reward components
    axes[2, 0].plot(steps, metrics['pos_sim'], 'green', label='Pos Sim', linewidth=2)
    axes[2, 0].plot(steps, metrics['neg_sim'], 'red', label='Neg Sim', linewidth=2)
    axes[2, 0].set_title('Text-Motion Similarity')
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(steps, metrics['phys'], 'blue', linewidth=2)
    axes[2, 1].set_title('Physical Reward')
    axes[2, 1].set_xlabel('Step')
    axes[2, 1].grid(True, alpha=0.3)

    axes[2, 2].plot(steps, metrics['num'], 'magenta', label='Numerical', linewidth=2)
    axes[2, 2].plot(steps, metrics['kin'], 'olive', label='Kinematic', linewidth=2)
    axes[2, 2].set_title('Numerical & Kinematic Rewards')
    axes[2, 2].set_xlabel('Step')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total steps: {len(steps)}")
    print(f"Epochs: {metrics['epoch'][-1] + 1}")
    print(f"\nLatest metrics (Step {steps[-1]}):")
    print(f"  Reward:      {metrics['reward'][-1]:.4f}")
    print(f"  Loss:        {metrics['loss'][-1]:.4f}")
    print(f"  LogProb:     {metrics['logprob'][-1]:.4f}")
    print(f"  KL:          {metrics['kl'][-1]:.4f}")
    print(f"  SFT KL:      {metrics['sft_kl'][-1]:.4f}")
    print(f"  Ratio:       {metrics['ratio'][-1]:.4f}")
    print(f"  Clip Frac:   {metrics['clip_frac'][-1]:.4f}")
    print(f"\nReward components:")
    print(f"  Pos Sim:     {metrics['pos_sim'][-1]:.4f}")
    print(f"  Neg Sim:     {metrics['neg_sim'][-1]:.4f}")
    print(f"  Physical:    {metrics['phys'][-1]:.4f}")
    print(f"  Numerical:   {metrics['num'][-1]:.4f}")
    print(f"  Kinematic:   {metrics['kin'][-1]:.4f}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Monitor GRPO training')
    parser.add_argument('--log', type=str,
                       default='experiments_grpo/grpo_kinematic/run_grpo.log',
                       help='Path to training log file')
    parser.add_argument('--output', type=str,
                       default='training_metrics.png',
                       help='Output plot filename')
    args = parser.parse_args()

    print(f"Parsing log file: {args.log}")
    metrics = parse_log(args.log)

    if metrics['step']:
        plot_metrics(metrics, args.output)
    else:
        print("No training metrics found in log file yet.")

if __name__ == '__main__':
    main()
