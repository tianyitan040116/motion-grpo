#!/usr/bin/env python3
"""
Training Health Analyzer for GRPO
Analyzes training metrics trends and provides health assessment
"""

import re
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class TrendStatus(Enum):
    IMPROVING = "✓ 改善中"
    STABLE = "→ 稳定"
    DEGRADING = "✗ 恶化"
    INSUFFICIENT_DATA = "? 数据不足"
    ABNORMAL = "⚠ 异常"

class HealthStatus(Enum):
    HEALTHY = "🟢 健康"
    WARNING = "🟡 警告"
    CRITICAL = "🔴 严重"
    UNKNOWN = "⚪ 未知"

@dataclass
class MetricAnalysis:
    name: str
    values: List[float]
    trend: TrendStatus
    health: HealthStatus
    mean: float
    std: float
    recent_mean: float  # Last 20% of data
    slope: float  # Linear regression slope
    issues: List[str]

class TrainingHealthAnalyzer:
    def __init__(self, log_path: str, window_size: int = 10):
        self.log_path = Path(log_path)
        self.window_size = window_size
        self.metrics: Dict[str, List[float]] = {}
        self.batch_numbers: List[int] = []

        # Define expected behavior for each metric
        self.metric_expectations = {
            'reward': 'increase',  # Higher is better
            'loss': 'decrease',    # Lower is better
            'kl': 'stable_low',    # Should be low and stable
            'sft_kl': 'stable_low',
            'clip_frac': 'moderate', # 0.1-0.3 is good
            'pos_sim': 'increase',  # Higher similarity to positive examples
            'neg_sim': 'decrease',  # Lower similarity to negative examples
            'phys': 'increase',     # Physical plausibility
            'num': 'increase',      # Numerical accuracy
            'kin': 'increase',      # Kinematic quality
            'logprob': 'stable',    # Should be stable
            'ratio': 'near_one',    # Should stay near 1.0
        }

    def parse_log(self):
        """Parse training log and extract metrics"""
        if not self.log_path.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_path}")

        pattern = re.compile(
            r'Batch \[(\d+)/\d+\] '
            r'Loss: ([-\d.]+), '
            r'Reward: ([\d.]+), '
            r'LogProb: ([-\d.]+), '
            r'KL: ([\d.]+), '
            r'SFT_KL: ([\d.]+), '
            r'Ratio: ([\d.]+), '
            r'ClipFrac: ([\d.]+), '
            r'InnerK: \d+, '
            r'LR: [\d.e+-]+, '
            r'PosSim: ([\d.]+), '
            r'NegSim: ([\d.]+), '
            r'Phys: ([\d.]+), '
            r'Num: ([\d.]+), '
            r'Kin: ([\d.]+)'
        )

        with open(self.log_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    batch_num = int(match.group(1))
                    self.batch_numbers.append(batch_num)

                    metrics_data = {
                        'loss': float(match.group(2)),
                        'reward': float(match.group(3)),
                        'logprob': float(match.group(4)),
                        'kl': float(match.group(5)),
                        'sft_kl': float(match.group(6)),
                        'ratio': float(match.group(7)),
                        'clip_frac': float(match.group(8)),
                        'pos_sim': float(match.group(9)),
                        'neg_sim': float(match.group(10)),
                        'phys': float(match.group(11)),
                        'num': float(match.group(12)),
                        'kin': float(match.group(13)),
                    }

                    for key, value in metrics_data.items():
                        if key not in self.metrics:
                            self.metrics[key] = []
                        self.metrics[key].append(value)

        print(f"✓ 解析完成: {len(self.batch_numbers)} 个batch的数据")

    def calculate_trend(self, values: List[float], expectation: str) -> Tuple[TrendStatus, float, List[str]]:
        """Calculate trend and detect issues"""
        issues = []

        if len(values) < 3:
            return TrendStatus.INSUFFICIENT_DATA, 0.0, issues

        # Check for NaN/Inf
        if any(np.isnan(v) or np.isinf(v) for v in values):
            issues.append("包含NaN或Inf值")
            return TrendStatus.ABNORMAL, 0.0, issues

        # Calculate linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        # Calculate recent trend (last 20% vs first 20%)
        split = max(3, len(values) // 5)
        early_mean = np.mean(values[:split])
        recent_mean = np.mean(values[-split:])

        # Check variance
        std = np.std(values)
        mean = np.mean(values)
        cv = std / (abs(mean) + 1e-8)  # Coefficient of variation

        if cv > 0.5:
            issues.append(f"波动过大 (CV={cv:.2f})")

        # Determine trend based on expectation
        if expectation == 'increase':
            if slope > 0.001 and recent_mean > early_mean * 1.05:
                return TrendStatus.IMPROVING, slope, issues
            elif slope < -0.001 and recent_mean < early_mean * 0.95:
                issues.append("指标下降（期望上升）")
                return TrendStatus.DEGRADING, slope, issues
            else:
                return TrendStatus.STABLE, slope, issues

        elif expectation == 'decrease':
            if slope < -0.001 and recent_mean < early_mean * 0.95:
                return TrendStatus.IMPROVING, slope, issues
            elif slope > 0.001 and recent_mean > early_mean * 1.05:
                issues.append("指标上升（期望下降）")
                return TrendStatus.DEGRADING, slope, issues
            else:
                return TrendStatus.STABLE, slope, issues

        elif expectation == 'stable_low':
            if mean > 0.1:
                issues.append(f"数值过高 (mean={mean:.3f})")
            if cv > 0.3:
                return TrendStatus.DEGRADING, slope, issues
            return TrendStatus.STABLE, slope, issues

        elif expectation == 'moderate':
            if mean < 0.05 or mean > 0.5:
                issues.append(f"数值异常 (mean={mean:.3f}, 期望0.1-0.3)")
            return TrendStatus.STABLE, slope, issues

        elif expectation == 'near_one':
            if abs(mean - 1.0) > 0.3:
                issues.append(f"偏离1.0过多 (mean={mean:.3f})")
                return TrendStatus.DEGRADING, slope, issues
            return TrendStatus.STABLE, slope, issues

        else:  # 'stable'
            if cv > 0.2:
                issues.append("波动过大")
                return TrendStatus.DEGRADING, slope, issues
            return TrendStatus.STABLE, slope, issues

    def assess_health(self, trend: TrendStatus, issues: List[str], metric_name: str) -> HealthStatus:
        """Assess overall health of a metric"""
        if trend == TrendStatus.ABNORMAL:
            return HealthStatus.CRITICAL

        if trend == TrendStatus.INSUFFICIENT_DATA:
            return HealthStatus.UNKNOWN

        # Critical metrics
        critical_metrics = ['reward', 'loss', 'kl']

        if metric_name in critical_metrics:
            if trend == TrendStatus.DEGRADING:
                return HealthStatus.CRITICAL
            elif len(issues) > 0:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY
        else:
            if trend == TrendStatus.DEGRADING and len(issues) > 1:
                return HealthStatus.WARNING
            elif len(issues) > 0:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY

    def analyze_all_metrics(self) -> Dict[str, MetricAnalysis]:
        """Analyze all metrics"""
        analyses = {}

        for metric_name, values in self.metrics.items():
            expectation = self.metric_expectations.get(metric_name, 'stable')
            trend, slope, issues = self.calculate_trend(values, expectation)
            health = self.assess_health(trend, issues, metric_name)

            split = max(3, len(values) // 5)
            recent_mean = np.mean(values[-split:]) if len(values) >= split else np.mean(values)

            analyses[metric_name] = MetricAnalysis(
                name=metric_name,
                values=values,
                trend=trend,
                health=health,
                mean=np.mean(values),
                std=np.std(values),
                recent_mean=recent_mean,
                slope=slope,
                issues=issues
            )

        return analyses

    def generate_report(self, analyses: Dict[str, MetricAnalysis]) -> str:
        """Generate text report"""
        report = []
        report.append("=" * 80)
        report.append("训练健康分析报告".center(80))
        report.append("=" * 80)
        report.append(f"\n总batch数: {len(self.batch_numbers)}")
        report.append(f"分析指标数: {len(analyses)}\n")

        # Overall health assessment
        health_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.CRITICAL: 0,
            HealthStatus.UNKNOWN: 0
        }

        for analysis in analyses.values():
            health_counts[analysis.health] += 1

        report.append("=" * 80)
        report.append("整体健康状态")
        report.append("=" * 80)

        total = len(analyses)
        healthy_pct = health_counts[HealthStatus.HEALTHY] / total * 100
        warning_pct = health_counts[HealthStatus.WARNING] / total * 100
        critical_pct = health_counts[HealthStatus.CRITICAL] / total * 100

        report.append(f"🟢 健康: {health_counts[HealthStatus.HEALTHY]}/{total} ({healthy_pct:.1f}%)")
        report.append(f"🟡 警告: {health_counts[HealthStatus.WARNING]}/{total} ({warning_pct:.1f}%)")
        report.append(f"🔴 严重: {health_counts[HealthStatus.CRITICAL]}/{total} ({critical_pct:.1f}%)")

        if critical_pct > 30:
            report.append("\n⚠️  训练状态严重: 超过30%的指标处于严重状态")
            overall_status = "🔴 严重问题"
        elif warning_pct + critical_pct > 50:
            report.append("\n⚠️  训练状态警告: 超过50%的指标有问题")
            overall_status = "🟡 需要关注"
        else:
            report.append("\n✓ 训练状态良好")
            overall_status = "🟢 健康"

        report.append(f"\n总体评估: {overall_status}\n")

        # Detailed metrics
        report.append("=" * 80)
        report.append("详细指标分析")
        report.append("=" * 80)

        # Sort by health (critical first)
        sorted_analyses = sorted(
            analyses.items(),
            key=lambda x: (x[1].health.value, x[0])
        )

        for metric_name, analysis in sorted_analyses:
            report.append(f"\n【{metric_name.upper()}】")
            report.append(f"  状态: {analysis.health.value} | 趋势: {analysis.trend.value}")
            report.append(f"  均值: {analysis.mean:.4f} | 标准差: {analysis.std:.4f}")
            report.append(f"  最近均值: {analysis.recent_mean:.4f} | 斜率: {analysis.slope:.6f}")

            if analysis.issues:
                report.append(f"  问题:")
                for issue in analysis.issues:
                    report.append(f"    - {issue}")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def plot_analysis(self, analyses: Dict[str, MetricAnalysis], output_path: str):
        """Generate visualization"""
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        fig.suptitle('训练健康分析', fontsize=16, fontweight='bold')

        metric_order = ['reward', 'loss', 'kl', 'sft_kl', 'logprob', 'ratio',
                       'clip_frac', 'pos_sim', 'neg_sim', 'phys', 'num', 'kin']

        for idx, metric_name in enumerate(metric_order):
            if metric_name not in analyses:
                continue

            analysis = analyses[metric_name]
            ax = axes[idx // 3, idx % 3]

            # Plot values
            x = self.batch_numbers[:len(analysis.values)]
            ax.plot(x, analysis.values, 'o-', markersize=3, linewidth=1, alpha=0.7)

            # Plot moving average
            if len(analysis.values) >= self.window_size:
                ma = np.convolve(analysis.values,
                               np.ones(self.window_size)/self.window_size,
                               mode='valid')
                ma_x = x[self.window_size-1:]
                ax.plot(ma_x, ma, 'r-', linewidth=2, label=f'MA({self.window_size})')

            # Plot trend line
            if len(analysis.values) >= 3:
                z = np.polyfit(range(len(analysis.values)), analysis.values, 1)
                p = np.poly1d(z)
                ax.plot(x, p(range(len(analysis.values))), 'g--',
                       linewidth=1.5, alpha=0.7, label='趋势')

            # Color background based on health
            if analysis.health == HealthStatus.CRITICAL:
                ax.set_facecolor('#ffebee')
            elif analysis.health == HealthStatus.WARNING:
                ax.set_facecolor('#fff9e6')
            else:
                ax.set_facecolor('#f1f8f4')

            # Title with status
            title = f"{metric_name.upper()}\n{analysis.trend.value} | {analysis.health.value}"
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel('Batch')
            ax.set_ylabel('Value')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 可视化已保存: {output_path}")

    def run_analysis(self, output_dir: Optional[str] = None):
        """Run complete analysis"""
        print("开始分析训练健康状态...")

        # Parse log
        self.parse_log()

        if len(self.batch_numbers) == 0:
            print("❌ 未找到训练数据")
            return

        # Analyze metrics
        analyses = self.analyze_all_metrics()

        # Generate report
        report = self.generate_report(analyses)
        print(report)

        # Save report
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            report_path = output_dir / "health_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n✓ 报告已保存: {report_path}")

            # Generate plot
            plot_path = output_dir / "health_analysis.png"
            self.plot_analysis(analyses, str(plot_path))

def main():
    parser = argparse.ArgumentParser(description='分析GRPO训练健康状态')
    parser.add_argument('--log', type=str, required=True, help='训练日志路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录（可选）')
    parser.add_argument('--window', type=int, default=10, help='移动平均窗口大小')

    args = parser.parse_args()

    analyzer = TrainingHealthAnalyzer(args.log, window_size=args.window)
    analyzer.run_analysis(output_dir=args.output)

if __name__ == '__main__':
    main()
