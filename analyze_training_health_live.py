#!/usr/bin/env python3
"""
Real-time Training Health Analyzer for GRPO
Continuously monitors training and updates health assessment
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
import time
import os
from datetime import datetime

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
    recent_mean: float
    slope: float
    issues: List[str]
    initial_value: float  # First N batches average
    best_value: float     # Historical best
    improvement_pct: float  # % improvement from initial
    distance_to_best_pct: float  # % distance to best

class TrainingHealthAnalyzer:
    def __init__(self, log_path: str, window_size: int = 10):
        self.log_path = Path(log_path)
        self.window_size = window_size
        self.metrics: Dict[str, List[float]] = {}
        self.batch_numbers: List[int] = []

        self.metric_expectations = {
            'reward': 'increase',
            'loss': 'decrease',
            'kl': 'stable_low',
            'sft_kl': 'stable_low',
            'clip_frac': 'moderate',
            'pos_sim': 'increase',
            'neg_sim': 'decrease',
            'phys': 'increase',
            'num': 'increase',
            'kin': 'increase',
            'logprob': 'stable',
            'ratio': 'near_one',
        }

    def parse_log(self):
        """Parse training log and extract metrics"""
        if not self.log_path.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_path}")

        # Reset metrics
        self.metrics = {}
        self.batch_numbers = []

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

    def calculate_trend(self, values: List[float], expectation: str) -> Tuple[TrendStatus, float, List[str]]:
        """Calculate trend and detect issues"""
        issues = []

        if len(values) < 3:
            return TrendStatus.INSUFFICIENT_DATA, 0.0, issues

        if any(np.isnan(v) or np.isinf(v) for v in values):
            issues.append("包含NaN或Inf值")
            return TrendStatus.ABNORMAL, 0.0, issues

        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        split = max(3, len(values) // 5)
        early_mean = np.mean(values[:split])
        recent_mean = np.mean(values[-split:])

        std = np.std(values)
        mean = np.mean(values)
        cv = std / (abs(mean) + 1e-8)

        if cv > 0.5:
            issues.append(f"波动过大 (CV={cv:.2f})")

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
            if mean < 0.05:
                issues.append(f"数值异常 (mean={mean:.3f}, 期望0.1-0.3)")
                return TrendStatus.DEGRADING, slope, issues
            elif mean > 0.5:
                issues.append(f"数值过高 (mean={mean:.3f})")
                return TrendStatus.DEGRADING, slope, issues
            return TrendStatus.STABLE, slope, issues

        elif expectation == 'near_one':
            if abs(mean - 1.0) > 0.1:
                issues.append(f"偏离1.0过多 (mean={mean:.3f})")
                return TrendStatus.DEGRADING, slope, issues
            return TrendStatus.STABLE, slope, issues

        else:  # stable
            if cv > 0.3:
                issues.append(f"波动过大 (CV={cv:.2f})")
                return TrendStatus.DEGRADING, slope, issues
            return TrendStatus.STABLE, slope, issues

    def determine_health(self, metric_name: str, trend: TrendStatus, values: List[float], issues: List[str]) -> HealthStatus:
        """Determine health status"""
        if trend == TrendStatus.ABNORMAL:
            return HealthStatus.CRITICAL

        if len(issues) > 0:
            if trend == TrendStatus.DEGRADING:
                return HealthStatus.CRITICAL
            else:
                return HealthStatus.WARNING

        if trend == TrendStatus.DEGRADING:
            return HealthStatus.WARNING

        if trend == TrendStatus.IMPROVING:
            return HealthStatus.HEALTHY

        return HealthStatus.HEALTHY

    def analyze_all_metrics(self) -> Dict[str, MetricAnalysis]:
        """Analyze all metrics"""
        analyses = {}

        # Skip warmup period (first 10 batches) for baseline calculation
        warmup_batches = 10

        for metric_name, values in self.metrics.items():
            expectation = self.metric_expectations.get(metric_name, 'stable')
            trend, slope, issues = self.calculate_trend(values, expectation)
            health = self.determine_health(metric_name, trend, values, issues)

            split = max(3, len(values) // 5)
            recent_mean = np.mean(values[-split:]) if len(values) >= split else np.mean(values)

            # Calculate baseline (after warmup)
            if len(values) > warmup_batches + 10:
                # Use batches 11-20 as baseline (after warmup, before too much training)
                baseline_values = values[warmup_batches:warmup_batches+10]
                initial_value = np.mean(baseline_values)
            elif len(values) > warmup_batches:
                # Not enough data, use what we have after warmup
                initial_value = np.mean(values[warmup_batches:])
            else:
                # Still in warmup, use all data
                initial_value = np.mean(values)

            # Find best value (depends on metric expectation)
            if expectation in ['increase', 'stable']:
                best_value = np.max(values)
            elif expectation in ['decrease', 'stable_low']:
                best_value = np.min(values)
            elif expectation == 'near_one':
                # Best is closest to 1.0
                best_value = values[np.argmin(np.abs(np.array(values) - 1.0))]
            elif expectation == 'moderate':
                # Best is closest to 0.2 (middle of 0.1-0.3)
                best_value = values[np.argmin(np.abs(np.array(values) - 0.2))]
            else:
                best_value = recent_mean

            # Calculate improvement percentage
            if abs(initial_value) > 1e-8:
                if expectation in ['increase', 'stable']:
                    improvement_pct = ((recent_mean - initial_value) / abs(initial_value)) * 100
                elif expectation in ['decrease', 'stable_low']:
                    improvement_pct = ((initial_value - recent_mean) / abs(initial_value)) * 100
                else:
                    improvement_pct = 0.0
            else:
                improvement_pct = 0.0

            # Calculate distance to best
            if abs(best_value) > 1e-8:
                if expectation in ['increase', 'stable']:
                    distance_to_best_pct = ((best_value - recent_mean) / abs(best_value)) * 100
                elif expectation in ['decrease', 'stable_low']:
                    distance_to_best_pct = ((recent_mean - best_value) / abs(best_value)) * 100
                else:
                    distance_to_best_pct = abs((recent_mean - best_value) / best_value) * 100
            else:
                distance_to_best_pct = 0.0

            analyses[metric_name] = MetricAnalysis(
                name=metric_name.upper(),
                values=values,
                trend=trend,
                health=health,
                mean=np.mean(values),
                std=np.std(values),
                recent_mean=recent_mean,
                slope=slope,
                issues=issues,
                initial_value=initial_value,
                best_value=best_value,
                improvement_pct=improvement_pct,
                distance_to_best_pct=distance_to_best_pct
            )

        return analyses

    def generate_summary(self, analyses: Dict[str, MetricAnalysis]) -> str:
        """Generate compact summary for terminal"""
        health_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.CRITICAL: 0
        }

        for analysis in analyses.values():
            health_counts[analysis.health] += 1

        total = len(analyses)
        summary = f"\n{'='*80}\n"
        summary += f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += f"总batch数: {len(self.batch_numbers)} | 分析指标数: {total}\n"
        summary += f"{'='*80}\n"

        # Overall status
        if health_counts[HealthStatus.CRITICAL] > total * 0.3:
            overall = "🔴 严重"
        elif health_counts[HealthStatus.WARNING] > total * 0.5:
            overall = "🟡 警告"
        else:
            overall = "🟢 健康"

        summary += f"整体状态: {overall}\n"
        summary += f"  健康: {health_counts[HealthStatus.HEALTHY]}/{total} | "
        summary += f"警告: {health_counts[HealthStatus.WARNING]}/{total} | "
        summary += f"严重: {health_counts[HealthStatus.CRITICAL]}/{total}\n"
        summary += f"{'='*80}\n"

        # Improvement trends - show what's getting better
        improving = []
        stable = []
        degrading = []

        for metric, analysis in analyses.items():
            if analysis.trend == TrendStatus.IMPROVING:
                improving.append(analysis.name)
            elif analysis.trend == TrendStatus.STABLE:
                stable.append(analysis.name)
            elif analysis.trend == TrendStatus.DEGRADING:
                degrading.append(analysis.name)

        if improving:
            summary += f"✓ 改善中 ({len(improving)}): {', '.join(improving)}\n"
        if stable:
            summary += f"→ 稳定 ({len(stable)}): {', '.join(stable)}\n"
        if degrading:
            summary += f"✗ 恶化 ({len(degrading)}): {', '.join(degrading)}\n"
        summary += f"{'='*80}\n"

        # Show all metrics with multiple window comparisons
        all_metrics = sorted(analyses.keys())

        # Define multiple windows
        total_batches = len(self.batch_numbers)
        windows = []
        for w in [5, 10, 20, 40]:
            if total_batches >= w:
                windows.append(w)

        # If not enough data, use what we have
        if not windows:
            windows = [min(3, total_batches)]

        summary += f"指标详情 (对比窗口: {', '.join(map(str, windows))} batch):\n"
        summary += f"{'='*80}\n"

        for metric in all_metrics:
            a = analyses[metric]

            # Calculate values for different windows
            window_values = []
            for w in windows:
                if len(a.values) >= w:
                    val = np.mean(a.values[-w:])
                    window_values.append(f"{val:.4f}")
                else:
                    window_values.append("N/A")

            # Calculate change from earliest to latest window
            if len(a.values) >= windows[-1] * 2:
                early_val = np.mean(a.values[:windows[-1]])
                recent_val = np.mean(a.values[-windows[0]:])
                change_pct = ((recent_val - early_val) / (abs(early_val) + 1e-8)) * 100
                change_str = f"({change_pct:+.1f}%)"
            else:
                change_str = ""

            summary += f"\n【{a.name}】{a.health.value} {a.trend.value}\n"

            # Show improvement from baseline
            if a.improvement_pct != 0:
                if a.improvement_pct > 0:
                    summary += f"  📈 相比基线改善: +{a.improvement_pct:.1f}% (基线={a.initial_value:.4f}, 最佳={a.best_value:.4f})\n"
                else:
                    summary += f"  📉 相比基线退步: {a.improvement_pct:.1f}% (基线={a.initial_value:.4f}, 最佳={a.best_value:.4f})\n"

            # Show distance to best
            if abs(a.distance_to_best_pct) > 0.1:
                summary += f"  🎯 距离历史最佳: {a.distance_to_best_pct:.1f}%\n"

            # Show window averages
            if len(windows) == 1:
                summary += f"  最近{windows[0]}批: {window_values[0]}\n"
            elif len(windows) == 2:
                summary += f"  最近{windows[0]}批: {window_values[0]} | 最近{windows[1]}批: {window_values[1]}\n"
            elif len(windows) == 3:
                summary += f"  最近{windows[0]}批: {window_values[0]} | {windows[1]}批: {window_values[1]} | {windows[2]}批: {window_values[2]}\n"
            else:
                summary += f"  最近{windows[0]}批: {window_values[0]} | {windows[1]}批: {window_values[1]} | {windows[2]}批: {window_values[2]} | {windows[3]}批: {window_values[3]}\n"

            summary += f"  全局: 均值={a.mean:.4f}, 标准差={a.std:.4f}, 斜率={a.slope:.6f}\n"

            if a.issues:
                for issue in a.issues:
                    summary += f"  ⚠ {issue}\n"

        summary += f"{'='*80}\n"
        return summary

def main():
    parser = argparse.ArgumentParser(description='实时监控GRPO训练健康状态')
    parser.add_argument('--log', type=str, required=True, help='训练日志路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录（可选）')
    parser.add_argument('--interval', type=int, default=60, help='刷新间隔（秒）')
    parser.add_argument('--window', type=int, default=10, help='移动平均窗口大小')

    args = parser.parse_args()

    analyzer = TrainingHealthAnalyzer(args.log, window_size=args.window)

    print(f"开始实时监控训练健康状态...")
    print(f"日志文件: {args.log}")
    print(f"刷新间隔: {args.interval}秒")
    print(f"按 Ctrl+C 停止监控\n")

    try:
        while True:
            # Clear screen
            os.system('clear' if os.name != 'nt' else 'cls')

            # Parse and analyze
            try:
                analyzer.parse_log()

                if len(analyzer.batch_numbers) == 0:
                    print("等待训练数据...")
                else:
                    analyses = analyzer.analyze_all_metrics()
                    summary = analyzer.generate_summary(analyses)
                    print(summary)

                    # Save detailed report if output dir specified
                    if args.output:
                        output_dir = Path(args.output)
                        output_dir.mkdir(parents=True, exist_ok=True)

                        report_path = output_dir / "health_report_latest.txt"
                        with open(report_path, 'w', encoding='utf-8') as f:
                            f.write(summary)

            except Exception as e:
                print(f"错误: {e}")

            # Wait for next update
            print(f"\n下次更新: {args.interval}秒后...")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\n监控已停止")

if __name__ == '__main__':
    main()
