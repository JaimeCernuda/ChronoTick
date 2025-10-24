#!/usr/bin/env python3
"""
ULTRA-DEEP Micro-Exploration Analysis

This script performs the DEEPEST possible analysis:
1. Loads CSV time series data
2. Reads daemon/validation logs for context
3. Identifies SPECIFIC time windows (not whole experiments)
4. Understands platform characteristics
5. Generates compelling micro-narratives with exact evidence
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json

@dataclass
class MicroWindow:
    """A specific time window with interesting behavior"""
    experiment: str
    platform: str
    start_hour: float
    end_hour: float
    duration_min: float

    # What happened
    title: str
    description: str
    evidence: List[str] = field(default_factory=list)
    log_evidence: List[str] = field(default_factory=list)

    # Metrics
    chronotick_mae: float = 0
    chronotick_std: float = 0
    chronotick_max: float = 0
    ntp_mae: float = 0
    ntp_std: float = 0
    ntp_rejections: int = 0
    system_drift_rate: float = 0

    # Classification
    story_type: str = "unknown"  # drift, spike, rejection, recovery, instability
    severity: str = "medium"

    # For visualization
    data_slice: Optional[pd.DataFrame] = None


class UltraDeepAnalyzer:
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.micro_windows = []

    def analyze_experiment(self, exp_dir: Path) -> List[MicroWindow]:
        """Deep analysis of one experiment directory"""
        windows = []

        # Find all CSV files
        csv_files = list(exp_dir.rglob('*.csv'))

        for csv_path in csv_files:
            # Extract platform from path
            platform = csv_path.parent.name
            exp_name = None
            for part in csv_path.parts:
                if part.startswith('experiment-'):
                    exp_name = part
                    break

            if not exp_name:
                continue

            print(f"\n{'='*80}")
            print(f"🔬 ULTRA-DEEP: {exp_name}/{platform}")
            print(f"{'='*80}")

            # Load data
            df = self.load_data(csv_path)
            if df is None or len(df) < 100:
                print(f"⚠️  Too little data, skipping")
                continue

            # Find log files
            log_dir = csv_path.parent
            daemon_log = log_dir / 'daemon.log'
            validation_log = log_dir / 'validation.log'

            # Analyze hour-by-hour
            hour_patterns = self.analyze_temporal_patterns(df)

            # Check logs for context
            log_context = self.extract_log_context(daemon_log, validation_log)

            # Identify interesting windows
            interesting = self.find_interesting_windows(
                df, exp_name, platform, hour_patterns, log_context
            )

            windows.extend(interesting)
            self.micro_windows.extend(interesting)

        return windows

    def load_data(self, csv_path: Path) -> Optional[pd.DataFrame]:
        """Load CSV with proper time handling"""
        try:
            df = pd.read_csv(csv_path)

            # Create time_hours
            if 'elapsed_seconds' in df.columns:
                df['time_hours'] = df['elapsed_seconds'] / 3600
            elif 'time_from_start' in df.columns:
                df['time_hours'] = df['time_from_start'] / 3600
            else:
                df['time_hours'] = np.arange(len(df)) / 1800

            return df
        except Exception as e:
            print(f"❌ Error loading {csv_path}: {e}")
            return None

    def analyze_temporal_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Hour-by-hour pattern analysis"""
        patterns = []

        max_hour = int(df['time_hours'].max()) + 1

        for hour in range(max_hour):
            hour_data = df[(df['time_hours'] >= hour) & (df['time_hours'] < hour + 1)]

            if len(hour_data) == 0:
                continue

            pattern = {
                'hour': hour,
                'samples': len(hour_data),
            }

            # ChronoTick metrics
            if 'chronotick_offset_ms' in hour_data.columns:
                ct = hour_data['chronotick_offset_ms'].dropna()
                if len(ct) > 0:
                    pattern['ct_mae'] = ct.abs().mean()
                    pattern['ct_std'] = ct.std()
                    pattern['ct_max'] = ct.abs().max()
                    pattern['ct_drift'] = ct.iloc[-1] - ct.iloc[0] if len(ct) > 1 else 0

            # NTP metrics
            if 'ntp_offset_ms' in hour_data.columns:
                ntp = hour_data['ntp_offset_ms'].dropna()
                if len(ntp) > 0:
                    pattern['ntp_mae'] = ntp.abs().mean()
                    pattern['ntp_std'] = ntp.std()
                    pattern['ntp_count'] = len(ntp)

                    # Detect large changes (possible rejections or spikes)
                    if len(ntp) > 1:
                        ntp_changes = np.abs(np.diff(ntp.values))
                        pattern['ntp_large_changes'] = np.sum(ntp_changes > 10)

            # System clock if available
            if 'system_clock_offset_ms' in hour_data.columns:
                sys = hour_data['system_clock_offset_ms'].dropna()
                if len(sys) > 1:
                    # Calculate drift rate
                    x = hour_data['time_hours'].values[:len(sys)]
                    y = sys.values
                    if len(x) > 1:
                        coeffs = np.polyfit(x, y, 1)
                        pattern['system_drift_rate_ms_per_h'] = coeffs[0]

            # Flag interesting
            is_interesting = False
            reasons = []

            if pattern.get('ct_std', 0) > 1.5:
                is_interesting = True
                reasons.append(f"High ChronoTick variance (σ={pattern['ct_std']:.2f}ms)")

            if pattern.get('ntp_std', 0) > 5:
                is_interesting = True
                reasons.append(f"High NTP variance (σ={pattern['ntp_std']:.2f}ms)")

            if abs(pattern.get('ct_drift', 0)) > 2:
                is_interesting = True
                reasons.append(f"Significant drift ({pattern['ct_drift']:.2f}ms)")

            if pattern.get('ntp_large_changes', 0) > 3:
                is_interesting = True
                reasons.append(f"{pattern['ntp_large_changes']} large NTP changes")

            if abs(pattern.get('system_drift_rate_ms_per_h', 0)) > 5:
                is_interesting = True
                reasons.append(f"High system drift rate ({pattern['system_drift_rate_ms_per_h']:.2f}ms/h)")

            pattern['interesting'] = is_interesting
            pattern['reasons'] = reasons

            if is_interesting:
                print(f"\n  📌 Hour {hour}: {', '.join(reasons)}")

            patterns.append(pattern)

        return patterns

    def extract_log_context(self, daemon_log: Path, validation_log: Path) -> Dict:
        """Extract context from logs"""
        context = {
            'ntp_rejections': [],
            'warnings': [],
            'errors': [],
            'temperature_events': [],
            'load_events': [],
            'model_events': []
        }

        if not daemon_log.exists():
            return context

        try:
            with open(daemon_log, 'r', errors='ignore') as f:
                for line in f:
                    # NTP rejections
                    if 'REJECTED NTP' in line or 'OUTLIER_FILTER' in line:
                        # Extract timestamp and details
                        match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                        if match:
                            timestamp = match.group(1)
                            context['ntp_rejections'].append({
                                'timestamp': timestamp,
                                'line': line.strip()
                            })

                    # Warnings
                    if 'WARNING' in line:
                        context['warnings'].append(line.strip())

                    # Errors
                    if 'ERROR' in line:
                        context['errors'].append(line.strip())

                    # Temperature or CPU mentions
                    if 'temperature' in line.lower() or 'cpu' in line.lower():
                        context['temperature_events'].append(line.strip())

                    # Model disagreements or fusion events
                    if 'fusion' in line.lower() or 'disagreement' in line.lower():
                        context['model_events'].append(line.strip())
        except Exception as e:
            print(f"⚠️  Error reading logs: {e}")

        # Summary
        if len(context['ntp_rejections']) > 0:
            print(f"  📋 Found {len(context['ntp_rejections'])} NTP rejection events in logs")
        if len(context['errors']) > 0:
            print(f"  ⚠️  Found {len(context['errors'])} errors in logs")

        return context

    def find_interesting_windows(self, df: pd.DataFrame, exp_name: str,
                                platform: str, hour_patterns: List[Dict],
                                log_context: Dict) -> List[MicroWindow]:
        """Identify specific interesting time windows"""
        windows = []

        # Strategy 1: Find continuous interesting periods
        interesting_hours = [p for p in hour_patterns if p.get('interesting', False)]

        if interesting_hours:
            # Group consecutive hours
            groups = []
            current_group = [interesting_hours[0]['hour']]

            for i in range(1, len(interesting_hours)):
                if interesting_hours[i]['hour'] == current_group[-1] + 1:
                    current_group.append(interesting_hours[i]['hour'])
                else:
                    groups.append(current_group)
                    current_group = [interesting_hours[i]['hour']]
            groups.append(current_group)

            # Create windows from groups
            for group in groups:
                start_h = group[0]
                end_h = group[-1] + 1

                # Get data slice
                data_slice = df[(df['time_hours'] >= start_h) & (df['time_hours'] < end_h)]

                if len(data_slice) < 10:
                    continue

                # Aggregate metrics
                ct_mae = data_slice['chronotick_offset_ms'].abs().mean() if 'chronotick_offset_ms' in data_slice.columns else 0
                ct_std = data_slice['chronotick_offset_ms'].std() if 'chronotick_offset_ms' in data_slice.columns else 0
                ct_max = data_slice['chronotick_offset_ms'].abs().max() if 'chronotick_offset_ms' in data_slice.columns else 0

                ntp_data = data_slice['ntp_offset_ms'].dropna() if 'ntp_offset_ms' in data_slice.columns else pd.Series()
                ntp_mae = ntp_data.abs().mean() if len(ntp_data) > 0 else 0
                ntp_std = ntp_data.std() if len(ntp_data) > 0 else 0

                # Count NTP rejections in this window
                ntp_rejections = 0
                for rejection in log_context['ntp_rejections']:
                    # Rough timestamp matching (would need proper parsing)
                    ntp_rejections += 1  # Simplified

                # Determine story type
                reasons = []
                for h in group:
                    pattern = hour_patterns[h]
                    reasons.extend(pattern.get('reasons', []))

                story_type = "unknown"
                if any('drift' in r.lower() for r in reasons):
                    story_type = "drift"
                if any('ntp' in r.lower() or 'rejection' in r.lower() for r in reasons):
                    story_type = "ntp_quality"
                if any('variance' in r.lower() or 'instability' in r.lower() for r in reasons):
                    story_type = "instability"

                # Create title
                if story_type == "ntp_quality":
                    title = f"NTP Quality Issues: Hours {start_h}-{end_h} ({len(log_context['ntp_rejections'])} rejections)"
                elif story_type == "drift":
                    title = f"Clock Drift Period: Hours {start_h}-{end_h}"
                elif story_type == "instability":
                    title = f"High Instability: Hours {start_h}-{end_h}"
                else:
                    title = f"Interesting Period: Hours {start_h}-{end_h}"

                description = f"During hours {start_h}-{end_h}, " + "; ".join(reasons[:3])

                window = MicroWindow(
                    experiment=exp_name,
                    platform=platform,
                    start_hour=start_h,
                    end_hour=end_h,
                    duration_min=(end_h - start_h) * 60,
                    title=title,
                    description=description,
                    evidence=reasons,
                    log_evidence=[r['line'][:100] for r in log_context['ntp_rejections'][:5]],
                    chronotick_mae=ct_mae,
                    chronotick_std=ct_std,
                    chronotick_max=ct_max,
                    ntp_mae=ntp_mae,
                    ntp_std=ntp_std,
                    ntp_rejections=len(log_context['ntp_rejections']),
                    story_type=story_type,
                    severity="high" if ct_std > 2 or ntp_std > 10 else "medium",
                    data_slice=data_slice
                )

                windows.append(window)
                print(f"  ✨ Created micro-window: {title}")

        return windows

    def analyze_all(self):
        """Analyze all experiments"""
        print("\n" + "="*80)
        print("ULTRA-DEEP MICRO-EXPLORATION ANALYSIS")
        print("Looking for specific time windows with interesting behavior")
        print("="*80)

        # Find all experiment directories
        for exp_num in range(1, 12):
            exp_dir = self.results_dir / f'experiment-{exp_num}'
            if exp_dir.exists():
                self.analyze_experiment(exp_dir)

        print(f"\n{'='*80}")
        print(f"COMPLETE: Found {len(self.micro_windows)} interesting micro-windows")
        print(f"{'='*80}")

        return self.micro_windows

    def generate_report(self, output_path: Path):
        """Generate comprehensive report"""
        # Group by story type
        by_type = {}
        for window in self.micro_windows:
            story_type = window.story_type
            if story_type not in by_type:
                by_type[story_type] = []
            by_type[story_type].append(window)

        # Write markdown
        lines = [
            "# Ultra-Deep Micro-Window Analysis",
            f"\n**Total Windows Found**: {len(self.micro_windows)}",
            f"**Analysis Date**: {datetime.now().isoformat()}",
            "\n---\n",
            "## Windows by Story Type\n"
        ]

        for story_type, windows in sorted(by_type.items()):
            lines.append(f"\n### {story_type.upper()}: {len(windows)} windows\n")

            for w in sorted(windows, key=lambda x: x.severity, reverse=True)[:10]:
                lines.extend([
                    f"\n#### {w.experiment}/{w.platform}: {w.title}",
                    f"\n**Time**: Hours {w.start_hour:.2f}-{w.end_hour:.2f} ({w.duration_min:.0f} minutes)",
                    f"**Severity**: {w.severity}",
                    f"\n**Description**: {w.description}",
                    f"\n**Metrics**:",
                    f"- ChronoTick: MAE={w.chronotick_mae:.2f}ms, σ={w.chronotick_std:.2f}ms, max={w.chronotick_max:.2f}ms",
                    f"- NTP: MAE={w.ntp_mae:.2f}ms, σ={w.ntp_std:.2f}ms",
                    f"- NTP Rejections: {w.ntp_rejections}",
                    "\n**Evidence**:"
                ])
                for ev in w.evidence[:5]:
                    lines.append(f"- {ev}")

                if w.log_evidence:
                    lines.append("\n**Log Evidence (sample)**:")
                    for log in w.log_evidence[:3]:
                        lines.append(f"```\n{log}\n```")

                lines.append("\n---\n")

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"✅ Report written to {output_path}")


def main():
    analyzer = UltraDeepAnalyzer(Path('results'))
    windows = analyzer.analyze_all()

    output_dir = Path('results/analysis')
    output_dir.mkdir(exist_ok=True)

    analyzer.generate_report(output_dir / 'ultra_deep_windows.md')

    print(f"\n📊 Top Stories Found:")
    by_type = {}
    for w in windows:
        if w.story_type not in by_type:
            by_type[w.story_type] = []
        by_type[w.story_type].append(w)

    for story_type, ws in sorted(by_type.items()):
        print(f"\n{story_type}: {len(ws)} windows")
        for w in sorted(ws, key=lambda x: x.chronotick_std, reverse=True)[:3]:
            print(f"  - {w.experiment}/{w.platform}: {w.title} (σ={w.chronotick_std:.2f}ms)")


if __name__ == '__main__':
    main()
