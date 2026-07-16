"""
Memory tracking utilities for comparing strategy performance.

This module provides tools to track and compare memory usage across different
optimization strategies (baseline vs smart cache approaches).
"""

import sys
import gc
import tracemalloc
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class MemorySnapshot:
    """Single memory measurement snapshot."""
    timestamp: float
    label: str
    strategy_name: str
    rss_kb: float = 0.0       # Resident Set Size in KB (if psutil available)
    vms_kb: float = 0.0       # Virtual Memory Size in KB (if psutil available)
    python_kb: float = 0.0    # Python objects memory in KB
    entries_count: int = 0    # Number of cache entries (for caches)
    custom_data: Dict[str, Any] = None


class StrategyMemoryTracker:
    """
    Memory tracker for individual optimization strategies.
    
    Tracks memory usage throughout strategy execution and provides
    comparison utilities.
    """
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.snapshots: List[MemorySnapshot] = []
        self.has_psutil = False
        self.process = None
        
        # Try to import psutil for system memory tracking
        try:
            import psutil
            self.process = psutil.Process()
            self.has_psutil = True
        except ImportError:
            pass
        
        # Start Python memory tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        # Take initial snapshot
        self.snapshot("initialization")
    
    def snapshot(self, label: str = "", entries_count: int = 0, custom_data: Dict[str, Any] = None) -> MemorySnapshot:
        """Take a memory snapshot at current point."""
        # Get Python memory usage in KB
        current_python, _ = tracemalloc.get_traced_memory()
        python_kb = current_python / 1024
        
        # Get system memory if available (convert to KB)
        rss_kb = vms_kb = 0.0
        if self.has_psutil:
            memory_info = self.process.memory_info()
            rss_kb = memory_info.rss / 1024
            vms_kb = memory_info.vms / 1024
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            label=label,
            strategy_name=self.strategy_name,
            rss_kb=rss_kb,
            vms_kb=vms_kb,
            python_kb=python_kb,
            entries_count=entries_count,
            custom_data=custom_data or {}
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_peak_memory(self) -> MemorySnapshot:
        """Get snapshot with peak memory usage."""
        if not self.snapshots:
            return None
        
        # Use RSS if available, otherwise Python memory
        key = 'rss_kb' if self.has_psutil else 'python_kb'
        return max(self.snapshots, key=lambda s: getattr(s, key))
    
    def get_final_memory(self) -> MemorySnapshot:
        """Get final memory snapshot."""
        return self.snapshots[-1] if self.snapshots else None
    
    def get_memory_growth(self) -> float:
        """Get memory growth from start to end (in KB)."""
        if len(self.snapshots) < 2:
            return 0.0
        
        start = self.snapshots[0]
        end = self.snapshots[-1]
        
        # Use RSS if available, otherwise Python memory
        if self.has_psutil:
            return end.rss_kb - start.rss_kb
        else:
            return end.python_kb - start.python_kb
    
    def export_snapshots(self, filepath: Path):
        """Export all snapshots to JSON file."""
        data = {
            'strategy_name': self.strategy_name,
            'has_psutil': self.has_psutil,
            'snapshots': [asdict(snapshot) for snapshot in self.snapshots]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class MemoryComparator:
    """
    Utility to compare memory usage across multiple strategies.
    """
    
    def __init__(self):
        self.strategies: Dict[str, Dict] = {}
    
    def add_strategy(self, name: str, memory_data: Dict):
        """Add memory data for a strategy."""
        self.strategies[name] = memory_data.copy()
    
    def add_tracker(self, tracker: StrategyMemoryTracker):
        """Add a strategy tracker to comparison."""
        final_snapshot = tracker.get_final_memory()
        peak = tracker.get_peak_memory()
        
        memory_data = {
            'strategy': tracker.strategy_name,
            'current_rss_kb': final_snapshot.rss_kb if final_snapshot and tracker.has_psutil else 0,
            'current_vms_kb': final_snapshot.vms_kb if final_snapshot and tracker.has_psutil else 0,
            'current_python_kb': final_snapshot.python_kb if final_snapshot else 0,
            'cache_data_kb': 0.0,
            'entries_count': final_snapshot.entries_count if final_snapshot else 0,
            'avg_entry_size_kb': 0.0,
            'peak_memory': peak,
            'memory_growth': tracker.get_memory_growth()
        }
        
        self.add_strategy(tracker.strategy_name, memory_data)
    
    def create_tracker(self, strategy_name: str) -> StrategyMemoryTracker:
        """Create and add new tracker for strategy."""
        tracker = StrategyMemoryTracker(strategy_name)
        self.add_tracker(tracker)
        return tracker
    
    def get_all_data(self) -> Dict:
        """Get all memory data for export."""
        return {
            'strategies': self.strategies,
            'comparison_summary': {
                'total_strategies': len(self.strategies),
                'has_system_memory': any(
                    'current_rss_kb' in data and data['current_rss_kb'] > 0 
                    for data in self.strategies.values()
                )
            }
        }
    
    def get_comparison_summary(self) -> str:
        """Get formatted comparison summary."""
        if not self.strategies:
            return "No strategies tracked."
        
        summary = "MEMORY USAGE COMPARISON\n"
        summary += "=" * 60 + "\n\n"
        
        # Strategy comparison table
        for strategy_name, data in self.strategies.items():
            summary += f"{strategy_name}:\n"
            summary += f"  Current Python: {data.get('current_python_kb', 0):.1f} KB\n"
            
            if data.get('current_rss_kb', 0) > 0:
                summary += f"  Current RSS: {data['current_rss_kb']:.1f} KB\n"
                summary += f"  Current VMS: {data['current_vms_kb']:.1f} KB\n"
            
            peak = data.get('peak_memory')
            if peak:
                summary += f"  Peak Python: {peak.python_kb:.1f} KB\n"
                if hasattr(peak, 'rss_kb') and peak.rss_kb > 0:
                    summary += f"  Peak RSS: {peak.rss_kb:.1f} KB\n"
            
            summary += f"  Memory growth: {data.get('memory_growth', 0):+.1f} KB\n"
            summary += f"  Cache entries: {data.get('entries_count', 0)}\n"
            
            if data.get('cache_data_kb', 0) > 0:
                summary += f"  Cache data: {data['cache_data_kb']:.1f} KB\n"
                summary += f"  Avg entry size: {data['avg_entry_size_kb']:.2f} KB\n"
            
            summary += "\n"
        # Comparison metrics between strategies (smart cache vs baseline)
        if len(self.strategies) >= 2:
            strategies_list = list(self.strategies.items())
            summary += "RELATIVE COMPARISONS:\n"
            summary += "-" * 30 + "\n"
            
            # Compare pairs (assuming smart cache vs baseline pattern)
            for i in range(0, len(strategies_list), 2):
                if i + 1 < len(strategies_list):
                    smart_name, smart_data = strategies_list[i]
                    baseline_name, baseline_data = strategies_list[i + 1]
                    
                    # Use RSS if available, otherwise Python memory
                    smart_memory = smart_data.get('current_rss_kb', 0) or smart_data.get('current_python_kb', 0)
                    baseline_memory = baseline_data.get('current_rss_kb', 0) or baseline_data.get('current_python_kb', 0)
                    
                    if baseline_memory > 0:
                        ratio = smart_memory / baseline_memory
                        savings = baseline_memory - smart_memory
                        efficiency = (1 - ratio) * 100
                        
                        summary += f"{smart_name} vs {baseline_name}:\n"
                        summary += f"  Memory ratio: {ratio:.2f}x\n"
                        summary += f"  Memory difference: {savings:+.1f} KB\n"
                        summary += f"  Efficiency: {efficiency:+.1f}%\n\n"
        
        return summary
    
    def export_to_json(self, filepath: str):
        """Export comparison data to JSON file."""
        export_data = {
            'timestamp': time.time(),
            **self.get_all_data()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    



# Utility function for easy integration
def create_memory_tracker(strategy_name: str) -> StrategyMemoryTracker:
    """Create a memory tracker for a strategy (convenience function)."""
    return StrategyMemoryTracker(strategy_name)