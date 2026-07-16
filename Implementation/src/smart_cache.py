"""
Smart Zonotope Cache with Region Extension and Intelligent Merging

This module implements an advanced caching algorithm that:
1. Extends cache regions when new optimizations yield similar results
2. Creates new entries only when significantly better solutions are found
3. Intelligently merges entries while preserving the best objectives

Author: SADSE Team
Date: October 2025
"""

import sys

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import warnings
import time
from dataclasses import dataclass
import gc
import tracemalloc

from .zonotope_ops import Zonotope, zonotope_union, zonotope_subtract
from .domain import DomainSpec, ConstrainedDomain, Configuration, ODD

def condition_for_coverage(odd_entry, odd, threshold = 0):
        #return hash(odd_entry) == hash(odd) 
        #print("Checking coverage between ODDs with threshold:", threshold)
        #print("ODD Entry:", odd_entry)
        #print("ODD New:", odd)
        return odd_entry.contains(odd, threshold) or odd.contains(odd_entry,threshold) 




@dataclass
class SmartCacheEntry:
    """
    Enhanced cache entry with tracking for region extensions.
    
    Attributes:
        odd_hash: Hash of the ODD conditions
        ds_zonotope: Design space region (can grow via extension)
        optimal_config: Best configuration found
        optimal_objective: Best objective value
        metadata: Additional information
        extension_count: Number of times this region was extended
        timestamp: Creation time
    """
    odd: ODD
    ds_zonotope: Zonotope
    optimal_config: Configuration
    optimal_objective: float
    metadata: Dict[str, Any]
    extension_count: int = 0
    timestamp: float = 0.0


class SmartZonotopeCache:
    r"""
    Advanced zonotope cache with intelligent region extension and merging.
    
    Algorithm:
    ----------
    Given new DS A and existing cache C:
    
    1. Compute unexplored = A \ C (set difference)
    2. If unexplored is empty: CACHE HIT (return cached result)
    3. If unexplored is non-empty:
       a. Optimize in unexplored regions ONLY
       b. Compare: objective(unexplored) vs objective(C_overlapping)
       c. If objective(unexplored) >= objective(C):
          → EXTEND C to include unexplored (same solution good enough)
       d. If objective(unexplored) < objective(C):
          → CREATE new entry, try MERGE with compatible entries
    
    Key Advantages:
    ---------------
    - Only optimizes in unexplored regions (much faster!)
    - Automatically extends regions when appropriate (fewer entries)
    - Keeps best solution when merging (optimal coverage)
    - Mathematically sound (conservative approximations)
    """
    
    def __init__(
        self,
        domain_spec: DomainSpec,
        extension_threshold: float = 0.05,  # 5% worse is acceptable for extension
        merge_threshold: float = 0.10,       # 10% difference for merging
        merge_frequency: int = 5,
        strategy_name: str = "SmartZonotopeCache"
    ):
        """
        Initialize smart cache.
        
        Args:
            domain_spec: Domain specification
            extension_threshold: Max relative objective difference for extension (0.05 = 5%)
            merge_threshold: Max relative objective difference for merging
            merge_frequency: Merge every N additions
            strategy_name: Name of the strategy using this cache
        """
        self.domain_spec = domain_spec
        self.dimension = domain_spec.dimension
        self.strategy_name = strategy_name
        
        # Configuration
        self.extension_threshold = extension_threshold
        self.merge_threshold = merge_threshold
        self.merge_frequency = merge_frequency
        
        # Storage
        self.entries: List[SmartCacheEntry] = []
        
        # Statistics
        self.total_queries = 0
        self.cache_hits = 0
        self.extensions = 0
        self.merges = 0
        self.total_additions = 0
        self._adds_since_merge = 0
        
        # Memory tracking
        from .memory_utils import create_memory_tracker
        self.memory_tracker = create_memory_tracker(self.strategy_name)
        
        # Performance tracking
        self.query_times: List[float] = []
        self.optimization_times: List[float] = []
        
        # Coverage tracking
        self.explored_volume = 0.0  # Track cumulative explored volume
    
    def _update_explored_volume(self):
        """Recalculate total explored volume from all cache entries."""
        if not self.entries:
            self.explored_volume = 0.0
            return
        
        # For simplicity, sum all entry volumes (this is an upper bound)
        # A more accurate calculation would use zonotope_union, but that's expensive
        #self.explored_volume = sum(entry.ds_zonotope.volume() for entry in self.entries)

         # find geometric volume by union of all zonotopes
        all_zonotopes = [entry.ds_zonotope for entry in self.entries]
        union_zonotope = all_zonotopes[0]
        for zono in all_zonotopes[1:]:
            union_zonotope = zonotope_union(union_zonotope, zono)
        
        self.explored_volume = union_zonotope.volume()
    

    def query_and_optimize(
        self,
        odd: ODD,
        ds: ConstrainedDomain,
        optimization_function: callable
    ) -> Tuple[Configuration, float, Dict[str, Any], str]:
        """
        Query cache and optimize if needed (implements the smart algorithm).
        
        This is the MAIN METHOD that implements your algorithm!
        
        Args:
            odd: Operational Design Domain
            ds: Design space A (new DS to query)
            optimization_function: Function(unexplored_zonotope) -> (config, objective, metadata)
        
        Returns:
            (optimal_config, optimal_objective, metadata, status)
            where status is: 'hit', 'extended', 'new_entry'
        """
        self.total_queries += 1
        start_time = time.time()
        
        # Sample memory at start of query
        self.memory_tracker.snapshot(f"query_{self.total_queries}", len(self.entries))
        
        if ds.is_empty():
            warnings.warn("Empty DS provided to query_and_optimize")
            return None, float('inf'), {}, 'error'
        
        # Convert DS to zonotope
        ds_zonotope_A = Zonotope.from_box(ds.to_bounds_list())
        odd_hash = hash(odd)
        
        # Find overlapping entries with same ODD
        overlapping_entries = self._find_overlapping_entries(odd, ds_zonotope_A)
        
        # DEBUG: Print diagnostic information
        #print(f"DEBUG: Query #{self.total_queries}")
        #print(f"DEBUG: Current ODD: {odd}")
        #print(f"DEBUG: Current ODD hash: {odd_hash}")
        #print(f"DEBUG: Total cache entries: {len(self.entries)}")
        #print(f"DEBUG: Found {len(overlapping_entries)} overlapping entries")
        #if self.entries:
        #    print(f"DEBUG: Sample entry ODD hash: {hash(self.entries[0].odd)}")
        #    print(f"DEBUG: ODD hash match with first entry: {hash(self.entries[0].odd) == odd_hash}")
        #    # Show first few entries and their hashes
        #    for i, entry in enumerate(self.entries[:3]):
        #        print(f"DEBUG: Entry {i}: hash={hash(entry.odd)}, match={hash(entry.odd) == odd_hash}")
        #print(f"DEBUG: DS volume: {ds_zonotope_A.volume():.2e}")
        
        # STEP 1: Compute unexplored regions (A \ C)
        unexplored_regions = self._compute_unexplored(ds_zonotope_A, overlapping_entries)
        
        query_time = time.time() - start_time
        self.query_times.append(query_time * 1000)
        
        # STEP 2: Check if fully explored (CACHE HIT)
        if not unexplored_regions or all(r.is_empty() for r in unexplored_regions):
            self.cache_hits += 1
            # Return best result from overlapping entries
            best_entry = min(overlapping_entries, key=lambda e: e.optimal_objective)
            return (best_entry.optimal_config, best_entry.optimal_objective, 
                    best_entry.metadata, 'hit')
        
        # STEP 3: Optimize in unexplored regions ONLY
        opt_start = time.time()
        unexplored_config, unexplored_objective, unexplored_metadata = \
            self._optimize_in_regions(unexplored_regions, optimization_function)
        opt_time = time.time() - opt_start
        self.optimization_times.append(opt_time * 1000)
        
        if unexplored_config is None:
            # No feasible solution in unexplored
            return None, float('inf'), {}, 'infeasible'
        
        # STEP 4: Decide whether to EXTEND or CREATE NEW ENTRY
        if overlapping_entries:
            #print("Found overlapping entries, comparing objectives...")
            best_existing_entry = min(overlapping_entries, key=lambda e: e.optimal_objective)
            existing_objective = best_existing_entry.optimal_objective
            
            # Check if unexplored solution is similar enough to extend
            relative_diff = abs(unexplored_objective - existing_objective) / max(abs(existing_objective), 1.0)
            
            if relative_diff <= self.extension_threshold or self.extension_threshold < 0:
                # EXTEND: Unexplored is not significantly worse → extend existing region
                if unexplored_objective < existing_objective:
                    # Update to better solution
                    best_existing_entry.optimal_config = unexplored_config
                    best_existing_entry.optimal_objective = unexplored_objective
                    best_existing_entry.metadata = unexplored_metadata
                #print("EXTEND: Extending existing cache entry to include unexplored region.")
                self._extend_entry(best_existing_entry, unexplored_regions)
                return (best_existing_entry.optimal_config, best_existing_entry.optimal_objective,
                        best_existing_entry.metadata, 'extended')
            else:
                # CREATE NEW: Unexplored has significantly different objective
                print(f" No EXTEND: Creating new entry due to significant objective improvement: {relative_diff:.2f}.")
                new_entry = self._create_and_add_entry(
                    odd, unexplored_regions, unexplored_config, 
                    unexplored_objective, unexplored_metadata
                )
                return (unexplored_config, unexplored_objective, unexplored_metadata, 'new_entry')
        else:
            # No overlapping entries → create new entry
            new_entry = self._create_and_add_entry(
                odd, unexplored_regions, unexplored_config,
                unexplored_objective, unexplored_metadata
            )
            return (unexplored_config, unexplored_objective, unexplored_metadata, 'new_entry')
    
    def _find_overlapping_entries(
        self,
        odd: ODD,
        ds_zonotope: Zonotope
    ) -> List[SmartCacheEntry]:
        """Find all entries that overlap with DS and have same ODD."""
        overlapping = []
        ds_bounds = ds_zonotope.to_box_bounds()
        
        same_odd_count = 0
        bounds_overlap_count = 0
        odd_hash = hash(odd)
        for entry in self.entries:
            # Check if ODD is contained within a bigger ODD
            #if entry.odd_hash != odd_hash:
            #    continue
            #if entry.odd.contains(odd) is False or odd.contains(entry.odd) is False:
            #    continue
            if condition_for_coverage(entry.odd, odd)is False:
                continue
            same_odd_count += 1
            
            # Check if regions overlap (using AABB)
            entry_bounds = entry.ds_zonotope.to_box_bounds()
            if self._bounds_overlap(ds_bounds, entry_bounds):
                bounds_overlap_count += 1
                overlapping.append(entry)
        
        #print(f"DEBUG: Entries with same ODD: {same_odd_count}")
        #print(f"DEBUG: Entries with bounds overlap: {bounds_overlap_count}")
        
        return overlapping
    
    def _bounds_overlap(self, bounds1: np.ndarray, bounds2: np.ndarray) -> bool:
        """Check if two bounding boxes overlap."""
        # No overlap if separated in any dimension
        for i in range(bounds1.shape[0]):
            if bounds1[i, 1] < bounds2[i, 0] or bounds2[i, 1] < bounds1[i, 0]:
                return False
        return True
    
    def _compute_unexplored(
        self,
        ds_zonotope: Zonotope,
        overlapping_entries: List[SmartCacheEntry]
    ) -> List[Zonotope]:
        r"""Compute unexplored regions: DS \ union(overlapping_entries)."""
        if not overlapping_entries:
            return [ds_zonotope]
        
        # Start with full DS
        unexplored = [ds_zonotope]
        #print("overlapping_entries:", overlapping_entries)
        # Subtract each overlapping entry's region
        for entry in overlapping_entries:
            new_unexplored = []
            for region in unexplored:
                fragments = zonotope_subtract(region, entry.ds_zonotope)
                new_unexplored.extend(fragments)
            unexplored = new_unexplored
            
            if not unexplored:
                break  # Fully explored
        #print("unexplored:", unexplored)
        return unexplored
    
    def _optimize_in_regions(
        self,
        regions: List[Zonotope],
        optimization_function: callable
    ) -> Tuple[Optional[Configuration], float, Dict[str, Any]]:
        """
        Optimize over all unexplored regions and return best result.
        
        Args:
            regions: List of zonotope regions to optimize over
            optimization_function: Function(zonotope) -> (config, objective, metadata)
        
        Returns:
            Best configuration, objective, and metadata across all regions
        """
        best_config = None
        best_objective = float('inf')
        best_metadata = {}
        
        for region in regions:
            if region.is_empty():
                continue
            
            try:
                config, objective, metadata = optimization_function(region)
                
                if config is not None and objective < best_objective:
                    best_config = config
                    best_objective = objective
                    best_metadata = metadata
            except Exception as e:
                warnings.warn(f"Optimization failed in region: {e}")
                continue
        
        return best_config, best_objective, best_metadata
    
    def _extend_entry(
        self,
        entry: SmartCacheEntry,
        unexplored_regions: List[Zonotope]
    ):
        """
        Extend existing cache entry to cover unexplored regions.
        
        This grows the DS zonotope to include the unexplored areas.
        """
        # Union entry's zonotope with all unexplored regions
        extended_zonotope = entry.ds_zonotope
        
        for region in unexplored_regions:
            if not region.is_empty():
                extended_zonotope = zonotope_union(extended_zonotope, region)
        
        entry.ds_zonotope = extended_zonotope
        entry.extension_count += 1
        self.extensions += 1
        self._update_explored_volume()  # Update coverage after extension
    
    def _create_and_add_entry(
        self,
        odd: ODD,
        regions: List[Zonotope],
        config: Configuration,
        objective: float,
        metadata: Dict[str, Any]
    ) -> SmartCacheEntry:
        """Create new cache entry from unexplored regions."""
        # Union all regions into one zonotope
        combined_zonotope = regions[0]
        for region in regions[1:]:
            if not region.is_empty():
                combined_zonotope = zonotope_union(combined_zonotope, region)
        
        # Create entry
        new_entry = SmartCacheEntry(
            odd=odd,
            ds_zonotope=combined_zonotope,
            optimal_config=config,
            optimal_objective=objective,
            metadata=metadata,
            extension_count=0,
            timestamp=time.time()
        )
        
        # Try to merge with existing compatible entries
        merged = self._try_merge_with_existing(new_entry)
        
        if not merged:
            self.entries.append(new_entry)
        
        self.total_additions += 1
        self._adds_since_merge += 1
        self._update_explored_volume()  # Update coverage after adding
        
        # Periodic consolidation
        if self._adds_since_merge >= self.merge_frequency:
            self._consolidate()
            self._adds_since_merge = 0
        
        return new_entry
    
    def _try_merge_with_existing(self, new_entry: SmartCacheEntry) -> bool:
        """
        Try to merge new entry with existing compatible entries.
        
        Keeps the MINIMUM (best) objective value when merging.
        """
        for i, existing in enumerate(self.entries):
            # Same ODD?
            #if hash(existing.odd) != hash(new_entry.odd):
            #    continue
            if condition_for_coverage(existing.odd, new_entry.odd) is False:
                continue
            # Check objective compatibility
            obj_diff = abs(existing.optimal_objective - new_entry.optimal_objective)
            min_obj = min(existing.optimal_objective, new_entry.optimal_objective)
            relative_diff = obj_diff / max(abs(min_obj), 1.0)
            
            if relative_diff > self.merge_threshold:
                continue
            
            # Try geometric merge
            merged_zonotope = zonotope_union(existing.ds_zonotope, new_entry.ds_zonotope)
            
            # Check if merge is reasonable (not too much over-approximation)
            vol_existing = existing.ds_zonotope.volume()
            vol_new = new_entry.ds_zonotope.volume()
            vol_merged = merged_zonotope.volume()
            
            if True:
                # MERGE: Keep the BEST (minimum) objective
                if new_entry.optimal_objective < existing.optimal_objective:
                    # New entry is better
                    self.entries[i] = SmartCacheEntry(
                        odd=new_entry.odd,
                        ds_zonotope=merged_zonotope,
                        optimal_config=new_entry.optimal_config,
                        optimal_objective=new_entry.optimal_objective,
                        metadata=new_entry.metadata,
                        extension_count=existing.extension_count + 1,
                        timestamp=new_entry.timestamp
                    )
                else:
                    # Existing is better or equal
                    self.entries[i] = SmartCacheEntry(
                        odd=existing.odd,
                        ds_zonotope=merged_zonotope,
                        optimal_config=existing.optimal_config,
                        optimal_objective=existing.optimal_objective,
                        metadata=existing.metadata,
                        extension_count=existing.extension_count + 1,
                        timestamp=existing.timestamp
                    )
                
                self.merges += 1
                return True
        
        return False
    
    def _consolidate(self):
        """Periodically try to merge compatible entries."""
        if len(self.entries) < 2:
            return
        
        i = 0
        while i < len(self.entries):
            j = i + 1
            while j < len(self.entries):
                if self._try_merge_entries(i, j):
                    # Entry j was merged into i, don't increment j
                    pass
                else:
                    j += 1
            i += 1
    
    def _try_merge_entries(self, i: int, j: int) -> bool:
        """Try to merge entries at indices i and j. Returns True if merged."""
        if i >= len(self.entries) or j >= len(self.entries):
            return False
        
        entry_i = self.entries[i]
        entry_j = self.entries[j]
        
        # Same ODD?
        #if hash(entry_i.odd) != hash(entry_j.odd):
        #    return False
        if entry_i.odd.contains(entry_j.odd) is False or entry_j.odd.contains(entry_i.odd) is False:
            return False

        # Check objective compatibility
        obj_diff = abs(entry_i.optimal_objective - entry_j.optimal_objective)
        min_obj = min(entry_i.optimal_objective, entry_j.optimal_objective)
        relative_diff = obj_diff / max(abs(min_obj), 1.0)
        if relative_diff > self.merge_threshold and self.merge_threshold > 0:
            return False
        
        # Try geometric merge
        merged_zonotope = zonotope_union(entry_i.ds_zonotope, entry_j.ds_zonotope)
        vol_i = entry_i.ds_zonotope.volume()
        vol_j = entry_j.ds_zonotope.volume()
        vol_merged = merged_zonotope.volume()
        
        if vol_merged <= 2.5 * (vol_i + vol_j):
            # Keep the better (minimum) objective
            if entry_j.optimal_objective < entry_i.optimal_objective:
                self.entries[i] = SmartCacheEntry(
                    odd=entry_j.odd,
                    ds_zonotope=merged_zonotope,
                    optimal_config=entry_j.optimal_config,
                    optimal_objective=entry_j.optimal_objective,
                    metadata=entry_j.metadata,
                    extension_count=entry_i.extension_count + entry_j.extension_count + 1,
                    timestamp=entry_j.timestamp
                )
            else:
                self.entries[i] = SmartCacheEntry(
                    odd=entry_i.odd,
                    ds_zonotope=merged_zonotope,
                    optimal_config=entry_i.optimal_config,
                    optimal_objective=entry_i.optimal_objective,
                    metadata=entry_i.metadata,
                    extension_count=entry_i.extension_count + entry_j.extension_count + 1,
                    timestamp=entry_i.timestamp
                )
            
            # Remove entry j
            del self.entries[j]
            self.merges += 1
            return True
        
        return False
    
    def print_summary(self):
        print(self.__str__())

    def __str__(self):
        """ Return as string"""
        ret = ""
        ret += "=" * 70 + "\n"
        ret += f"SMART ZONOTOPE CACHE SUMMARY ({self.dimension}D)\n"
        ret += "=" * 70 + "\n"
        ret += f"Domain: {', '.join(self.domain_spec.parameter_names)}\n"
        ret += "\n"
        ret += f"Cache Entries:        {len(self.entries)}\n"
        ret += f"Total Queries:        {self.total_queries}\n"
        ret += f"Cache Hits:           {self.cache_hits}\n"
        ret += f"Region Extensions:    {self.extensions}\n"
        ret += f"Hit Rate:             {100*self.cache_hits/max(self.total_queries,1):.1f}%\n"
        ret += f"Extension Rate:       {100*self.extensions/max(self.total_queries-self.cache_hits,1):.1f}%\n"
        ret += f"Successful Merges:    {self.merges}\n"
        ret += "\n"
        if self.query_times:
            ret += f"Avg Query Time:       {np.mean(self.query_times):.2f} ms\n"
        if self.optimization_times:
            ret += f"Avg Optimization:     {np.mean(self.optimization_times):.2f} ms\n"
        ret += "=" * 70 + "\n"
        return ret
    
    def __len__(self):
        return len(self.entries)
    
    def __repr__(self):
        hit_rate = 100 * self.cache_hits / max(self.total_queries, 1)
        return f"SmartZonotopeCache({self.dimension}D, {len(self)} entries, {hit_rate:.1f}% hit rate, {self.extensions} extensions)"
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics for this cache."""
        # Force garbage collection before measuring
        gc.collect()
        
        # Take a memory snapshot
        snapshot = self.memory_tracker.snapshot("current_usage", len(self.entries))
        
        # Calculate cache-specific memory more accurately (in KB)
        cache_memory_kb = 0.0
        for entry in self.entries:
            # Calculate deep size of each cache entry
            entry_size = sys.getsizeof(entry) / 1024  # Base entry object
            
            # Add zonotope memory (matrix data)
            if entry.ds_zonotope and hasattr(entry.ds_zonotope, 'center'):
                entry_size += sys.getsizeof(entry.ds_zonotope.center) / 1024
                if hasattr(entry.ds_zonotope, 'generators'):
                    entry_size += sys.getsizeof(entry.ds_zonotope.generators) / 1024
            
            # Add config memory (handle both dict and Configuration object)
            if entry.optimal_config:
                entry_size += sys.getsizeof(entry.optimal_config) / 1024
                if hasattr(entry.optimal_config, 'items'):
                    # Dictionary-like object
                    for key, value in entry.optimal_config.items():
                        entry_size += (sys.getsizeof(key) + sys.getsizeof(value)) / 1024
                elif hasattr(entry.optimal_config, '__dict__'):
                    # Configuration object with attributes
                    for key, value in entry.optimal_config.__dict__.items():
                        entry_size += (sys.getsizeof(key) + sys.getsizeof(value)) / 1024
            
            # Add metadata memory
            if entry.metadata:
                entry_size += sys.getsizeof(entry.metadata) / 1024
                
            cache_memory_kb += entry_size
        
        return {
            'strategy': self.strategy_name,
            'current_rss_kb': snapshot.rss_kb,
            'current_vms_kb': snapshot.vms_kb,
            'current_python_kb': snapshot.python_kb,
            'cache_data_kb': cache_memory_kb,
            'entries_count': len(self.entries),
            'avg_entry_size_kb': cache_memory_kb / max(len(self.entries), 1),
            'peak_memory': self.memory_tracker.get_peak_memory(),
            'memory_growth': self.memory_tracker.get_memory_growth()
        }
    
    def get_memory_summary(self) -> str:
        """Get formatted memory usage summary for this cache."""
        stats = self.get_memory_usage()
        peak = self.memory_tracker.get_peak_memory()
        
        summary = f"Memory Usage Summary - {self.strategy_name}\n"
        summary += "=" * 50 + "\n"
        
        if self.memory_tracker.has_psutil:
            summary += f"Current RSS: {stats['current_rss_kb']:.1f} KB\n"
            summary += f"Current VMS: {stats['current_vms_kb']:.1f} KB\n"
            if peak:
                summary += f"Peak RSS: {peak.rss_kb:.1f} KB\n"
        
        summary += f"Current Python: {stats['current_python_kb']:.1f} KB\n"
        if peak:
            summary += f"Peak Python: {peak.python_kb:.1f} KB\n"
        
        summary += f"Memory growth: {stats['memory_growth']:+.1f} KB\n"
        summary += f"Cache data: {stats['cache_data_kb']:.1f} KB\n"
        summary += f"Cache entries: {stats['entries_count']}\n"
        summary += f"Avg entry size: {stats['avg_entry_size_kb']:.2f} KB\n"
        summary += f"Snapshots taken: {len(self.memory_tracker.snapshots)}\n"
        
        return summary
    
    def sample_memory(self, label: str = ""):
        """Take a memory sample with custom label."""
        return self.memory_tracker.snapshot(label, len(self.entries))
