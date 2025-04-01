from enum import Enum
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from collections import OrderedDict

class CacheObject:
    def __init__(self, object_id: int, size: int, seq_number: int):
        self.object_id = int(object_id)
        self.size = int(size)
        self.seq_number = int(seq_number)
        self.last_accessed = time.time()  # For LRU
        self.access_count = 0             # For LFU
        self.age = 0                      # For potential aging algorithms
    def __str__(self):
        return f"CacheObject(id={self.object_id}, size={self.size}, seqnum={self.seq_number}, last_accessed={self.last_accessed}, access_count={self.access_count})"

class CacheAlgorithm(Enum):
    LRU = 1
    LFU = 2
    # LRB = 3
    # GLCACHE = 4

class CacheSimulator:
    def __init__(self, cache_size: int, algorithm: CacheAlgorithm, trace_data: pd.DataFrame):
        """
        Initialize the cache simulator.
        
        Args:
            cache_size: Total size of the cache in arbitrary units
            algorithm: Cache replacement algorithm to use
            trace_data: DataFrame containing the trace data with columns in order: seqnum, objectid, size
        """
        self.cache_size = cache_size
        self.algorithm = algorithm
        
        # Rename columns if they don't have headers
        if len(trace_data.columns) == 4:
            trace_data.columns = ['seqnum', 'objectid', 'size', "random"]
        self.trace_data = trace_data
        
        # Cache state
        self.cache: {int, CacheObject} = {}
        self.lru_order: OrderedDict = OrderedDict()
        self.current_size = 0
        
        # Statistics
        self.numReq = 0
        self.hits = 0
        self.misses = 0
        self.hit_rates = [0.0]
        self.miss_rates = [0.0]
        
    def _update_stats(self):
        """Update hit and miss rate statistics."""
        total = self.hits + self.misses
        if total > 0:
            hit_rate = self.hits / total
            miss_rate = self.misses / total
        else:
            hit_rate = 0.0
            miss_rate = 0.0
            
        self.hit_rates.append(hit_rate)
        self.miss_rates.append(miss_rate)
    
    def _lru_evict(self, required_space: int) -> int:
        """
        Evict objects using LRU policy until enough space is available.
        
        Args:
            required_space: Space needed for the new object
            
        Returns:
            Amount of space freed
        """
        freed_space = 0
        
        # We need both the cache dict and an OrderedDict to track LRU order
        # Initialize in __init__ as:
        # self.cache = {}  # {object_id: CacheObject}
        # self.lru_order = OrderedDict()  # {object_id: None}
        
        while (self.current_size + required_space) > self.cache_size and self.lru_order:
            # Get the least recently used item
            obj_id, _ = self.lru_order.popitem(last=False)
            
            if obj_id in self.cache:
                # Remove from cache and count freed space
                evicted_obj = self.cache.pop(obj_id)
                freed_space += evicted_obj.size
                self.current_size -= evicted_obj.size
        
        return freed_space
    
    def _lfu_evict(self, required_space: int) -> int:
        """
        Evict objects using LFU policy until enough space is available.
        
        Args:
            required_space: Space needed for the new object
            
        Returns:
            Amount of space freed
        """
        freed_space = 0
        
        # Sort cache by access count (least frequently used first)
        self.cache.sort(key=lambda x: x.access_count)
        
        # Evict least frequently used objects until we have enough space
        while self.cache and (self.current_size + required_space) > self.cache_size:
            evicted = self.cache.pop(0)
            freed_space += evicted.size
            self.current_size -= evicted.size
            
        return freed_space
    
    def _process_lru(self, obj: CacheObject):
        """Process an object using LRU policy."""
        # Check if object is in cache
        if obj.object_id in self.cache:
            # Cache hit - update last accessed time
            self.cache[obj.object_id].last_accessed = time.time()
            self.lru_order.move_to_end(obj.object_id)
            self.hits += 1
            return

        
        # Cache miss
        self.misses += 1 # includes cold miss
        if obj.size > self.cache_size:
            #print(f"Object was too big to fit in cache: {obj.__str__()}")
            return
        else:
            # Make space in cache
            if (self.current_size + obj.size) > self.cache_size and self.current_size != 0:
                self._lru_evict(obj.size)
            
            # Add to cache if there's enough space
            if (self.current_size + obj.size) <= self.cache_size:
                self.cache[obj.object_id] = obj
                self.lru_order[obj.object_id] = None  # Value doesn't matter
                self.current_size += obj.size
                # New objects go to the end (most recently used)
                self.lru_order.move_to_end(obj.object_id)

    
    def _process_lfu(self, obj: CacheObject):
        """Process an object using LFU policy."""
        # Check if object is in cache
        for cached_obj in self.cache:
            if cached_obj.object_id == obj.object_id:
                # Cache hit - increment access count
                cached_obj.access_count += 1
                self.hits += 1
                return
        
        # Cache miss
        self.misses += 1
        
        # Make space if needed
        if (self.current_size + obj.size) > self.cache_size:
            self._lfu_evict(obj.size)
        
        # Add to cache if there's enough space
        if (self.current_size + obj.size) <= self.cache_size:
            obj.access_count = 1  # Initial access
            self.cache.append(obj)
            self.current_size += obj.size
    
    def simulate(self):
        """Run the cache simulation on the trace data."""
        for _, row in self.trace_data.iterrows():
            self.numReq += 1
            if self.numReq % 1000000 == 0:
                print(self.get_results())
            
            obj = CacheObject(
                object_id=row['objectid'],
                size=row['size'],
                seq_number=row['seqnum']
            )
            
            # Process based on selected algorithm
            if self.algorithm == CacheAlgorithm.LRU:
                self._process_lru(obj)
            elif self.algorithm == CacheAlgorithm.LFU:
                self._process_lfu(obj)
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            
            # Update statistics after each request
            self._update_stats()
    
    def get_results(self) -> Dict:
        """Get simulation results."""
        return {
            'total_requests': self.numReq,
            'hits': self.hits,
            'misses': self.misses,
            'final_hit_rate': self.hit_rates[-1],
            'final_miss_rate': self.miss_rates[-1],
            'cache_size': self.cache_size,
            'algorithm': self.algorithm.name
        }
    
    def plot_results(self):
        """Plot hit/miss rates over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.hit_rates, label='Hit Rate')
        plt.plot(self.miss_rates, label='Miss Rate')
        plt.xlabel('Requests')
        plt.ylabel('Rate')
        plt.title(f'Cache Performance ({self.algorithm.name}, Size={self.cache_size})')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Load trace data (no headers)
    trace_data = pd.read_csv('~/Projects/695/LRB/Wiki_Test.csv', header=None, delim_whitespace=True)
    
    print("Running LRU simulation...")
    lru_sim = CacheSimulator(cache_size=10*1024**3, algorithm=CacheAlgorithm.LRU, trace_data=trace_data)
    lru_sim.simulate()
    print(lru_sim.get_results())
    lru_sim.plot_results()
