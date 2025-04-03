from enum import Enum
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from dataStructures.object import CacheObject

class CacheAlgorithm(Enum):
    LRU = 1
    LFU = 2
    FIFO = 3
    # LRB = 3
    # GLCACHE = 4


class CacheSimulator:
    def __init__(
        self, cache_size: int, algorithm: CacheAlgorithm, trace_data: pd.DataFrame, logIntvl: int, warmupNumReq: int
    ):
        """
        Initialize the cache simulator.

        Args:
            cache_size: Total size of the cache in arbitrary units
            algorithm: Cache replacement algorithm to use
            trace_data: DataFrame containing the trace data with columns in order: seqnum, objectid, size
        """
        self.cache_size = cache_size
        self.algorithm = algorithm
        self.logIntvl = logIntvl
        self.warmupNumReq = warmupNumReq # always a percentage
        # Rename columns if they don't have headers, applies for Wiki_Trace
        if len(trace_data.columns) == 4:
            trace_data.columns = ["seqnum", "objectid", "size", "random"]
        self.trace_data = trace_data

        if self.algorithm == CacheAlgorithm.LRU:
            from evictionAlgorithms.LRU.LRU_EvictionPolicy import LRU
            from evictionAlgorithms.LRU.LRU_Cache import LRUCache
            self.cache = LRUCache(cache_size)
            self.policy = LRU(self.cache)

        else:
            # format of slabNumber and contents in the slab itself. \
            # Slab number varies from min to max of the dataset

            self.cache: {slabNumber: int, contents: List[page]} = {}
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

    def _process_lru(self, obj: CacheObject):
        """Process an object using LRU policy."""
        hit = self.policy.getObject(obj)
        if hit:
            self.hits += 1
            return

        # Cache miss
        self.misses += 1  # includes cold miss
        self.policy.admit(obj)
        return


    def simulate(self):
        """Run the cache simulation on the trace data."""
        for _, row in self.trace_data.iterrows():
            self.numReq += 1
            if self.numReq % self.logIntvl == 0:
                print(self.get_results())

            obj = CacheObject(
                object_id=row["objectid"], size=row["size"], seq_number=row["seqnum"]
            )

            # Process based on selected algorithm
            if self.algorithm == CacheAlgorithm.LRU:
                self._process_lru(obj)
            elif self.algorithm == CacheAlgorithm.LFU:
                self._process_lfu(obj)
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")

            # Update statistics after each request
            if self.warmupNumReq != 0: # complete this
                self._update_stats()

    def get_results(self) -> Dict:
        """Get simulation results."""
        return {
            "total_requests": self.numReq,
            "hits": self.hits,
            "misses": self.misses,
            "final_hit_rate": self.hit_rates[-1],
            "final_miss_rate": self.miss_rates[-1],
            "cache_size": self.cache_size,
            "algorithm": self.algorithm.name,
        }

    def plot_results(self):
        """Plot hit/miss rates over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.hit_rates, label="Hit Rate")
        plt.plot(self.miss_rates, label="Miss Rate")
        plt.xlabel("Requests")
        plt.ylabel("Rate")
        plt.title(f"Cache Performance ({self.algorithm.name}, Size={self.cache_size})")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Load trace data (no headers)
    trace_data = pd.read_csv(
        "~/Projects/695/LRB/Wiki_Test.csv", 
        header=None, 
        delim_whitespace=True
    )

    print("Running LRU simulation...")
    lru_sim = CacheSimulator(
        cache_size=10 * 1024**3, 
        algorithm=CacheAlgorithm.LRU, 
        trace_data=trace_data,
        logIntvl = 1000**2,
        warmupNumReq = 20
    )
    lru_sim.simulate()
    print(lru_sim.get_results())
    lru_sim.plot_results()
