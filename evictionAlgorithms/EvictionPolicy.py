
from dataStructures.object import CacheObject
from dataStructures.cacheStructue import Cache
from abc import ABC, abstractmethod

class EvictionPolicy(ABC):
    def __init__(self, cache: Cache):
        self.cache = cache
    """Abstract base class for cache eviction policies"""
    @abstractmethod
    def getObject(self, obj: CacheObject) -> bool:
        """Process an object access, returns True if cache hit"""
        pass
    def admit(self, obj: CacheObject) -> bool:
        """Process an object access, returns True if cache hit"""
        pass
    
    @abstractmethod
    def evict(self, required_space: int) -> int:
        """Evict objects to make space, returns bytes freed"""
        pass