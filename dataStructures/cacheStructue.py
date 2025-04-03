from abc import ABC, abstractmethod
from typing import Optional
from dataStructures.object import CacheObject


class Cache(ABC):
    """Abstract base class for cache storage"""
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.current_size = 0
        
    @abstractmethod
    def get(self, obj_id: CacheObject) -> Optional[CacheObject]:
        pass
        
    @abstractmethod
    def put(self, obj: CacheObject) -> bool:
        pass
        
    @abstractmethod
    def evict(self, size: int) -> int:
        pass