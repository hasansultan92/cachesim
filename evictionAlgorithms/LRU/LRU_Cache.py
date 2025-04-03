from collections import OrderedDict
from typing import Dict, Optional
from dataStructures.object import CacheObject
from dataStructures.cacheStructue import Cache

class LRUCache(Cache):
    """Cache with built-in LRU tracking"""
    def __init__(self, cache_size: int):
        super().__init__(cache_size)
        self.storage: Dict[int, CacheObject] = {}
        self.access_order = OrderedDict()
        
    def get(self, obj: CacheObject) -> Optional[CacheObject]:
        if obj.object_id in self.storage:
            self.access_order.move_to_end(obj.object_id)
            return self.storage[obj.object_id]
        return None
        
    def put(self, obj: CacheObject) -> bool:
        if obj.size > self.cache_size:
            return False
        if obj.object_id not in self.storage:
            self.storage[obj.object_id] = obj
            self.current_size += obj.size
        self.access_order[obj.object_id] = None
        self.access_order.move_to_end(obj.object_id)
        return True
        
    def evict(self, size: int) -> int:
        freed = 0
        while self.current_size + size > self.cache_size and self.access_order:
            obj_id, _ = self.access_order.popitem(last=False)
            if obj_id in self.storage:
                freed += self.storage[obj_id].size
                self.current_size -= freed
                del self.storage[obj_id]
        return freed