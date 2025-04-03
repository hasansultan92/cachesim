
from dataStructures.object import CacheObject
from evictionAlgorithms.EvictionPolicy import EvictionPolicy


class LRU(EvictionPolicy):
    def getObject(self, obj: CacheObject):
        """ Returns whether there is a hit or not """
        returnVal = self.cache.get(obj)
        if returnVal:
            return True
        else:
            return False

    def admit(self, obj: CacheObject):

        if obj.size > self.cache.cache_size:
            print(obj.__str__())
            return False  # Object too large to fit in cache
        else:
            if (self.cache.current_size + obj.size ) > self.cache.cache_size:
                freeSpace = self.cache.evict(obj.size)
                self.cache.put(obj)
                return True
            else:
                self.cache.put(obj)
                return True
    
    def evict(self, required_space):
        pass