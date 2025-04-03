
import time

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