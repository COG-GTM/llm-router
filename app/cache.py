from collections import OrderedDict


class LRUCache:
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self._data: OrderedDict = OrderedDict()

    def get(self, key):
        value = self._data.get(key)
        if value is not None:
            self._data.move_to_end(key)
        return value

    def put(self, key, value) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)
