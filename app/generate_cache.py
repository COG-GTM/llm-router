from collections import OrderedDict


class ResponseCache:
    """Bounded LRU cache of rendered response bodies."""

    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self._data: OrderedDict[tuple, bytes] = OrderedDict()

    def get(self, key: tuple) -> bytes | None:
        body = self._data.get(key)
        if body is not None:
            self._data.move_to_end(key)
        return body

    def put(self, key: tuple, body: bytes) -> None:
        self._data[key] = body
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: tuple) -> bool:
        return key in self._data
