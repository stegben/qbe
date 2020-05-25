import bisect
from typing import Tuple


class RangeLookup:

    def __init__(self):
        self.positions = []
        self.labels = []

    def add(self, position: int, label: str):
        if self.positions:
            if position < self.positions[-1]:
                raise ValueError(f'Adding new element should be incremetal. Got {position}: {label}')
        self.positions.append(position)
        self.labels.append(label)

    def __getitem__(self, position: int) -> Tuple[str, int]:
        idx = bisect.bisect_left(self.positions, position)
        if idx >= 1:
            start = self.positions[idx - 1]
        else:
            start = 0
        return self.labels[idx], position - start
