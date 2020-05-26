import bisect
from typing import Tuple, Union


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

    def _get_idx(self, position):
        return bisect.bisect_left(self.positions, position)

    def __getitem__(self, position: Union[int, slice]) -> Tuple[str, int]:
        if isinstance(position, slice):  # slicing
            start = position.start
            stop = position.stop
            assert stop > start

            start_idx = self._get_idx(start)
            stop_idx = self._get_idx(stop)

            if start_idx >= 1:
                start_position = self.positions[start_idx - 1] + 1
            else:
                start_position = 0
            result = [(self.labels[start_idx], start - start_position)]
            for idx in range(start_idx, stop_idx):
                result.append((self.labels[idx], self.positions[idx] - start_position))
                result.append((self.labels[idx + 1], 0))
                start_position = self.positions[idx] + 1
            if stop > start_position:
                result.append((self.labels[stop_idx], stop - start_position))
            return result
        else:
            idx = self._get_idx(position)
            if idx >= 1:
                start_position = self.positions[idx - 1] + 1
            else:
                start_position = 0
            return self.labels[idx], position - start_position
