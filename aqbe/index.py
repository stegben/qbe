from collections import Counter, defaultdict

from hnswlib import Index

from .types import AudioFeatureType


class HoughAccumulations:

    def __init__(self):
        self.counts = Counter()
        self.key2labels = defaultdict(list)

    def add(self, slope, offset, label):
        key = self.hash(slope, offset)
        self.counts.update([key])
        self.key2labels[key].append(label)

    def peaks(self, k):
        candidates = self.counts.most_common(k)
        result = []
        for key, count in candidates:
            result.append((key, count, self.key2labels[key]))
        return result

    def hash(self, x, y):
        return (int(x), int(y))


class SimpleRails:

    def __init__(self, total_frames, n_nearest_frames=100, n_hough_peaks=100):
        self.index = Index(space='l2', dim=39)
        self.index.init_index(max_elements=total_frames, ef_construction=200, M=16)
        self.total_frames = total_frames
        self.n_nearest_frames = n_nearest_frames

    def add(self, feature: AudioFeatureType, idxs):
        assert len(idxs.shape) == 1  # 1-D
        assert feature.shape[0] == idxs[0]  # same size
        assert idxs.max() < self.total_frames
        self.index.add_items(feature, idxs)

    def query(self, feature: AudioFeatureType):
        knn_points, _distances = self.index.knn_query(feature, k=self.n_nearest_frames)

        accumulations = HoughAccumulations()
        for m_idx, n_idxs in enumerate(list(knn_points)):
            # slope constraint
            slope_candidates = [1]
            for slope in slope_candidates:
                for n_idx in list(n_idxs):
                    offset = slope * -m_idx + n_idx
                    accumulations.add(slope, offset, n_idx)

        candidates = accumulations.peaks(HOUGH_PEAKS)

        merged = set()
        result = []
        for idx, ((_, offset), count, points) in enumerate(candidates):
            if idx in merged:
                continue
            cur_left = min(points)
            cur_right = max(points)
            cur_count = count
            for idx2 in range(idx + 1, HOUGH_PEAKS):
                if idx2 in merged:
                    continue
                (_, offset_2), count_2, points_2 = candidates[idx2]
                if (offset - offset_2) < OFFSET_MERGE_THRESHOLD:
                    cur_count += count_2
                    cur_left = min(cur_left, min(points_2))
                    cur_right = max(cur_right, max(points_2))
                    merged.add(idx2)
            result.append((cur_count, cur_left, cur_right))  # score, start_frame, end_frame



