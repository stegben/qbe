import json
from pathlib import Path
from collections import Counter, defaultdict

from hnswlib import Index
from tqdm import tqdm

from .types import AudioFeatureType, IndexQueryResult, FrameIdxType


class HoughAccumulations:

    def __init__(self):
        self.counts = Counter()
        self.key2labels = defaultdict(list)

    def add(self, slope, offset, label):
        key = self.hash(slope, offset)
        self.counts.update([key])
        self.key2labels[key].append(label)

    # TODO: add slopes one by one is the bottle neck
    # def add_batch(self, data):
    #     keys = [self.hash(slope, offset) for slope, offset, _ in data]
    #     self.counts.update(keys)
    #     for key, label

    def peaks(self, k):
        candidates = self.counts.most_common(k)
        result = []
        for key, count in candidates:
            result.append((key, count, self.key2labels[key]))
        return result

    def hash(self, x, y):
        return (int(x), int(y))


SAVED_INDEX_NAME = 'index.bin'
SAVED_BUILD_ARGS_NAME = 'build_args.json'


class SimpleRails:

    def __init__(
            self,
            dim: int,
            total_frames: int,
            hnsw_space='l2',
            hnsw_ef_construction=200,
            hnsw_M=16,
        ):
        self.dim = dim
        self.total_frames = total_frames

        self.hnsw_space = hnsw_space
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_M = hnsw_M

        self.index = Index(space=hnsw_space, dim=dim)
        self.index.init_index(
            max_elements=total_frames,
            ef_construction=hnsw_ef_construction,
            M=hnsw_M,
        )

    def add(self, feature: AudioFeatureType, idxs: FrameIdxType):
        assert len(idxs.shape) == 1  # 1-D
        assert feature.shape[0] == idxs.shape[0]  # same size
        assert idxs.max() < self.total_frames
        self.index.add_items(feature, idxs)

    def set_query_params(
            self,
            ef=200,
            n_nearest_frames=100,
            n_hough_peaks=100,
            offset_merge_threshold=10,
        ):
        self.index.set_ef(ef)
        self.n_nearest_frames = n_nearest_frames
        self.n_hough_peaks = n_hough_peaks
        self.offset_merge_threshold = offset_merge_threshold

    def query(self, feature: AudioFeatureType) -> IndexQueryResult:
        knn_points, _distances = self.index.knn_query(feature, k=self.n_nearest_frames)

        accumulations = HoughAccumulations()
        for m_idx, n_idxs in enumerate(list(knn_points)):
            # slope constraint
            slope_candidates = [1]
            for slope in slope_candidates:
                for n_idx in list(n_idxs):
                    offset = slope * -m_idx + n_idx
                    accumulations.add(slope, offset, n_idx)

        candidates = accumulations.peaks(self.n_hough_peaks)

        merged = set()
        result = []
        for idx, ((_, offset), count, points) in enumerate(candidates):
            if idx in merged:
                continue
            cur_left = min(points)
            cur_right = max(points)
            cur_count = count
            for idx2 in range(idx + 1, self.n_hough_peaks):
                if idx2 in merged:
                    continue
                (_, offset_2), count_2, points_2 = candidates[idx2]
                if abs((offset - offset_2)) < self.offset_merge_threshold:
                    cur_count += count_2
                    cur_left = min(cur_left, min(points_2))
                    cur_right = max(cur_right, max(points_2))
                    merged.add(idx2)
            result.append((cur_count, cur_left, cur_right))  # score, start_frame, end_frame
        return result

    def save(self, path):
        try:
            path = Path(path)
            path.mkdir(mode=0o775, parents=True, exist_ok=True)
            self.index.save_index(str(path / SAVED_INDEX_NAME))
            build_args = {
                'dim': self.dim,
                'total_frames': self.total_frames,
                'hnsw_space': self.hnsw_space,
                'hnsw_ef_construction': self.hnsw_ef_construction,
                'hnsw_M': self.hnsw_M,
            }
            with open(str(path / SAVED_BUILD_ARGS_NAME), 'w') as fw:
                json.dump(build_args, fw)
            return True
        except Exception as e:
            print(e)
            return False

    @classmethod
    def load(cls, path):
        path = Path(path)
        with open(str(path / SAVED_BUILD_ARGS_NAME), 'r') as f:
            build_args = json.load(f)

        index = cls(**build_args)
        index.index.load_index(str(path / SAVED_INDEX_NAME))
        return index
