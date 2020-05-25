from typing import Any, List, Tuple

import numpy as np
from nptyping import NDArray


AudioKey = int
Score = float
Second = float
AudioFeatureType = NDArray[(Any, Any), np.float32]
FrameIdxType = NDArray[(Any,), np.int64]
CandidateType = Tuple[Score, AudioKey, Second, Second]  # score, Audio key, start second, end second
QueryResult = List[CandidateType]
