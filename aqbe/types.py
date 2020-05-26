from typing import Any, List, Tuple

import numpy as np
from nptyping import NDArray


FrameID = int
AudioKey = str
Score = float
Second = float

AudioType = NDArray[(Any,), np.float32]  # single channel, -1 ~ 1, length == sr * secs
LoadedAudioType = Tuple[AudioType, int]  # Audio, sampleing rate

AudioFeatureType = NDArray[(Any, Any), np.float32]
FrameIdxType = NDArray[(Any,), np.int64]

IndexQueryResult = List[Tuple[Score, FrameID, FrameID]]  # score, start frame id, end frame id
CandidateType = Tuple[Score, AudioKey, Second, Second]  # score, Audio key, start second, end second
QueryResult = List[CandidateType]

AlignmentType = Tuple[str, AudioKey, Second, Second]  # label, Audio key, start_second, end_second
