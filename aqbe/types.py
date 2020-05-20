from typing import Any, List, Tuple

import numpy as np
from nptyping import NDArray


VoiceKey = int
Score = float
Second = float
VoiceType = NDArray[(Any, Any), np.float32]
CandidateType = Tuple[Score, VoiceKey, Second, Second]  # score, voice key, start second, end second
QueryResult = List[CandidateType]
