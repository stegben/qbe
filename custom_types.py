from typing import Any, List, Tuple

import numpy as np
from nptyping import NDArray


VoiceIndex = int
Second = float
VoiceType = NDArray[(Any, Any), np.float32]
CandidateType = Tuple[VoiceIndex, Second, Second]  # voice ID, start second, end second
QueryResult = List[CandidateType]
