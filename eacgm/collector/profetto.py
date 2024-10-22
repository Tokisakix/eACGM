from typing import List

from eacgm.sampler import eBPFSamplerState

def to_perfetto(states:List[eBPFSamplerState]) -> List:
    res = []
    for state in states:
        if not isinstance(state, eBPFSamplerState):
            continue
        state = state.collect()
        res.append(state)
    return res