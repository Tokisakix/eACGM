from typing import List

from eacgm.sampler import eBPFSamplerState

def to_perfetto(states:List[eBPFSamplerState]) -> List:
    res = []
    last_event = {}
    for state in states:
        if not isinstance(state, eBPFSamplerState):
            continue
        state = state.collect()
        last_state = last_event.get(state["name"], None)
        if last_state is None:
            last_event[state["name"]] = state
            continue
        res.append(last_state)
        res.append(state)
        last_event[state["name"]] = None
    return res