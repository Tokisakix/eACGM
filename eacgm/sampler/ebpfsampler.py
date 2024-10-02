import time
from typing import List

from .base import BaseSamplerState, BaseSampler
from eacgm.bpf import BPFState, BccBPF

class eBPFSamplerState(BaseSamplerState):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def from_ebpfstate(other:BPFState) -> "eBPFSamplerState":
        state = eBPFSamplerState()
        state.task = other.task
        state.pid  = other.pid
        state.cpu  = other.cpu
        state.timestamp = other.timestamp
        state.message   = other.message
        return state
    
    def __repr__(self) -> str:
        info = f"eBPFSamplerState {super().__repr__()}"
        return info

class eBPFSampler(BaseSampler):
    def __init__(self, bpf:BccBPF) -> None:
        super().__init__(name="eBPFSampler")
        self.bpf = bpf
        return
    
    def run(self, attach_config:List) -> None:
        for attach_info in attach_config:
            name = attach_info["name"]
            exe_path = attach_info["exe_path"]
            exe_sym = attach_info["exe_sym"]
            for path in exe_path:
                for sym in exe_sym:
                    try:
                        self.bpf.attach_uprobe(path, sym, sym + "Entry")
                        self.bpf.attach_uretprobe(path, sym, sym + "Exit")
                    except Exception as e:
                        print(e)
        return
    
    def sample(self, time_stamp:float) -> List[eBPFSamplerState]:
        samplers = []
        start_time = time.perf_counter()

        flag = True
        while flag:
            if time.perf_counter() > start_time + time_stamp:
                flag = False
            state = self.bpf.trace_ebpf(True)
            if state.is_none():
                continue
            state = eBPFSamplerState.from_ebpfstate(state)
            samplers.append(state)

        return samplers

    def close(self) -> None:
        self.bpf.cleanup()
        return