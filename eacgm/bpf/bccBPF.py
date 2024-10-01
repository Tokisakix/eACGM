from bcc import BPF
from typing import List

from .base import BPFState, BaseBPF

class BccBPF(BaseBPF):
    def __init__(self, name:str, text:str, cflags:List=[]) -> None:
        super().__init__(name)
        self.bpf = BPF(text=text, cflags=cflags)
        return
    
    def attach_kprobe(self, event, fn_name):
        return self.bpf.attach_kprobe(event, fn_name=fn_name)

    def attach_uprobe(self, exe_path:str, exe_sym:str, bpf_func:str) -> bool:
        self.bpf.attach_uprobe(exe_path, exe_sym, bpf_func)
        return

    def attach_uretprobe(self, exe_path:str, exe_sym:str, bpf_func:str) -> bool:
        self.bpf.attach_uretprobe(exe_path, exe_sym, bpf_func)
        return
    
    def trace_ebpf(self, nonblocking:bool) -> BPFState:
        (task, pid, cpu, _, timestamp, message) = self.bpf.trace_fields(nonblocking)
        state = BPFState()
        if task is not None:
            state.task = task
            state.pid  = int(pid)
            state.cpu  = int(cpu)
            state.timestamp = int(timestamp * 1_000_000)
            state.message   = message
        return state