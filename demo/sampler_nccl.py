import time

from eacgm.bpf import BccBPF
from eacgm.sampler import eBPFSampler

text = """
#include <uapi/linux/ptrace.h>

int ncclAllReduceEntry(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    u64 size_count = PT_REGS_PARM3(ctx);
    u64 data_type  = PT_REGS_PARM4(ctx);
    u64 reduce_op  = PT_REGS_PARM5(ctx);
    bpf_trace_printk("%ld start ncclAllReduce %ld\\n", ts, size_count);
    bpf_trace_printk("%ld %ld\\n", data_type, reduce_op);
    return 0;
};

int ncclAllReduceExit(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld end ncclAllReduce\\n", ts);
    return 0;
};
"""

bpf = BccBPF("NCCLeBPF", text, ["-w"])

attach_config = [
    {
        "name": "NCCLSampler",
        "exe_path": [
            "/home/msc-user/miniconda3/envs/py312-torch24-cu124/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2",
        ],
        "exe_sym": [
            "ncclAllReduce",
        ]
    },
]

sampler = eBPFSampler(bpf)

sampler.run(attach_config)

while True:
    try:
        samples = sampler.sample(time_stamp=1)
        for sample in samples:
            print(sample)
        print("---")
    except KeyboardInterrupt:
        break

sampler.close()