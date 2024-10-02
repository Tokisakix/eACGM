import time

from eacgm.bpf import BccBPF
from eacgm.sampler import eBPFSampler

text = """
int cudaMallocEntry(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld start cudaMalloc\\n", ts);
    return 0;
};

int cudaMallocExit(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld end cudaMalloc\\n", ts);
    return 0;
};

int cudaMemcpyEntry(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld start cudaMemcpy\\n", ts);
    return 0;
};

int cudaMemcpyExit(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld end cudaMemcpy\\n", ts);
    return 0;
};

int cudaFreeEntry(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld start cudaFree\\n", ts);
    return 0;
};

int cudaFreeExit(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld end cudaFree\\n", ts);
    return 0;
};

int cudaLaunchKernelEntry(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld start cudaLaunchKernel\\n", ts);
    return 0;
};

int cudaLaunchKernelExit(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld end cudaLaunchKernel\\n", ts);
    return 0;
};
"""

bpf = BccBPF("CUDAeBPF", text, ["-w"])

attach_config = [
    {
        "name": "CUDASampler",
        "exe_path": [
            "/home/msc-user/miniconda3/envs/py312-torch24-cu124/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12",
        ],
        "exe_sym": [
            "cudaMalloc",
            "cudaMemcpy",
            "cudaFree",
            "cudaLaunchKernel",
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