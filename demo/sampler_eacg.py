import time
import json

from eacgm.bpf import BccBPF
from eacgm.sampler import eBPFSampler, NVMLSampler, GPUSampler
from eacgm.collector import to_perfetto

text = """
#include <uapi/linux/ptrace.h>

int cudaLaunchKernelEntry(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    u32 g_x = PT_REGS_PARM2(ctx) & 0xFFFF;
    u32 g_y = PT_REGS_PARM2(ctx) >> 32;
    u32 g_z = PT_REGS_PARM3(ctx) & 0xFFFF;
    u32 b_x = PT_REGS_PARM4(ctx) & 0xFFFF;
    u32 b_y = PT_REGS_PARM4(ctx) >> 32;
    u32 b_z = PT_REGS_PARM5(ctx) & 0xFFFF;
    // bpf_trace_printk("0 ----- cudaLaunchKernel %u %u %u\\n", g_x, g_y, g_z);
    // bpf_trace_printk("0 ----- cudaLaunchKernel %u %u %u\\n", b_x, b_y, b_z);
    u32 stream_num = g_x * g_y * g_z * b_x * b_y * b_z;
    bpf_trace_printk("%ld@start@cudaLaunchKernel@%u\\n", ts, stream_num);
    return 0;
};

int cudaLaunchKernelExit(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld@end@cudaLaunchKernel\\n", ts);
    return 0;
};

int ncclAllReduceEntry(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    u64 size_count = PT_REGS_PARM3(ctx);
    u64 data_type  = PT_REGS_PARM4(ctx);
    u64 reduce_op  = PT_REGS_PARM5(ctx);
    if(reduce_op == 0){
        bpf_trace_printk("%ld@start@ncclAllReduce@%ld %ld Sum\\n", ts, size_count, data_type);
    }
    return 0;
};

int ncclAllReduceExit(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld@end@ncclAllReduce\\n", ts);
    return 0;
};

int PyObject_CallFunctionEntry(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld@start@PyObject_CallFunction\\n", ts);
    return 0;
};

int PyObject_CallFunctionExit(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld@end@PyObject_CallFunction\\n", ts);
    return 0;
};


"""

bpf = BccBPF("eACGSampler", text, ["-w"])

attach_config = [
    {
        "name": "CUDASampler",
        "exe_path": [
            "/home/msc-user/miniconda3/envs/py312-torch24-cu124/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12",
            "./temp.out",
        ],
        "exe_sym": [
            "cudaLaunchKernel",
        ]
    },
    {
        "name": "NCCLSampler",
        "exe_path": [
            "/home/msc-user/miniconda3/envs/py312-torch24-cu124/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2",
        ],
        "exe_sym": [
            "ncclAllReduce",
        ]
    },
    {
        "name": "PythonSampler",
        "exe_path": [
            "/home/msc-user/miniconda3/envs/py312-torch24-cu124/bin/python",
        ],
        "exe_sym": [
            # "PyObject_CallFunction",
        ]
    },
    {
        "name": "TorchSampler",
        "exe_path": [
            "/home/msc-user/miniconda3/envs/py312-torch24-cu124/lib/python3.12/site-packages/torch/./lib/libtorch_python.so",
        ],
        "exe_sym": [
            "_ZN5torch8autogradL16THPVariable_geluEP7_objectS2_S2_",
        ]
    },
]

eacg_sampler = eBPFSampler(bpf)
nvml_sampler = NVMLSampler()
gpu_sampler  = GPUSampler()

eacg_sampler.run(attach_config)

states = []
while True:
    try:
        samples = []
        samples += eacg_sampler.sample(time_stamp=1)
        samples += nvml_sampler.sample(time_stamp=1)
        samples += gpu_sampler.sample()
        states += samples
        for sample in samples:
            print(sample)
        print("---")
    except KeyboardInterrupt:
        break

eacg_sampler.close()
nvml_sampler.close()
gpu_sampler.close()

collector = to_perfetto(states)
json.dump(collector, open("temp.json", "w", encoding="utf-8"), indent=4)