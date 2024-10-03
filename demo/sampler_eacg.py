import time

from eacgm.bpf import BccBPF
from eacgm.sampler import eBPFSampler, NVMLSampler, GPUSampler

text = """
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

int ncclAllReduceEntry(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld start ncclAllReduce\\n", ts);
    return 0;
};

int ncclAllReduceExit(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld end ncclAllReduce\\n", ts);
    return 0;
};

int PyObject_CallFunctionEntry(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld start PyObject_CallFunction\\n", ts);
    return 0;
};

int PyObject_CallFunctionExit(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld end PyObject_CallFunction\\n", ts);
    return 0;
};

int _ZN5torch8autogradL16THPVariable_geluEP7_objectS2_S2_Entry(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld start TorchGeLU\\n", ts);
    return 0;
};

int _ZN5torch8autogradL16THPVariable_geluEP7_objectS2_S2_Exit(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld end TorchGeLU\\n", ts);
    return 0;
};
"""

bpf = BccBPF("eACGSampler", text, ["-w"])

attach_config = [
    {
        "name": "CUDASampler",
        "exe_path": [
            "/home/msc-user/miniconda3/envs/py312-torch24-cu124/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12",
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
            "PyObject_CallFunction",
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

while True:
    try:
        samples = []
        samples += eacg_sampler.sample(time_stamp=1)
        samples += nvml_sampler.sample(time_stamp=1)
        samples += gpu_sampler.sample()
        for sample in samples:
            print(sample)
        print("---")
    except KeyboardInterrupt:
        break

eacg_sampler.close()
nvml_sampler.close()
gpu_sampler.close()