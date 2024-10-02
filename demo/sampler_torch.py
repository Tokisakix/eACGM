import time

from eacgm.bpf import BccBPF
from eacgm.sampler import eBPFSampler

text = """
int TorchGeLUEntry(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld start TorchGeLU\\n", ts);
    return 0;
};

int TorchGeLUExit(struct pt_regs *ctx){
    u64 ts = bpf_ktime_get_ns();
    bpf_trace_printk("%ld end TorchGeLU\\n", ts);
    return 0;
};
"""

bpf = BccBPF("TorcheBPF", text, ["-w"])

attach_config = [
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