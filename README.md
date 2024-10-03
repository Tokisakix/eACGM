# eACGM - eBPF-based AI, CUDA, and GPU Monitoring Framework

目前实现了以下功能：

- [x] 基于 eBPF 实现对 CUDA Runtime 的事件探测
- [x] 基于 eBPF 实现对 NCCL GPU 通信库的事件探测
- [x] 基于 eBPF 实现对 Python 虚拟机的函数调用探测
- [x] 基于 eBPF 实现对 Pytorch 的算子探测
- [x] 基于 libnvml 实现对进程级的 GPU 信息探测
- [x] 基于 libnvml 实现对 GPU 的信息探测

后续需要完善的功能：

- [ ] 实现 eBPF 程序代码自动生成
- [ ] 实现对捕获到的 CUDA Runtime 事件进行分析
- [ ] 实现对捕获到的 NCCL GPU 通信库事件进行分析
- [ ] 实现对捕获到的 Python 虚拟机的函数调用进行分析
- [ ] 实现对捕获到的 Pytorch 的算子进行分析

闲的没事可以做的功能：

- [ ] 捕获更多的 CUDA Runtime 事件
- [ ] 捕获更多的 NCCL GPU 通信库事件
- [ ] 捕获更多的 Pytorch 的算子

## Sampler/采集器

### 1. eBPFSampler

`eBPFSampler` 基于 eBPF 进行事件采集，适当初始化后可实现对 CUDA Runtime、NCCL、Python 虚拟机和 Pytorch 算子的事件采集

`eBPFSampler` 进行采集时需要指定采样时间 `time_stamp`，单位为 `s`，进行采集时程序会阻塞

`eBPFSampler.sample()` 函数会返回过去采样时间段里采样的结果，返回值类型为 `List[eBPFSamplerState]`

其中 `eBPFSamplerState` 拥有以下字段属性：

```Python
task:str # 发生此事件的进程名
pid:int # 发生此事件的进程号
cpu:int # 进程使用 CPU 编号
timestamp:int # 此事件发生的事件
message:str # 此事件的相关信息
```

示例演示如下：

```Python
import time

from eacgm.bpf import BccBPF
from eacgm.sampler import eBPFSampler

# 注入的 bpf 程序
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

# 创建 bpf 类，用于 eBPF 注入
bpf = BccBPF("eACGSampler", text, ["-w"])

# eBPF 注入配置文件，规定 eBPF 在什么地方注入什么函数
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

# 创建采集器/Sampler
sampler = eBPFSampler(bpf)

# 运行采集器/Sampler
sampler.run(attach_config)

while True:
    try:
        # 定时采集，并输出
        samples = sampler.sample(time_stamp=1)
        for sample in samples:
            print(sample)
        print("---")
    except KeyboardInterrupt:
        break

# 关闭采集器/Sampler
sampler.close()
```

### 2. NVMLSampler

`NVMLSampler` 基于 libnvml 进行采集，用于采集每个进程的 GPU 信息

`NVMLSampler` 进行采集时需要指定采样时间 `time_stamp`，单位为 `s`，进行采集时程序不会阻塞

`NVMLSampler.sample()` 函数会返回过去采样时间段里采样的结果，返回值类型为 `List[NVMLSamplerState]`

其中 `NVMLSamplerState` 拥有以下字段属性：

```Python
task:str # 进程名
pid:int # 进程号
cpu:int # 进程使用 CPU 编号
timestamp:int # 采样时间
message:str # 相关信息
gpu:int # 进程使用的 GPU 信息
sm:int # GPU 使用率
mem:int # 显存使用率
enc:int # 编码率
dec:int # 解码率
```

示例演示如下：

```Python
import time

from eacgm.sampler import NVMLSampler

sampler = NVMLSampler()

sampler.run()

while True:
    try:
        for sample in sampler.sample(time_stamp=1):
            print(sample)
        time.sleep(1)
        print("---")
    except KeyboardInterrupt:
        break

sampler.close()
```

### 3. GPUSampler

`GPUSampler` 基于 libnvml 进行采集，用于采集 GPU 的信息

`GPUSampler` 进行采集时不需要额外参数，进行采集时程序不会阻塞

`GPUSampler.sample()` 函数会返回此刻采样的结果，返回值类型为 `List[GPUSamplerState]`

其中 `GPUSamplerState` 拥有以下字段属性：

```Python
gpu:int # GPU 编号
name:str # GPU 型号
sm:int # GPU 使用率
totMem:int # 总显存
usedMem:int # 已用显存
enc:int # 编码率
dec:int # 解码率
tmp:int # 温度
fan:int # 风扇使用率
usedPower:float # 当前功耗
totPower:float # 最大功耗
```

示例演示如下：

```Python
import time

from eacgm.sampler import GPUSampler

sampler = GPUSampler()

sampler.run()

while True:
    try:
        for sample in sampler.sample():
            print(sample)
        time.sleep(1)
        print("---")
    except KeyboardInterrupt:
        break

sampler.close()
```