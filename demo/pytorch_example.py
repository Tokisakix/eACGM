import os
import time
import torch
import torch.distributed as dist
from tqdm import tqdm

def init_process(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    init_process(rank, world_size)
    
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    transformer = torch.nn.Transformer(d_model=512, nhead=8, batch_first=True).to(device)
    inputs = torch.rand(512, 512).to(device)

    epochs = 1
    sleep = 1

    for _ in tqdm(range(epochs)):
        outputs = transformer(inputs, inputs)
        print(outputs.dtype)

        dist.all_reduce(outputs, op=dist.ReduceOp.SUM)
        
        outputs = torch.nn.functional.gelu(outputs)
        time.sleep(sleep)

    cleanup()

if __name__ == "__main__":
    world_size = 2
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '10001'
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
