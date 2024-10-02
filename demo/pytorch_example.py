import time
import torch
from tqdm import tqdm

epochs = 1024
sleep  = 0.01
device = "cpu"

transformer = torch.nn.Transformer(d_model=512, nhead=8, batch_first=True).to(device)
inputs = torch.rand(512, 512).to(device)

for _ in tqdm(range(epochs)):
    outputs = transformer(inputs, inputs)
    outputs = torch.nn.functional.gelu(inputs)
    time.sleep(sleep)
