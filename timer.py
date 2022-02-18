import torch
import monai
import numpy as np
import matplotlib.pyplot as plt
import glob
import tqdm
from tqdm import tqdm
import time

model = monai.networks.nets.UNet(
    dimensions=2,
    in_channels=1,
    out_channels=4,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

x = torch.randn(1000, 64)

cpu_times = []

for epoch in range(100):
    t0 = time.perf_counter()
    output = model(x)
    t1 = time.perf_counter()
    cpu_times.append(t1 - t0)

device = 'cuda'
model = model.to(device)
x = x.to(device)
torch.cuda.synchronize()

gpu_times = []
for epoch in range(100):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    output = model(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    gpu_times.append(t1 - t0)

print('CPU {}, GPU {}'.format(
    torch.tensor(cpu_times).mean(),
    torch.tensor(gpu_times).mean()))