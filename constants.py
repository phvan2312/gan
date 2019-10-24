import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

### training
d_lr = 8e-4
g_lr = 1e-3
epoch = 200
batch_size = 64
n_workers = 4

img_size = 128
img_margin = 15
debug_n_sample = -1

sample_interval = 1000
lr_schedule_after_epoch = 20