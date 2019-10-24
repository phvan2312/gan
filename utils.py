import torch
import torchvision.transforms as transforms
from copy import deepcopy
from torchvision.utils import make_grid
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt
def imgshow(img):
    plt.imshow(img)
    plt.show()

def tensor_show(tensor_inputs, n_imgs_per_row=8):
    n_images = tensor_inputs.size(0)
    grid_images = np.transpose(make_grid(tensor_inputs, nrow=8, normalize=True).cpu().numpy(), (1,2,0))

    imgshow(grid_images)

def save_models(g_model, d_model, other_params, out_path):
    save_dct = {
        "g_model_state_dict":deepcopy(g_model).state_dict(),
        "d_model_state_dict":deepcopy(d_model).state_dict()
    }
    save_dct.update(other_params)

    torch.save(save_dct, out_path)
    print("saved to %s ..." % out_path)
