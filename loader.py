import os
import random
import numpy as np
from tqdm import tqdm
import _pickle as cPickle

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as torch_transforms

import cv2

'''
DATASET
'''
def image_resize(image, width=None, height=None, inter=cv2.INTER_CUBIC):
    # try our best to resize image but still keep the aspect ratio.
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        width = max(height, int(w * r))
    else:
        r = width / float(w)
        height = max(width, int(h * r))

    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized

def random_crop(arr_image, width, height):
    x = random.randint(0, arr_image.shape[1] - width)
    y = random.randint(0, arr_image.shape[0] - height)
    arr_image = arr_image[y:y + height, x:x + width]
    return arr_image

def random_fliplr(arr_image):
    r = random.randint(0, 2)
    if r == 1:
        return np.fliplr(arr_image)
    return arr_image

class MotorbikeDataset(Dataset):
    def __init__(self, path, size=64, margin=10, count=None, transforms=[], tmp_path_pkl="./data/tmp_dumped_data.pkl"):
        self.size = size
        self.margin = margin

        files = os.listdir(path)
        files = [os.path.join(path, file) for file in files]
        if count is not None: files = files[:count]

        # load image from file and do resize
        self.images, self.fns = self.load_image_from_files(files, tmp_path_pkl)

        # find the coressponding mean, std
        mean, std = self.calc_mean_and_std()
        print ("mean:", mean, "std:",std)

        self.mean = np.array(mean).reshape(-1, 1, 1)
        self.std = np.array(std).reshape(-1, 1, 1)

        # fullfill the transforms by adding normalization
        self.transforms = torch_transforms.Compose(
            transforms + [torch_transforms.Normalize(list(mean), list(std))]
        )

    def load_image_from_files(self, files, tmp_path_pkl):
        # resize before storing on RAM to save memory ...
        images = []
        fns = []
        sizem = self.size + self.margin

        # check checkpoint existed, if True, load from disk -> then return
        if os.path.exists(tmp_path_pkl):
            infos = cPickle.load(open(tmp_path_pkl,'rb'))

            images, fns = infos['images'], infos['fns']
            return images, fns

        print ('Reading image from disk ...')
        for fn in tqdm(files):
            #print ("Reading: %s" % fn)
            try:
                image = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)

                if image.shape[0] > image.shape[1]:
                    image = image_resize(image, width=sizem)
                else:
                    image = image_resize(image, height=sizem)

                image = image[..., ::-1].copy()
                images.append(image)
                fns.append(fn)
            except:
                print (">>> Cannot read: %s" % fn)

        # save to disk for faster loading in the next time
        cPickle.dump({
            'images':images,
            'fns':fns
        }, open(tmp_path_pkl,'wb'))

        return images, fns

    def calc_mean_and_std(self):
        mean = np.zeros(3) # seperated by r-g-b channels
        std = np.zeros(3)  # seperated by r-g-b channels

        for x in self.images:
            x = np.transpose(x, (2, 0, 1)) # (h,w,c) -> (c,h,w)
            x = x.reshape(3, -1) # (c, h * w)

            mean += x.mean(1)
            std += x.std(1)

        m = (mean / len(self.images)) / 255.0
        s = (std / len(self.images)) / 255.0

        return m, s

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        fn  = self.fns[idx]

        #convert cv2 -> pil -> tensor
        pil_image = Image.fromarray(img).convert('RGB')
        tensor_image = self.transforms(pil_image)

        return tensor_image, fn
