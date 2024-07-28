"""
Paper: "UTRNet: High-Resolution Urdu Text Recognition In Printed Documents" presented at ICDAR 2023
Authors: Abdur Rahman, Arjun Ghosh, Chetan Arora
GitHub Repository: https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
Project Website: https://abdur75648.github.io/UTRNet/
Copyright (c) 2023-present: This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import math
import torch
from PIL import Image
import torch.utils.data
import torchvision.transforms as T
# from dataset import NormalizePAD

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = T.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img

def text_recognizer(img_cropped, model, converter, device):
    """ Image processing """
    img = img_cropped.convert('L')
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    w, h = img.size
    ratio = w / float(h)
    if math.ceil(32 * ratio) > 400:
        resized_w = 400
    else:
        resized_w = math.ceil(32 * ratio)
    img = img.resize((resized_w, 32), Image.Resampling.BICUBIC)
    transform = NormalizePAD((1, 32, 400))
    img = transform(img)
    img = img.unsqueeze(0)
    # print(img)
    batch_size = 1
    img = img.to(device)
    
    """ Prediction """
    preds = model(img)
    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
    _, preds_index = preds.max(2)
    preds_str = converter.decode(preds_index.data, preds_size.data)[0]
    return preds_str

