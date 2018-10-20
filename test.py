# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test
   Description :
   Author :       haxu
   date：          2018/10/12
-------------------------------------------------
   Change Activity:
                   2018/10/12:
-------------------------------------------------
"""
__author__ = 'haxu'

import torch
import torchvision

if __name__ == '__main__':
    x = torch.randn(160, 1, 101, 101)
    img_grid = torchvision.utils.make_grid(x)
    torchvision.utils.save_image(img_grid, 'img.png')
