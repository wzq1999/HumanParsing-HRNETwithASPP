# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset


class my_data(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=20,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=473,
                 crop_size=(473, 473),
                 downsample_rate=1,
                 scale_factor=11,
                 center_crop_test=False,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(my_data, self).__init__(ignore_label, base_size, crop_size,
                                      downsample_rate, scale_factor, mean, std)

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.img_list = [
            line.strip().split() for line in open(root + list_path)
        ]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]
        self.label_mapping = {
            -1: ignore_label,
            0: ignore_label,
            1: 1,
        }

    def read_files(self):
        files = []
        print(self.img_list)
        for item in self.img_list:
            image_path, label_path = item[:2]
            name = os.path.splitext(os.path.basename(label_path))[0]
            sample = {
                "img": image_path,
                "label": label_path,
                "name": name,
            }
            files.append(sample)
        return files

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(
            os.path.join(self.root, '', item["img"]),
            cv2.IMREAD_COLOR)
        label = cv2.imread(
            os.path.join(self.root, '',
                         item["label"]), cv2.IMREAD_GRAYSCALE)
        print('image path='+self.root+''+item["img"])
        print('label path='+self.root+''+item["label"])
        try:
        	label.shape
        except:
        	print("cant open label")
        try:
        	image.shape
        except:
        	print("cant open img")
        size = label.shape

        if 'testval' in self.list_path:
            image = cv2.resize(image,
                               self.crop_size,
                               interpolation=cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name

        if self.flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, ::flip, :]
            label = label[:, ::flip]

            if flip == -1:
                right_idx = [15, 17, 19]
                left_idx = [14, 16, 18]
                for i in range(0, 3):
                    right_pos = np.where(label == right_idx[i])
                    left_pos = np.where(label == left_idx[i])
                    label[right_pos[0], right_pos[1]] = left_idx[i]
                    label[left_pos[0], left_pos[1]] = right_idx[i]

        image, label = self.resize_image(image, label, self.crop_size)
        image, label = self.gen_sample(image, label, self.multi_scale, False)

        return image.copy(), label.copy(), np.array(size), name

    def inference(self, model, image, flip):
        size = image.size()
        pred = model(image)
        pred = F.upsample(input=pred,
                          size=(size[-2], size[-1]),
                          mode='bilinear')
        if False:
            flip_img = image.numpy()[:, :, :, ::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))
            flip_output = F.upsample(input=flip_output,
                                     size=(size[-2], size[-1]),
                                     mode='bilinear')
            flip_output = flip_output.cpu().numpy()
            flip_pred = flip_output.copy()
            flip_pred[:, 14, :, :] = flip_output[:, 15, :, :]
            flip_pred[:, 15, :, :] = flip_output[:, 14, :, :]
            flip_pred[:, 16, :, :] = flip_output[:, 17, :, :]
            flip_pred[:, 17, :, :] = flip_output[:, 16, :, :]
            flip_pred[:, 18, :, :] = flip_output[:, 19, :, :]
            flip_pred[:, 19, :, :] = flip_output[:, 18, :, :]
            flip_pred = torch.from_numpy(
                flip_pred[:, :, :, ::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i] + '.png'))
