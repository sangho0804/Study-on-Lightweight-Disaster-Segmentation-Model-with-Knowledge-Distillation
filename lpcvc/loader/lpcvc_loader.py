import os
import torch
import numpy as np
import imageio as m

from torch.utils import data
import torchvision.transforms.functional as TF
from lpcvc.utils import recursive_glob
from lpcvc.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale


class LPCVCLoader(data.Dataset):

    colors = [
            (0,  0,  0),
            (92, 177, 172),
            (50, 183, 250),
            (155, 188, 221),
            (189, 177, 252),
            (221, 255, 51),
            (255, 53, 94),
            (170, 240, 209),
            (255, 204, 51),
            (198, 116, 54),
            (61, 245, 61),
            (65, 105, 146),
            (240, 120, 240),
            (61, 61, 245) 
             ]   

    label_colours = dict(zip(range(14), colors))



    def __init__(
        self,
        root,
        split,
        augmentations=None,
        model_name=None,
        test_mode = False
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.test_mode = test_mode
        self.root = root
        self.split = split
        self.augmentations = augmentations
        self.model_name=model_name
        self.n_classes = 14

        self.images_base = os.path.join(self.root, "IMG", self.split)
        self.annotations_base = os.path.join(self.root, "GT", self.split)

        self.files = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.valid_classes = range(14)
  
        self.ignore_index = 250
        ignore_label = 250
        self.class_map = dict(zip(self.valid_classes, range(14)))

        if len(self.files)==0:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d images in: %s" % (len(self.files),split))

        # label change
        self.id_to_trainid = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 
                              11:11, 12:12, 13:13}

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def __len__(self):
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[index].rstrip()
        lbl_path = os.path.join(
                self.annotations_base,
                os.path.basename(img_path),
            )

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        h_,w_,c_ = img.shape

        lbl = m.imread(lbl_path)
        if len(lbl.shape)==3:
            lbl = lbl[:,:,0]
            
        # if you want to change label index
        # lbl = self.id2trainId(lbl)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        if self.test_mode:
            return img, lbl, os.path.basename(img_path), w_, h_
        else:
            return img, lbl


    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

