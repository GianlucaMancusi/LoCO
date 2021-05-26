# -*- coding: utf-8 -*-
# ---------------------

import json
from os import path
from typing import *

import numpy as np
import torch
from pycocotools.coco import COCO as MOTS
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

import utils
from conf import Conf

# 14 useful joints for the MOTSynth dataset
from models import BaseModel

USEFUL_JOINTS = [0, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18, 19, 20, 21]

# number of sequences
N_SEQUENCES = 256

# number frame in each sequence
N_FRAMES_IN_SEQ = 1800

# number of frames used for training in each sequence
N_SELECTED_FRAMES = 180

# maximum MOTSynth camera distance [m]
MAX_CAM_DIST = 100

SIGMA = 2

Point3D = Tuple[float, float, float]
Point2D = Tuple[float, float]


class MOTSynthDS(Dataset):
    """
    Dataset composed of pairs (x, y) in which:
    * x: 3D heatmap form the JTA dataset
    * y: json-list of gaussian centers (order: D, H, W)
    """

    def __init__(self, mode, cnf=None, debug=False):
        # type: (str, BaseModel, Conf, debug) -> None
        """
        :param mode: values in {'train', 'val'}
        :param cnf: configuration object
        """
        self.mode = mode
        self.cnf = cnf
        self.debug = debug
        assert mode in {'train', 'val', 'test'}, '`mode` must be \'train\' or \'val\''

        self.mots_ds = None
        if mode == 'train':
            self.mots_ds = MOTS(path.join(self.cnf.mot_synth_path, 'annotations', '000.json'))
            # self.mots_ds = MOTS(path.join(self.cnf.mot_synth_path, 'annotation_groups', 'MOTSynth_annotations_40.json'))
        if mode == 'val':
            self.mots_ds = MOTS(path.join(self.cnf.mot_synth_path, 'annotations', '001.json'))
        if mode == 'test':
            self.mots_ds = MOTS(path.join(self.cnf.mot_synth_path, 'annotations', '002.json'))

        self.catIds = self.mots_ds.getCatIds(catNms=['person'])
        self.imgIds = self.mots_ds.getImgIds(catIds=self.catIds)

        # max_cam_dist = self.get_dataset_max_cam_len()

        self.g = (SIGMA * 5 + 1) if (SIGMA * 5) % 2 == 0 else SIGMA * 5
        self.gaussian_patch = utils.gkern(
            w=self.g, h=self.g, d=self.g,
            center=(self.g // 2, self.g // 2, self.g // 2),
            s=SIGMA, device='cpu'
        )

    def __len__(self):
        # type: () -> int
        return len(self.imgIds)

    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor]

        # select sequence name and frame number
        img = self.mots_ds.loadImgs(self.imgIds[i])[0]

        # load corresponding data
        ann_ids = self.mots_ds.getAnnIds(imgIds=img['id'], catIds=self.catIds, iscrowd=None)
        anns = self.mots_ds.loadAnns(ann_ids)

        x_heatmap, aug_info = self.generate_3d_heatmap(anns, augmentation=True)

        x_image = self.get_frame(file_path=img['file_name'])

        plt.imshow(x_image.permute(1, 2, 0))
        plt.show()

        # image augmentation
        aug_scale, aug_h, aug_w = aug_info
        _, img_h, img_w = x_image.shape
        # convert the offset calculated for 3d points (for the 3d heatmap) to offset useful for
        # the Affine transformation by using the imgaug library
        aug_offset_h = -(aug_h - .5) * (img_h * aug_scale - img_h)
        aug_offset_w = -(aug_w - .5) * (img_w * aug_scale - img_w)
        aug_affine = iaa.Affine(scale=aug_scale, translate_px={'x': int(round(aug_offset_w)), 'y': int(round(aug_offset_h))})
        x_image = aug_affine(images=x_image.numpy(), return_batch=False)
        x_image = torch.from_numpy(x_image)

        plt.imshow(x_image.permute(1, 2, 0))
        plt.show()

        return x_image, x_heatmap

    def generate_3d_heatmap(self, anns, augmentation):
        # type: (List[dict], bool) -> Tuple[Tensor, Tuple[float, float, float]]
        # augmentation initialization (rescale + crop)
        h, w, d = self.cnf.hmap_h, self.cnf.hmap_w, self.cnf.hmap_d

        aug_scale = 2  # np.random.uniform(0.5, 2)
        aug_h = 1  # np.random.uniform(0, 1)
        aug_w = 1  # np.random.uniform(0, 1)
        aug_offset_h = aug_h * (h * aug_scale - h)
        aug_offset_w = aug_w * (w * aug_scale - w)

        all_hmaps = []
        y = []
        for jtype in USEFUL_JOINTS:

            # empty hmap
            x = torch.zeros((self.cnf.hmap_d, self.cnf.hmap_h, self.cnf.hmap_w)).to('cpu')

            joints = MOTSynthDS.get_joints_from_anns(anns, jtype)

            # for each joint of the same pose jtype
            for joint in joints:
                if joint['x2d'] < 0 or joint['y2d'] < 0 or joint['x2d'] > 1920 or joint['y2d'] > 1080:
                    continue

                # from GTA space to heatmap space
                cam_dist = np.sqrt(joint['x3d'] ** 2 + joint['y3d'] ** 2 + joint['z3d'] ** 2)
                cam_dist = cam_dist * ((self.cnf.hmap_d - 1) / MAX_CAM_DIST)
                joint['x2d'] /= 8
                joint['y2d'] /= 8

                # augmentation (rescale + crop)
                if augmentation:
                    joint['x2d'] = joint['x2d'] * aug_scale - aug_offset_w
                    joint['y2d'] = joint['y2d'] * aug_scale - aug_offset_h
                    cam_dist = cam_dist / aug_scale

                center = [
                    int(round(joint['x2d'])),
                    int(round(joint['y2d'])),
                    int(round(cam_dist))
                ]

                # ignore the point if due to augmentation the point goes out of the screen
                if min(center) < 0 or joint['x2d'] > w or joint['y2d'] > h or cam_dist > d:
                    continue

                center = center[::-1]

                xa, ya, za = max(0, center[2] - self.g // 2), max(0, center[1] - self.g // 2), max(0, center[
                    0] - self.g // 2)
                xb, yb, zb = min(center[2] + self.g // 2, self.cnf.hmap_w - 1), min(center[1] + self.g // 2,
                                                                                    self.cnf.hmap_h - 1), min(
                    center[0] + self.g // 2, self.cnf.hmap_d - 1)
                hg, wg, dg = (yb - ya) + 1, (xb - xa) + 1, (zb - za) + 1

                gxa, gya, gza = 0, 0, 0
                gxb, gyb, gzb = self.g - 1, self.g - 1, self.g - 1

                if center[2] - self.g // 2 < 0:
                    gxa = -(center[2] - self.g // 2)
                if center[1] - self.g // 2 < 0:
                    gya = -(center[1] - self.g // 2)
                if center[0] - self.g // 2 < 0:
                    gza = -(center[0] - self.g // 2)
                if center[2] + self.g // 2 > (self.cnf.hmap_w - 1):
                    gxb = wg - 1
                if center[1] + self.g // 2 > (self.cnf.hmap_h - 1):
                    gyb = hg - 1
                if center[0] + self.g // 2 > (self.cnf.hmap_d - 1):
                    gzb = dg - 1

                x[za:zb + 1, ya:yb + 1, xa:xb + 1] = torch.max(
                    torch.cat(tuple([
                        x[za:zb + 1, ya:yb + 1, xa:xb + 1].unsqueeze(0),
                        self.gaussian_patch[gza:gzb + 1, gya:gyb + 1, gxa:gxb + 1].unsqueeze(0)
                    ])), 0)[0]

                y.append([USEFUL_JOINTS.index(jtype)] + center)
            all_hmaps.append(x)
        y = json.dumps(y)
        x = torch.cat(tuple([h.unsqueeze(0) for h in all_hmaps]))
        return x, (aug_scale, aug_h, aug_w)

    def get_frame(self, file_path):
        # read input frame
        frame_path = self.cnf.mot_synth_path / file_path
        frame = utils.imread(frame_path)
        frame = transforms.ToTensor()(frame)
        frame = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(frame)
        return frame

    @staticmethod
    def get_joints_from_anns(anns, jtype):
        joints = []
        for ann in anns:
            joints.append({
                'x2d': ann['keypoints'][3 * jtype],
                'y2d': ann['keypoints'][3 * jtype + 1],
                'x3d': ann['keypoints_3d'][4 * jtype],
                'y3d': ann['keypoints_3d'][4 * jtype + 1],
                'z3d': ann['keypoints_3d'][4 * jtype + 2],
                'visibility': ann['keypoints_3d'][4 * jtype + 3],
                # visibility=0: not labeled (in which case x=y=z=0),
                # visibility=1: labeled but not visible
                # visibility=2: labeled and visible.
            })
        return joints


def main():
    import utils
    cnf = Conf(exp_name='default')

    ds = MOTSynthDS(mode='train', cnf=cnf, debug=True)
    loader = DataLoader(dataset=ds, batch_size=1, num_workers=0, shuffle=False)

    for i, sample in enumerate(loader):
        x_2d_image, y = sample

        print(f'({i}) Dataset example: x.shape={tuple(x_2d_image.shape)}, y={y}')


if __name__ == '__main__':
    main()
