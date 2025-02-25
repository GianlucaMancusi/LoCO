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
from models import CodePredictor
from models import Refiner
from post_processing import joint_association, filter_joints, refine_pose

import utils
from conf import Conf

import platform

# 14 useful joints for the MOTSynth dataset

USEFUL_JOINTS = [0, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18, 19, 20, 21]

# number of sequences
N_SEQUENCES = 256

# number frame in each sequence
N_FRAMES_IN_SEQ = 1800

# number of frames used for training in each sequence
N_SELECTED_FRAMES = 180

# maximum MOTSynth camera distance [m]
MAX_CAM_DIST = 100

# camera intrinsic parameters: fx, fy, cx, cy
CAMERA_PARAMS = 1158, 1158, 960, 540

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
        # type: (str, Conf, bool) -> None
        """
        :param mode: values in {'train', 'val'}
        :param cnf: configuration object
        """
        self.mode = mode
        self.cnf = cnf
        self.debug = debug
        assert self.mode in {'train', 'val', 'test'}, '`mode` must be \'train\' or \'val\''

        is_windows = any(platform.win32_ver())

        self.mots_ds = None
        path_to_anns = None

        if self.mode == 'train':
            if is_windows:
                path_to_anns = path.join(self.cnf.mot_synth_ann_path, 'annotations', '006.json')
            else:
                path_to_anns = path.join(self.cnf.mot_synth_ann_path, 'annotation_groups',
                                         'MOTSynth_annotations_10_train.json')
        if self.mode in ('val', 'test'):
            if is_windows:
                path_to_anns = path.join(self.cnf.mot_synth_ann_path, 'annotations', '002.json')
            else:
                path_to_anns = path.join(self.cnf.mot_synth_ann_path, 'annotation_groups',
                                         'MOTSynth_annotations_10_test.json')
        print(f'Annotation path to load: {path_to_anns}')
        self.mots_ds = MOTS(path_to_anns)

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
        if self.mode == 'train':
            return len(self.imgIds)
        elif self.mode in ('val', 'test'):
            return self.cnf.test_set_len

    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor]
        # select sequence name and frame number
        img = self.mots_ds.loadImgs(self.imgIds[i])[0]

        # load corresponding data
        ann_ids = self.mots_ds.getAnnIds(imgIds=img['id'], catIds=self.catIds, iscrowd=None)
        anns = self.mots_ds.loadAnns(ann_ids)

        augmentation = self.cnf.data_augmentation if self.mode == 'train' else 'no'
        x_heatmap, aug_info, y_coords3d, y_coords2d = self.generate_3d_heatmap(anns, augmentation=augmentation)

        x_image = self.get_frame(file_path=img['file_name'])

        x_image = np.array(x_image)
        original_image = np.copy(x_image)

        # resize to standard dimensions
        image_size_transformation = iaa.Resize({"height": 540, "width": 960}) if self.cnf.half_images \
            else iaa.Resize({"height": 1080, "width": 1920})
        x_image = image_size_transformation(image=x_image, return_batch=False)

        if augmentation in ('images', 'all'):
            img_aug_seq = iaa.Sequential([
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(1, 7)),
                    iaa.MedianBlur(k=(1, 11)),
                ]),
                iaa.SomeOf((0, 5),
                           [
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                               iaa.Sometimes(0.5, iaa.imgcorruptlike.MotionBlur(severity=(1, 3))),
                               iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                               iaa.imgcorruptlike.GaussianNoise(severity=(1, 2)),
                               iaa.SaltAndPepper((0.01, 0.2), per_channel=True),
                               iaa.Add((-10, 10), per_channel=0.5),
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                           ], random_order=True),
            ], random_order=True)
            x_image = img_aug_seq(image=x_image)

            # image augmentation
            aug_scale, aug_h, aug_w = aug_info
            img_h, img_w, _ = x_image.shape
            # convert the offset calculated for 3d points (for the 3d heat map) to offset useful for
            # the Affine transformation by using the imgaug library
            aug_offset_h = -(aug_h - .5) * (img_h * aug_scale - img_h)
            aug_offset_w = -(aug_w - .5) * (img_w * aug_scale - img_w)
            aug_affine = iaa.Affine(scale=aug_scale,
                                    translate_px={'x': int(round(aug_offset_w)), 'y': int(round(aug_offset_h))})
            x_image = aug_affine(image=x_image, return_batch=False)

        # utils.visualize_3d_hmap(x_heatmap[0], np.array(x_image))     # TODO rimuovere
        # plt.imsave(f'out/imgs/img_aug{i}.jpg', x_image)
        # x_image = torch.from_numpy(x_image).type(torch.FloatTensor).permute(2, 0, 1)
        x_image = transforms.ToTensor()(x_image)
        x_image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x_image)

        if self.mode == 'train':
            return x_image, x_heatmap, y_coords3d
        elif self.mode in ('val', 'test'):
            return (x_image, x_heatmap, y_coords3d, y_coords2d, *CAMERA_PARAMS, original_image)

    def generate_3d_heatmap(self, anns, augmentation):
        # type: (List[dict], str) -> Tuple[Tensor, Tuple[float, float, float], Any, Any]
        # augmentation initialization (rescale + crop)

        h, w, d = self.cnf.hmap_h, self.cnf.hmap_w, self.cnf.hmap_d

        aug_scale = np.random.uniform(0.5, 2) if augmentation == 'all' else 1
        aug_h = np.random.uniform(0, 1) if augmentation == 'all' else 0
        aug_w = np.random.uniform(0, 1) if augmentation == 'all' else 0
        aug_offset_h = aug_h * (h * aug_scale - h)
        aug_offset_w = aug_w * (w * aug_scale - w)

        all_hmaps = []
        y_coords3d = []
        y_coords2d = []
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
                if augmentation == 'all':
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

                # ignore the joint if the distance is such that the height of the person is less then about 25 pixel
                z3d_augmented = joint['z3d'] / aug_scale
                if z3d_augmented >= self.cnf.max_distance:
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

                y_coords3d.append([USEFUL_JOINTS.index(jtype)] + [joint['x3d'], joint['y3d'], joint['z3d']])
                y_coords2d.append([USEFUL_JOINTS.index(jtype)] + [
                    int(round(cam_dist)),
                    int(round(joint['y2d'])),
                    int(round(joint['x2d'])),
                ])
            all_hmaps.append(x)
        y_coords3d = json.dumps(y_coords3d)
        y_coords2d = json.dumps(y_coords2d)
        x = torch.cat(tuple([h.unsqueeze(0) for h in all_hmaps]))

        return x, (aug_scale, aug_h, aug_w), y_coords3d, y_coords2d

    def get_frame(self, file_path):
        # read input frame
        frame_path = self.cnf.mot_synth_path / file_path
        frame = utils.imread(frame_path).convert('RGB')
        # frame = transforms.ToTensor()(frame)
        return frame

    @staticmethod
    def get_joints_from_anns(anns, jtype):
        joints = []
        for ann in anns:
            n_visible_joints = np.array([visibility == 2 for visibility in ann['keypoints'][2::3]]).sum()
            # max_y, min_y = max(ann['keypoints'][1::3]), min(ann['keypoints'][1::3])
            # person_height = max_y - min_y
            # z = np.array(ann['keypoints_3d'][2::4]).mean()
            # if 48 < person_height < 52 and jtype == 2:
            #     print(f"person height: {person_height}, z: {z}")
            if n_visible_joints > 22 * 0.25:
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
    from models import Autoencoder
    from test_metrics import joint_det_metrics
    from models import CodePredictor

    cnf = Conf(exp_name='loco_debug')

    # init volumetric heatmap autoencoder
    autoencoder = Autoencoder()
    autoencoder.eval()
    autoencoder.requires_grad(False)
    autoencoder = autoencoder.to(cnf.device)

    code_predictor = CodePredictor(half_images=cnf.half_images)
    code_predictor = code_predictor.to(cnf.device)

    # init Hole Filler
    refiner = Refiner(pretrained=True)
    # refiner.to(cnf.device)
    refiner.eval()
    refiner.requires_grad(False)

    ds = MOTSynthDS(mode='train', cnf=cnf, debug=True)
    loader = DataLoader(dataset=ds, batch_size=1, num_workers=0, shuffle=False)

    ck_path = cnf.exp_log_path / 'training.ck'
    if ck_path.exists():
        ck = torch.load(ck_path, map_location=torch.device('cpu'))
        print(f'[loading checkpoint \'{ck_path}\']')
        code_predictor.load_state_dict(ck['model'])

    for i, sample in enumerate(loader):
        x_2d_image, heatmaps, y = sample

        coords3d_true = json.loads(y[0])

        print(f'({i}) Dataset example: x.shape={tuple(x_2d_image.shape)}, y={y}')

        code_pred = code_predictor.forward(x_2d_image.cuda()).unsqueeze(0)

        autoencoder_heatmaps = autoencoder.decode(code_pred).squeeze()

        # utils.visualize_3d_hmap(autoencoder_heatmaps[0], np.array(x_2d_image_real[0]))

        coords2d_pred = utils.local_maxima_3d(hmaps3d=autoencoder_heatmaps, threshold=0.1, device=cnf.device)

        # rescaled pseudo-3D coordinates --> [to_3d] --> real 3D coordinates
        coords3d_pred = []
        for i in range(len(coords2d_pred)):
            joint_type, cam_dist, y2d, x2d = coords2d_pred[i]
            x2d, y2d, cam_dist = utils.rescale_to_real(x2d=x2d, y2d=y2d, cam_dist=cam_dist, q=cnf.q)
            x3d, y3d, z3d = utils.to3d(x2d=x2d, y2d=y2d, cam_dist=cam_dist, fx=1158, fy=1158, cx=960, cy=540)
            coords3d_pred.append((joint_type, x3d, y3d, z3d))

        # real 3D
        metrics = joint_det_metrics(points_pred=coords3d_pred, points_true=coords3d_true, th=cnf.det_th)
        print()


if __name__ == '__main__':
    main()
