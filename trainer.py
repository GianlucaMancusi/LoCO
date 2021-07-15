# -*- coding: utf-8 -*-
# ---------------------

import json
import math
from datetime import datetime
from time import time

import numpy as np
import torch
import cv2
from tensorboardX import SummaryWriter
import torchvision as tv
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import utils
from conf import Conf
from dataset.mot_synth_ds import MOTSynthDS
from dataset.mot_17_ds import MOT17TestDS
from models import Autoencoder
from models import CodePredictor
from models import Refiner
from test_metrics import joint_det_metrics

import matplotlib.pyplot as plt

from post_processing import joint_association, filter_joints, refine_pose


class Trainer(object):

    def __init__(self, cnf):
        # type: (Conf) -> None
        self.cnf = cnf

        # init code predictor
        self.code_predictor = CodePredictor(half_images=self.cnf.half_images)
        self.code_predictor = self.code_predictor.to(self.cnf.device)

        # init volumetric heatmap autoencoder
        self.autoencoder = Autoencoder()
        self.autoencoder.eval()
        self.autoencoder.requires_grad(False)
        self.autoencoder = self.autoencoder.to(self.cnf.device)

        self.refiner = Refiner(pretrained=True)
        self.refiner.eval()
        self.refiner.requires_grad(False)
        self.refiner = self.refiner.to(self.cnf.device)

        # init optimizer
        self.optimizer = optim.Adam(params=self.code_predictor.parameters(), lr=self.cnf.lr)

        # init dataset(s)
        training_set = MOTSynthDS(mode='train', cnf=self.cnf)
        test_set = MOTSynthDS(mode='test', cnf=self.cnf)
        self.test_set_mot17 = MOT17TestDS(cnf=self.cnf)

        # init train/test loader
        self.train_loader = DataLoader(training_set, self.cnf.batch_size, num_workers=self.cnf.n_workers, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=1, num_workers=self.cnf.n_workers, shuffle=False)
        self.test_mot17_loader = DataLoader(self.test_set_mot17, batch_size=1, num_workers=self.cnf.n_workers, shuffle=False)

        # init logging stuffs
        self.log_path = self.cnf.exp_log_path
        print(f'tensorboard --logdir={self.cnf.project_log_path.abspath()}\n')
        self.sw = SummaryWriter(self.log_path)
        self.log_freq = len(self.train_loader)

        # starting values values
        self.epoch = 0
        self.best_test_f1 = None

        # possibly load checkpoint
        self.load_ck()

    def load_ck(self):
        """
        load training checkpoint
        """
        ck_path = self.log_path / 'training.ck'
        if ck_path.exists():
            ck = torch.load(ck_path, map_location=torch.device('cpu'))
            print(f'[loading checkpoint \'{ck_path}\']')
            self.epoch = ck['epoch']
            self.code_predictor.load_state_dict(ck['model'])
            self.best_test_f1 = self.best_test_f1
            self.optimizer.load_state_dict(ck['optimizer'])

    def save_ck(self):
        """
        save training checkpoint
        """
        ck = {
            'epoch': self.epoch,
            'model': self.code_predictor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_test_loss': self.best_test_f1
        }
        torch.save(ck, self.log_path / 'training.ck')

    def train(self):
        """
        train model for one epoch on the Training-Set.
        """
        self.code_predictor.train()
        self.code_predictor.requires_grad(True)

        train_losses = []
        times = []
        start_time = time()
        t = time()
        for step, sample in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            x, heatmap_true, _ = sample
            x, heatmap_true = x.to(self.cnf.device), heatmap_true.to(self.cnf.device)

            code_pred = self.code_predictor.forward(x)

            loss = nn.MSELoss()(code_pred, self.autoencoder.encode(heatmap_true))
            loss.backward()
            train_losses.append(loss.item())

            self.optimizer.step(None)

            # print an incredible progress bar
            progress = (step + 1) / self.cnf.epoch_len
            progress_bar = ('█' * int(50 * progress)) + ('┈' * (50 - int(50 * progress)))
            times.append(time() - t)
            t = time()
            if self.cnf.log_each_step or (not self.cnf.log_each_step and progress == 1):
                print('\r[{}] Epoch {:0{e}d}.{:0{s}d}: │{}│ {:6.2f}% │ Loss: {:.6f} │ ↯: {:5.2f} step/s'.format(
                    datetime.now().strftime("%m-%d@%H:%M"), self.epoch, step + 1,
                    progress_bar, 100 * progress,
                    np.mean(train_losses), 1 / np.mean(times),
                    e=math.ceil(math.log10(self.cnf.epochs)),
                    s=math.ceil(math.log10(self.cnf.epoch_len)),
                ), end='')

            if step >= self.cnf.epoch_len - 1:
                break

        # log average loss of this epoch
        mean_epoch_loss = np.mean(train_losses)  # type: float
        self.sw.add_scalar(tag='train/loss', scalar_value=mean_epoch_loss, global_step=self.epoch)

        # log epoch duration
        print(f' │ T: {time() - start_time:.2f} s')

    def test(self):
        """
        test model on the Validation-Set
        """
        print(f"[MOTSynth] Testing on epoch {self.epoch} started...")

        self.code_predictor.eval()
        self.code_predictor.requires_grad(False)

        t = time()
        test_prs = []
        test_res = []
        test_f1s = []

        test_losses = []

        test_results = {'ground_truth': [], 'prediction': [], 'ground_truth_depth': [], 'prediction_depth': []}
        for step, sample in enumerate(self.test_loader):
            x, heatmap, coords3d_true, coords2d_true, fx, fy, cx, cy, original_image = sample

            x, heatmap = x.to(self.cnf.device), heatmap.to(self.cnf.device)
            fx, fy, cx, cy = fx.item(), fy.item(), cx.item(), cy.item()
            coords3d_true = json.loads(coords3d_true[0])
            coords2d_true = json.loads(coords2d_true[0])

            # image --> [code_predictor] --> code
            code_pred = self.code_predictor.forward(x).unsqueeze(0)

            # code --> [decode] --> hmap(s)
            hmap_pred = self.autoencoder.decode(code_pred).squeeze()

            # utils.visualize_multiple_3d_hmap(hmap_pred)
            # utils.visualize_multiple_3d_hmap(heatmap)

            with torch.no_grad():
                loss = nn.MSELoss()(code_pred, self.autoencoder.encode(heatmap))
                test_losses.append(loss.item())

            # hmap --> [local_maxima_3d] --> rescaled pseudo-3D coordinates
            coords2d_pred = utils.local_maxima_3d(hmaps3d=hmap_pred, threshold=0.1, device=self.cnf.device)

            # rescaled pseudo-3D coordinates --> [to_3d] --> real 3D coordinates
            coords3d_pred = []
            for i in range(len(coords2d_pred)):
                joint_type, cam_dist, y2d, x2d = coords2d_pred[i]
                x2d, y2d, cam_dist = utils.rescale_to_real(x2d=x2d, y2d=y2d, cam_dist=cam_dist, q=self.cnf.q)
                x3d, y3d, z3d = utils.to3d(x2d=x2d, y2d=y2d, cam_dist=cam_dist, fx=fx, fy=fy, cx=cx, cy=cy)
                coords3d_pred.append((joint_type, x3d, y3d, z3d))

            # real 3D
            metrics = joint_det_metrics(points_pred=coords3d_pred, points_true=coords3d_true, th=self.cnf.det_th)
            pr, re, f1 = metrics['pr'], metrics['re'], metrics['f1']
            test_prs.append(pr)
            test_res.append(re)
            test_f1s.append(f1)

            # get images to print on tensorboard
            original_image = original_image[0].numpy()
            plt.imsave(f'out/imgs/img_test{step}_{self.cnf.exp_name}.jpg', original_image)
            if step in list(range(0, self.cnf.test_set_len, self.cnf.test_set_len // 8)):
                ground_truth_image = utils.get_3d_hmap_image(cnf=self.cnf, hmap=heatmap[0], image=original_image,
                                                             coords2d=None, normalize=False)
                pred_image = utils.get_3d_hmap_image(cnf=self.cnf, hmap=hmap_pred, image=original_image, coords2d=None,
                                                     normalize=False)
                #
                ground_truth_depth = utils.get_3d_hmap_image(cnf=self.cnf, hmap=heatmap[0], image=original_image,
                                                             coords2d=coords2d_true, normalize=False)
                pred_image_depth = utils.get_3d_hmap_image(cnf=self.cnf, hmap=hmap_pred, image=original_image,
                                                           coords2d=coords2d_pred, normalize=True)
                #
                test_results['ground_truth'].append(ground_truth_image.transpose(2, 0, 1))
                test_results['prediction'].append(pred_image.transpose(2, 0, 1))
                test_results['ground_truth_depth'].append(ground_truth_depth.transpose(2, 0, 1))
                test_results['prediction_depth'].append(pred_image_depth.transpose(2, 0, 1))
            # ###################################################### #

        # log average loss on test set
        mean_test_pr = float(np.mean(test_prs))
        mean_test_re = float(np.mean(test_res))
        mean_test_f1 = float(np.mean(test_f1s))

        # log average loss
        mean_test_loss = np.mean(test_losses)  # type: float
        self.sw.add_scalar(tag='test/loss', scalar_value=mean_test_loss, global_step=self.epoch)

        # print test metrics
        print(
            f'\t● AVG (PR, RE, F1) on TEST-set: '
            f'({mean_test_pr * 100:.2f}, '
            f'{mean_test_re * 100:.2f}, '
            f'{mean_test_f1 * 100:.2f}) ',
            end=''
        )
        print(f'│ T: {time() - t:.2f} s')

        self.sw.add_scalar(tag='test/precision', scalar_value=mean_test_pr, global_step=self.epoch)
        self.sw.add_scalar(tag='test/recall', scalar_value=mean_test_re, global_step=self.epoch)
        self.sw.add_scalar(tag='test/f1', scalar_value=mean_test_f1, global_step=self.epoch)

        # print images for this test
        grid = torch.cat([torch.Tensor(test_results['ground_truth']),
                          torch.Tensor(test_results['prediction']),
                          torch.Tensor(test_results['ground_truth_depth']),
                          torch.Tensor(test_results['prediction_depth'])], dim=0)
        grid = tv.utils.make_grid(grid, nrow=8)

        self.sw.add_image(tag=f'Visual_Predictions',
                          img_tensor=torch.Tensor(grid).to(torch.uint8)[[2, 1, 0], :, :],
                          global_step=self.epoch, )

        # save best model
        if self.best_test_f1 is None or mean_test_f1 >= self.best_test_f1:
            self.best_test_f1 = mean_test_f1
            torch.save(self.code_predictor.state_dict(), self.log_path / 'best.pth')

    def test_mot17(self):
        """
        test model on the Validation-Set
        """
        print(f"[MOT17] Testing on epoch {self.epoch} started...")

        self.code_predictor.eval()
        self.code_predictor.requires_grad(False)

        t = time()

        test_results = {'ground_truth': [], 'prediction': []}
        for step, sample in enumerate(self.test_mot17_loader):
            x_2d_image, bboxes_true, x_2d_original_image, img_id = sample

            x_2d_image = x_2d_image.to(self.cnf.device)

            bboxes_true = json.loads(bboxes_true[0])
            img_id = int(img_id[0])
            seq_num = int(str(img_id)[2:4])
            frame_id = int(str(img_id)[4:])
            self.test_set_mot17.update_computed_seq_frame(seq_num, frame_id)

            # image --> [code_predictor] --> code
            code_pred = self.code_predictor.forward(x_2d_image).unsqueeze(0)

            # code --> [decode] --> hmap(s)
            autoencoder_heatmaps = self.autoencoder.decode(code_pred).squeeze()

            # hmap --> [local_maxima_3d] --> rescaled pseudo-3D coordinates
            coords2d_pred = utils.local_maxima_3d(hmaps3d=autoencoder_heatmaps, threshold=0.1, device=self.cnf.device)

            # rescaled pseudo-3D coordinates --> [to_3d] --> real 3D coordinates
            coords3d_pred = []
            for i in range(len(coords2d_pred)):
                joint_type, cam_dist, y2d, x2d = coords2d_pred[i]
                x2d, y2d, cam_dist = utils.rescale_to_real(x2d=x2d, y2d=y2d, cam_dist=cam_dist, q=self.cnf.q)
                x3d, y3d, z3d = utils.to3d(x2d=x2d, y2d=y2d, cam_dist=cam_dist, fx=1158, fy=1158, cx=960, cy=540)
                coords3d_pred.append((joint_type, x3d, y3d, z3d))

            filter_joints(coords3d_pred, duplicate_th=0.05)

            # real 3D coordinates --> [association] --> list of poses
            poses = joint_association(coords3d_pred)

            # 3D poses -> [refiner] -> refined 3D poses
            refined_poses = []
            for _pose in poses:
                refined_pose = refine_pose(pose=_pose, refiner=self.refiner, device=self.cnf.device)
                if refined_pose is not None:
                    refined_poses.append(refined_pose)

            # Get 2d poses
            refined_2d_poses = []
            for pose_3d in refined_poses:
                refined_2d_poses.append(utils.to2d(pose_3d, fx=1158, fy=1158, cx=960, cy=540))

            # Get bounding boxes
            bboxes_pred = []
            for j, pose in enumerate(refined_2d_poses):
                x_values = np.transpose(np.array(pose), (1, 0))[0]
                y_values = np.transpose(np.array(pose), (1, 0))[1]
                max_x, max_y, min_x, min_y = np.max(x_values), np.max(y_values), np.min(x_values), np.min(y_values)
                bboxes_pred.append([min_x, min_y, (max_x - min_x), (max_y - min_y)])

                # save CSV bbox information
                self.test_set_mot17.add_to_mot17_csv(seq_num, frame_id,
                                                     bb_info={
                                                         'bb_id': j,
                                                         'bb_left': min_x, 'bb_top': min_y,
                                                         'bb_width': max_x - min_x, 'bb_height': max_y - min_y
                                                     })

            if step in list(range(0, self.cnf.mot_17_test_set_len, self.cnf.mot_17_test_set_len // 8)):
                bboxes_true = [[el['x'], el['y'], el['w'], el['h']] for el in bboxes_true]
                ground_image = utils.draw_bboxes(x_2d_original_image[0].numpy(), bboxes_true)
                pred_image = utils.draw_bboxes(x_2d_original_image[0].numpy(), bboxes_pred)
                ground_image = cv2.resize(ground_image, (1280, 720))
                pred_image = cv2.resize(pred_image, (1280, 720))
                test_results['ground_truth'].append(ground_image.transpose(2, 0, 1))
                test_results['prediction'].append(pred_image.transpose(2, 0, 1))

        results = self.test_set_mot17.get_eval_results()
        # log average loss on test set
        mean_test_pr = float(results['pr'][9])
        mean_test_re = float(results['re'][9])
        tp, fn, fp = np.array(results['tp']), np.array(results['fn']), np.array(results['fp'])
        f1 = 2 * tp / (2 * tp + fn + fp)
        mean_test_f1 = float(f1[9])

        # print test metrics
        print(
            f'\t● AVG (PR, RE, F1) on TEST-set (MOT17): '
            f'({mean_test_pr * 100:.2f}, '
            f'{mean_test_re * 100:.2f}, '
            f'{mean_test_f1 * 100:.2f}) ',
            end=''
        )
        print(f'│ T: {time() - t:.2f} s')

        self.sw.add_scalar(tag='test_mot_17/precision', scalar_value=mean_test_pr, global_step=self.epoch)
        self.sw.add_scalar(tag='test_mot_17/recall', scalar_value=mean_test_re, global_step=self.epoch)
        self.sw.add_scalar(tag='test_mot_17/f1', scalar_value=mean_test_f1, global_step=self.epoch)

        # print images for this test
        grid = torch.cat([torch.Tensor(test_results['ground_truth']),
                          torch.Tensor(test_results['prediction'])], dim=0)
        grid = tv.utils.make_grid(grid, nrow=8)

        self.sw.add_image(tag=f'test_mot_17/Visual_Predictions',
                          img_tensor=torch.Tensor(grid).to(torch.uint8)[[2, 1, 0], :, :],
                          global_step=self.epoch, )

    def run(self):
        """
        start model training procedure (train > test > checkpoint > repeat)
        """
        for _ in range(self.epoch, self.cnf.epochs):
            self.train()
            self.test()
            if self.epoch % 2 == 0 and self.epoch != 0:
                self.test_mot17()
            self.epoch += 1
            if not self.cnf.debug:
                self.save_ck()
