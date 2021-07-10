# -*- coding: utf-8 -*-
# ---------------------

import json
from os import path, makedirs
from typing import *
from typing import Dict, Any

import numpy as np
import torch
import pandas as pd
from pycocotools.coco import COCO as MOT17
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import run_mot_challenge as MOTChallengeEval

import os, errno
import utils
from conf import Conf


class MOT17TestDS(Dataset):
    """
    Dataset composed of pairs (x, y) in which:
    * x: 2D image
    * y: bounding boxes (x, y, w, h)
    """
    gt_dfs: Dict[int, pd.DataFrame]
    tracker_dfs: Dict[int, list]

    def __init__(self, cnf=None):
        # type: (str, Conf, bool) -> None
        """
        :param cnf: configuration object
        """
        self.cnf = cnf

        path_to_anns = path.join(self.cnf.mot_17_path, 'annotations', 'MOT17.json')
        print(f'Annotation path to load: {path_to_anns}')
        self.mot17_ds = MOT17(path_to_anns)

        self.catIds = self.mot17_ds.getCatIds(catNms=['person'])
        self.imgIds = self.mot17_ds.getImgIds(catIds=self.catIds)
        self.imgIds = self._get_ids_loop_by_seq()

        self.seq_to_frame = self._get_seq_to_frame()
        self._mot_track_eval = MOTChallengeEval.MOTChallengeTrackEval(exp_name=self.cnf.exp_name)

        self.training_sequences = [2, 4, 5, 9, 10, 11, 13]
        self.computed_seq_frames = []
        self.tracker_dfs = {seq: [] for seq in self.training_sequences}
        self.gt_dfs = {}
        self._read_gt_dfs()

    def __len__(self):
        # type: () -> int
        return min(self.cnf.mot_17_test_set_len, len(self.imgIds))

    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor]
        # select sequence name and frame number
        img = self.mot17_ds.loadImgs(self.imgIds[i])[0]

        # load corresponding data
        ann_ids = self.mot17_ds.getAnnIds(imgIds=img['id'], catIds=self.catIds, iscrowd=None)
        anns = self.mot17_ds.loadAnns(ann_ids)

        bboxes = []
        for ann in anns:
            if ann['bbox'][3] >= 25:
                bboxes.append({
                    'x': ann['bbox'][0],
                    'y': ann['bbox'][1],
                    'w': ann['bbox'][2],
                    'h': ann['bbox'][3]
                })

        x_image = self._get_frame(file_path=self.cnf.mot_17_path / img['file_name'])

        x_image = np.array(x_image)
        x_original_image = x_image.copy()

        x_image = transforms.ToTensor()(x_image)
        x_image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x_image)

        bboxes = json.dumps(bboxes)
        return x_image, bboxes, x_original_image, img['id']

    def _get_frame(self, file_path):
        # read input frame
        frame_path = self.cnf.mot_synth_path / file_path
        frame = utils.imread(frame_path).convert('RGB')
        return frame

    def _get_seq_to_frame(self):
        seq_to_frame_dict = {}
        for img_id in self.imgIds:
            seq, frame_id = MOT17TestDS.seq_frame_from_image_id(img_id)
            seq_to_frame_dict.setdefault(seq, []).append(frame_id)
        return seq_to_frame_dict

    @staticmethod
    def seq_frame_from_image_id(img_id):
        seq = int(str(img_id)[2:4])
        frame_id = int(str(img_id)[4:])
        return seq, frame_id

    def _get_ids_loop_by_seq(self):
        """
        Re-order the ids array such that the ids are looped over the sequences
        :return:
        """
        seq_dict = {}
        for img_id in self.imgIds:
            seq = int(str(img_id)[2:4])
            # frame_id = int(str(img_id)[4:])
            seq_dict.setdefault(seq, []).append(img_id)

        result_array = []
        sequences = list(seq_dict.keys())
        seq_counters = [len(seq_dict[key]) for key in seq_dict.keys()]
        i = 0
        while i < len(self.imgIds):
            curr_seq = i % len(sequences)
            seq_counters[curr_seq] -= 1
            if seq_counters[curr_seq] < 0:
                del sequences[curr_seq]
                del seq_counters[curr_seq]
                continue
            seq_index = (len(seq_dict[sequences[curr_seq]]) - 1) - seq_counters[curr_seq]
            result_array.append(seq_dict[sequences[curr_seq]][seq_index])
            i += 1
        return result_array

    def _read_gt_dfs(self):
        for i in self.training_sequences:
            file_path = path.join(self._mot_track_eval.dataset_config['GT_FOLDER'], 'MOT17-train',
                                  f'MOT17-{i:02d}-MOTSynth', 'gt', "gt.txt")
            df = pd.read_csv(file_path, header=None)
            self.gt_dfs.update({i: df})

    def _reset_variables(self):
        self.computed_seq_frames = []
        self.tracker_dfs = {seq: [] for seq in self.training_sequences}

    def _generate_ground_truth(self):
        seq_row_index_dict = {}
        for seq, frame_id in self.computed_seq_frames:
            indexes_to_save = self.gt_dfs[seq].index[self.gt_dfs[seq].iloc[:, 0] == frame_id].tolist()
            seq_row_index_dict.setdefault(seq, []).extend(indexes_to_save)

        for seq in seq_row_index_dict.keys():
            self.gt_dfs[seq].drop(set(self.gt_dfs[seq].index.values) - set(seq_row_index_dict[seq]), inplace=True)

            # generate files
            file_path = path.join(self._mot_track_eval.dataset_config['GT_FOLDER'], 'MOT17-train',
                                  f'MOT17-{seq:02d}-MOTSynth',
                                  f'gt_sub_{self.cnf.exp_name}', "gt.txt")
            makedirs(path.dirname(file_path), exist_ok=True)
            print(f"Creating partial ground truth of 128 elements: {file_path}")
            self.gt_dfs[seq].to_csv(file_path, index=False, header=False)

    def _save_tmp_tracker_csv(self):
        for seq in self.training_sequences:
            seq_df = pd.DataFrame(self.tracker_dfs[seq], columns=np.arange(10))
            file_path = path.join(self._mot_track_eval.dataset_config['TRACKERS_FOLDER'], 'MOT17-train', f'SynthDET',
                                  f'{self.cnf.exp_name}', f"MOT17-{seq:02d}-MOTSynth.txt")
            makedirs(path.dirname(file_path), exist_ok=True)
            seq_df.to_csv(file_path, header=False, index=False, mode='w')
        self._reset_variables()

    def add_to_mot17_csv(self, seq_num, frame_id, bb_info):
        self.tracker_dfs[seq_num].append([frame_id, bb_info['bb_id'],
                                          bb_info['bb_left'], bb_info['bb_top'],
                                          bb_info['bb_width'], bb_info['bb_height'],
                                          -1, -1, -1, -1])

    def update_computed_seq_frame(self, seq, frame):
        self.computed_seq_frames.append((seq, frame))

    def get_eval_results(self):
        self._generate_ground_truth()
        # save data to the tmp directory to allow the MOT17 script to compute the metrics
        self._save_tmp_tracker_csv()

        # compute the metrics and output the useful results
        output_res, output_msg = self._mot_track_eval.run_mot_challenge()
        res_HOTA = output_res['MotChallenge2DBox']['SynthDET']['COMBINED_SEQ']['pedestrian']['HOTA']
        return {"re": res_HOTA['DetRe'], "pr": res_HOTA['DetPr'],
                'tp': res_HOTA['HOTA_TP'], 'fp': res_HOTA['HOTA_FP'], 'fn': res_HOTA['HOTA_FN']}


def main():
    import utils
    from models import Autoencoder
    from models import CodePredictor
    from models import Refiner
    from post_processing import joint_association, filter_joints, refine_pose

    cnf = Conf(exp_name='fix_m_debug')
    cnf.data_augmentation = True

    # init volumetric heatmap autoencoder
    autoencoder = Autoencoder()
    autoencoder.eval()
    autoencoder.requires_grad(False)
    autoencoder = autoencoder.to(cnf.device)

    code_predictor = CodePredictor(half_images=False)
    code_predictor = code_predictor.to(cnf.device)

    # init Hole Filler
    refiner = Refiner(pretrained=True)
    # refiner.to(cnf.device)
    refiner.eval()
    refiner.requires_grad(False)

    ds = MOT17TestDS(cnf=cnf)
    loader = DataLoader(dataset=ds, batch_size=1, num_workers=1, shuffle=False)

    ck_path = cnf.exp_log_path / 'training.ck'
    if ck_path.exists():
        ck = torch.load(ck_path, map_location=torch.device('cpu'))
        print(f'[loading checkpoint \'{ck_path}\']')
        code_predictor.load_state_dict(ck['model'])

    # perform model prediction
    for i, sample in enumerate(loader):
        x_2d_image, bboxes_true, x_2d_original_image, img_id = sample

        bboxes_true = json.loads(bboxes_true[0])
        img_id = int(img_id[0])

        seq_num = int(str(img_id)[2:4])
        frame_id = int(str(img_id)[4:])
        ds.update_computed_seq_frame(seq_num, frame_id)
        print(
            f'({i}, seq={seq_num}, frame_id={frame_id}) Dataset example: x.shape={tuple(x_2d_image.shape)}, bboxes_true={bboxes_true}')

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

        filter_joints(coords3d_pred, duplicate_th=0.05)

        # real 3D coordinates --> [association] --> list of poses
        poses = joint_association(coords3d_pred)

        # 3D poses -> [refiner] -> refined 3D poses
        refined_poses = []
        for _pose in poses:
            refined_pose = refine_pose(pose=_pose, refiner=refiner)
            if refined_pose is not None:
                refined_poses.append(refined_pose)

        # Get 2d poses
        refined_2d_poses = []
        for pose_3d in refined_poses:
            refined_2d_poses.append(utils.to2d_by_def(pose_3d, fx=1158, fy=1158, cx=960, cy=540))

        # Get bounding boxes
        bboxes_pred = []
        for j, pose in enumerate(refined_2d_poses):
            pose_transposed = np.transpose(np.array(pose), (1, 0))
            x_values = pose_transposed[0]
            y_values = pose_transposed[1]
            max_x, max_y, min_x, min_y = np.max(x_values), np.max(y_values), np.min(x_values), np.min(y_values)
            bboxes_pred.append([min_x, min_y, (max_x - min_x), (max_y - min_y)])

            # save CSV bbox information
            ds.add_to_mot17_csv(seq_num, frame_id,
                                bb_info={
                                    'bb_id': j,
                                    'bb_left': min_x, 'bb_top': min_y,
                                    'bb_width': max_x - min_x, 'bb_height': max_y - min_y
                                })

        # real 3D
        # metrics = joint_det_metrics(points_pred=coords3d_pred, points_true=coords3d_true, th=cnf.det_th)
        pred_image = utils.draw_bboxes(x_2d_original_image[0].numpy(), bboxes_pred)
        plt.imsave('img_2.png', pred_image)

    results = ds.get_eval_results()
    print(results)


if __name__ == '__main__':
    main()
