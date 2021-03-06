'''
Visualization
Written by Li Jiang
'''

import numpy as np
import mayavi.mlab as mlab
import os, glob, argparse
import torch
import pickle
from operator import itemgetter

COLOR20 = np.array(
        [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
        [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
        [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]])

COLOR40 = np.array(
        [[88,170,108], [174,105,226], [78,194,83], [198,62,165], [133,188,52], [97,101,219], [190,177,52], [139,65,168], [75,202,137], [225,66,129],
        [68,135,42], [226,116,210], [146,186,98], [68,105,201], [219,148,53], [85,142,235], [212,85,42], [78,176,223], [221,63,77], [68,195,195],
        [175,58,119], [81,175,144], [184,70,74], [40,116,79], [184,134,219], [130,137,46], [110,89,164], [92,135,74], [220,140,190], [94,103,39],
        [144,154,219], [160,86,40], [67,107,165], [194,170,104], [162,95,150], [143,110,44], [146,72,105], [225,142,106], [162,83,86], [227,124,143]])

SEMANTIC_IDXS = np.array([0, 1])
SEMANTIC_NAMES = np.array(['background', 'arm'])
CLASS_COLOR = {
    'background': [143, 223, 142],
    'arm': [189, 189, 57]
}
SEMANTIC_IDX2NAME = {0: 'background', 1: 'arm'}


def visualize_pts_rgb(fig, pts, rgb, scale=0.02):
    pxs = pts[:, 0]
    pys = pts[:, 1]
    pzs = pts[:, 2]
    pt_colors = np.zeros((pxs.size, 4), dtype=np.uint8)
    pt_colors[:, 0:3] = rgb
    pt_colors[:, 3] = 255  # transparent

    scalars = np.arange(pxs.__len__())
    points = mlab.points3d(pxs, pys, pzs,  scalars,
                           mode='point',  # point sphere
                           # colormap='Accent',
                           scale_mode='vector',
                           scale_factor=scale,
                           figure=fig)
    points.module_manager.scalar_lut_manager.lut.table = pt_colors


def get_data(self,id,data_type):
    curr_file_name = self.file_names[data_type][id]

    with open(curr_file_name, 'rb') as f:
        x = pickle.load(f)
        return x


def get_coords_color(opt):
    input_file = os.path.join(opt.data_root, opt.dataset, opt.file_name + '.pickle')
    print('Opening: ', input_file)
    assert os.path.isfile(input_file), 'File not exist - {}.'.format(input_file)

    with open(input_file, 'rb') as f:
        x = pickle.load(f)
        if opt.dataset == 'test':
            xyz, rgb,_,__ = x
        else:
            xyz, rgb, label, inst_label = x
    
    rgb = (rgb + 1) * 127.5

    if (opt.task == 'semantic_gt'):
        label = label.astype(np.int)
        label_rgb = np.zeros(rgb.shape)
        label_rgb[label >= 0] = np.array(itemgetter(*SEMANTIC_NAMES[label[label >= 0]])(CLASS_COLOR))
        rgb = label_rgb

    elif (opt.task == 'instance_gt'):
        inst_label = inst_label.astype(np.int)
        print("Instance number: {}".format(inst_label.max() + 1))
        inst_label_rgb = np.zeros(rgb.shape)
        object_idx = (inst_label >= 0)
        inst_label_rgb[object_idx] = COLOR20[inst_label[object_idx] % len(COLOR20)]
        rgb = inst_label_rgb

    elif (opt.task == 'semantic_pred'):
        semantic_file = os.path.join(opt.result_root,opt.dataset, 'semantic', opt.file_name + '.npy')
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(np.int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

    elif (opt.task == 'instance_pred'):
        assert opt.room_split != 'train'
        instance_file = os.path.join(opt.result_root, opt.room_split, opt.room_name + '.txt')
        assert os.path.isfile(instance_file), 'No instance result - {}.'.format(instance_file)
        f = open(instance_file, 'r')
        masks = f.readlines()
        masks = [mask.rstrip().split() for mask in masks]
        inst_label_pred_rgb = np.zeros(rgb.shape)  # np.ones(rgb.shape) * 255 #
        for i in range(len(masks) - 1, -1, -1):
            mask_path = os.path.join(opt.result_root, opt.room_split, masks[i][0])
            assert os.path.isfile(mask_path), mask_path
            if (float(masks[i][2]) < 0.09):
                continue
            mask = np.loadtxt(mask_path).astype(np.int)
            print('{} {}: {} pointnum: {}'.format(i, masks[i], SEMANTIC_IDX2NAME[int(masks[i][1])], mask.sum()))
            inst_label_pred_rgb[mask == 1] = COLOR20[i % len(COLOR20)]
        rgb = inst_label_pred_rgb

    if opt.dataset != 'test':
        sem_valid = (label != -100)
        xyz = xyz[sem_valid]
        rgb = rgb[sem_valid]

    return xyz, rgb



""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='path to the input dataset files', default='dataset/alivev1')
    parser.add_argument('--result_root', help='path to the predicted results', default='exp/alivev1/pointgroup/pointgroup_alive/trial/result/epoch30_nmst0.3_scoret0.09_npointt100')
    #parser.add_argument('--room_name', help='room_name', default='scene0000_00')
    #parser.add_argument('--room_split', help='train / val / test', default='train')
    parser.add_argument('--task', help='input / semantic_gt / semantic_pred / instance_gt / instance_pred', default='semantic_gt')
    parser.add_argument('--dataset', help='train/val/test', default='val')
    parser.add_argument('--file_name', help='enter the file name', default='moving_001_4')
    opt = parser.parse_args()

    xyz, rgb = get_coords_color(opt)

    fig = mlab.figure(figure=None, bgcolor=(1.0, 1.0, 1.0), size=((800, 800)))
    visualize_pts_rgb(fig, xyz, rgb)
    mlab.show()

