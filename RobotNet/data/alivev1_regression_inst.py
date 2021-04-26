import os, sys, glob, math
import numpy as np
import scipy.ndimage
import scipy.interpolate
import torch
import pickle
from torch.utils.data import DataLoader

sys.path.append('../')  # noqa

from util.config import cfg
from util.log import logger
from lib.pointgroup_ops.functions import pointgroup_ops

import ipdb


class Dataset:
    def __init__(self, test=False):
        self.data_root = cfg.data_root
        self.dataset = cfg.dataset
        self.filename_suffix = cfg.filename_suffix
        self.batch_size = cfg.batch_size
        self.train_workers = cfg.train_workers
        self.val_workers = cfg.train_workers
        self.full_scale = cfg.full_scale
        self.scale = cfg.scale
        self.max_npoint = cfg.max_npoint
        self.mode = cfg.mode
        self.epoch = 0
        self.prepare_epochs = cfg.prepare_epochs
        self.iteration_cnt = 0
        self.file_cnt = 0
        self.file_names = dict()

        if test:
            self.test_split = cfg.split  # val or test
            self.test_workers = cfg.test_workers
            cfg.batch_size = 1
            self.batch_size = 1

    def trainLoader(self):
        self.train_data_loader = self.get_loader(
                                    'train',
                                    self.trainMerge,
                                    num_workers=self.train_workers,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    drop_last=True
                                )

    def valLoader(self):
        self.val_data_loader = self.get_loader('val', self.valMerge, num_workers=self.val_workers, batch_size=self.batch_size)

    def testLoader(self):
        self.test_data_loader = self.get_loader(self.test_split, self.testMerge, num_workers=self.test_workers)

    def get_loader(self, kw, collate_fn,
                   num_workers=1, batch_size=1, shuffle=False, drop_last=False):

        self.file_names[kw] = glob.glob(os.path.join(self.data_root, self.dataset, kw, '*' + self.filename_suffix))
        self.file_names[kw] = [fn for fn in self.file_names[kw] if fn[-16::] == '_semantic.pickle' and 'dark' not in fn]
        self.file_names[kw].sort()
        file_cnt = len(self.file_names[kw])
        self.batch_cnt = math.ceil(file_cnt / batch_size)
        current_set = list(range(len(self.file_names[kw])))
        return DataLoader(
            current_set,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True
        )

    # Elastic distortion
    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(x).max(0).astype(np.int32)//gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
        interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
        def g(x_):
            return np.hstack([i(x_)[:,None] for i in interp])
        return x + g(x) * mag

    def getInstanceInfo(self, xyz, instance_label):
        '''
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~nInst-1, -100)
        :return: instance_num, dict
        '''
        instance_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0   # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_pointnum = []   # (nInst), int
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)

            ### instance_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            instance_info_i = instance_info[inst_idx_i]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = min_xyz_i
            instance_info_i[:, 6:9] = max_xyz_i
            instance_info[inst_idx_i] = instance_info_i

            ### instance_pointnum
            instance_pointnum.append(inst_idx_i[0].size)

        return instance_num, {"instance_info": instance_info, "instance_pointnum": instance_pointnum}

    def dataAugment(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        return np.matmul(xyz, m)

    def crop(self, xyz):
        '''
        :param xyz: (n, 3) >= 0
        '''
        xyz_offset = xyz.copy()
        valid_idxs = (xyz_offset.min(1) >= 0)
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.max_npoint):
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs

    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def get_data(self, id, data_type):
        curr_file_name = self.file_names[data_type][id]
        filehandler = open(curr_file_name, 'rb')
        x = pickle.load(filehandler, encoding='bytes')
        filehandler.close()

        with open(curr_file_name.replace('_semantic', ''), 'rb') as fp:
            original_data = pickle.load(fp, encoding='bytes')

        return x, original_data, curr_file_name

    def merge(self, kw, id):
        semantics_scores = list()
        file_names = list()
        poses = list()
        locs = list()
        locs_float = list()
        batch_offsets = [0]

        self.iteration_cnt += 1
        self.epoch = math.ceil(self.iteration_cnt/self.batch_cnt)

        for i, idx in enumerate(id):
            semantics, (xyz_origin, rgb, label, instance_label, pose), file_name = self.get_data(idx, kw)

            xyz_middle = self.dataAugment(xyz_origin)
            xyz = xyz_middle
            xyz -= xyz.min(0)
            xyz, valid_idxs = self.crop(xyz)

            xyz_middle = xyz_middle[valid_idxs]
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
            semantics = torch.from_numpy(semantics)
            semantics_scores.append(semantics)
            # print(type(semantics), semantics.shape)

            # merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + semantics.shape[0])
            file_names.append(file_name)
            poses.append(torch.from_numpy(np.array(pose, dtype=np.float32)))
            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))

        # merge all the scenes in the batchd
        semantics_scores = torch.cat(semantics_scores, 0)
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)
        locs = torch.cat(locs, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        poses = torch.cat(poses, 0)

        return {
            'semantics': semantics_scores,
            'locs': locs,
            'locs_float': locs_float,
            'id': id,
            'offsets': batch_offsets,
            'poses': poses,
            'file_names': file_names
        }

    def trainMerge(self, id):
        return self.merge('train', id)

    def valMerge(self, id):
        return self.merge('val', id)

    def testMerge(self, id):
        return self.merge('test', id)
