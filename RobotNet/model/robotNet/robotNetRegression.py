import sys

from collections import OrderedDict

import torch
import torch.nn as nn
import spconv
from spconv.modules import SparseModule
import open3d as o3d
import numpy as np

sys.path.append('../../')  # noqa
from lib.pointgroup_ops.functions import pointgroup_ops
from util import utils


class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id)) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id+1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output.features = torch.cat((identity.features, output_decoder.features), dim=1)

            output = self.blocks_tail(output)

        return output


class RobotNetRegression(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        m = cfg.m

        self.fc1_hidden = cfg.fc1_hidden
        self.fc2_hidden = cfg.fc2_hidden
        self.regres_dim = cfg.regres_dim

        self.max_point_lim = cfg.max_point_lim
        self.prepare_epochs = cfg.prepare_epochs

        self.pretrain_path = cfg.pretrain_path
        self.pretrain_module = cfg.pretrain_module
        self.fix_module = cfg.fix_module

        self.regression = nn.Sequential(
            nn.Linear(self.max_point_lim*3, cfg.fc1_hidden),
            nn.ReLU(),
            nn.Linear(cfg.fc1_hidden, cfg.fc1_hidden),
            nn.ReLU(),
            nn.Linear(cfg.fc1_hidden, cfg.fc1_hidden),
            nn.ReLU(),
            nn.Linear(cfg.fc1_hidden, cfg.fc1_hidden),
            nn.ReLU(),
            nn.Linear(cfg.fc1_hidden, cfg.fc2_hidden),
            nn.ReLU(),
            nn.Linear(cfg.fc2_hidden, cfg.fc2_hidden),
            nn.ReLU(),
            nn.Linear(cfg.fc2_hidden, cfg.fc2_hidden),
            nn.ReLU(),
            nn.Linear(cfg.fc2_hidden, cfg.regres_dim),
            # nn.Linear(self.fc1_hidden, self.regres_dim),
        )

        self.apply(self.set_bn_init)

        module_map = {'regression': self.regression}

        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False

        #### load pretrain weights
        if self.pretrain_path is not None:
            pretrain_dict = torch.load(self.pretrain_path)
            for m in self.pretrain_module:
                print("Load pretrained " + m + ": %d/%d" % utils.load_model_param(module_map[m], pretrain_dict, prefix=m))

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, semantic_scores, coords, batch_idxs, feats, batch_offsets, file_names, epoch):

        ret = {}
        semantic_preds = semantic_scores.max(1)[1]  # (N), long

        # Extracting Arm
        pcd = o3d.geometry.PointCloud()
        arm_regress_list = list()
        for ii in range(len(batch_offsets)-1):
            batch_coords = coords[batch_offsets[ii]:batch_offsets[ii+1]]
            batch_feats = feats[batch_offsets[ii]:batch_offsets[ii+1]]
            arm_mask = semantic_preds[batch_offsets[ii]:batch_offsets[ii+1]] == 1
            arm_points = batch_coords[arm_mask]
            arm_colors = batch_feats[arm_mask]
            device = arm_points.device

            # Moving to CPU
            pcd.points = o3d.utility.Vector3dVector(arm_points.cpu().numpy())
            pcd.colors = o3d.utility.Vector3dVector(arm_colors.cpu().numpy())
            vox_pcd = pcd.voxel_down_sample(voxel_size=0.005)
            length = np.asarray(vox_pcd.points).shape[0]

            if length > self.max_point_lim:
                ind = np.random.choice(length, self.max_point_lim, replace=False)
                down_pcd = vox_pcd.select_by_index(ind)

            # we should update
            else:
                arm_regress_list.append(-1)
                continue

            points = np.asarray(down_pcd.points)
            colors = np.asarray(down_pcd.colors)
            # points = points.reshape(-1)

            new_arm_points = torch.from_numpy(points).float().to(device)
            new_arm_colors = torch.from_numpy(colors).float().to(device)

            print('color points')
            print(new_arm_points)
            print(new_arm_points.shape)
            print(new_arm_colors)
            print(new_arm_colors.shape)
            print('color points end')


            # Normalization
            new_arm_points = (new_arm_points - new_arm_points.min()) / (new_arm_points.max() - new_arm_points.min())

            arm_regress = self.regression(new_arm_points)
            arm_regress_list.append(arm_regress)

        ret['arm_regress'] = arm_regress_list

        return ret


def model_fn_decorator(test=False):
    # config
    from util.config import cfg

    # criterion
    regression_criterion = nn.MSELoss(reduction='sum').cuda()
    cos_regression_criterion = nn.CosineSimilarity(dim=0, eps=1e-6)

    def test_model_fn(batch, model, epoch):
        coords = batch['locs'].cuda()              # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()          # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()          # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()  # (N, 3), float32, cuda
        feats = batch['feats'].cuda().float()              # (N, C), float32, cuda
        poses = batch['poses'].cuda().float()
        batch_offsets = batch['offsets'].cuda()    # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda
        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)
        ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets,None ,epoch)
        semantic_scores = ret['semantic_scores']  # (N, nClass) float32, cuda

        ##### preds
        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores

            if (epoch > cfg.prepare_epochs):
                preds['regression'] = ret['arm_regress']

        return preds

    def model_fn(batch, model, epoch):
        semantic_scores = batch['semantics'].cuda()  # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        file_names = batch['file_names']
        batch_offsets = batch['offsets'].cuda()  # (B + 1), int, cuda
        poses = batch['poses'].cuda().float()  # (B,7), float32, cuda
        feats = batch['feats'].cuda().float()  # (N, C), float32, cuda

        coords = batch['locs'].cuda()  # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        coords_float = batch['locs_float'].cuda()  # (N, 3), float32, cuda

        ret = model(
            semantic_scores,
            coords_float,
            coords[:, 0].int(),
            feats,
            batch_offsets,
            file_names,
            epoch
        )

        loss_inp = dict()

        arm_regress = ret['arm_regress']
        loss_inp['arm_regress'] = (arm_regress, poses)
        if epoch > 220 and not isinstance(poses, int):
            print()
            for i, _ in enumerate(file_names):
                if not isinstance(arm_regress[i], int):
                    print(
                        file_names[i],
                        'pred:', tuple(arm_regress[i].tolist()),
                        'gt:', tuple(poses[(i)*7:(i+1)*7].tolist()))
            print()

        loss, loss_out, infos = loss_fn(loss_inp, epoch)

        # accuracy / visual_dict / meter_dict
        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores

            visual_dict = {}
            visual_dict['loss'] = loss
            for k, v in loss_out.items():
                visual_dict[k] = v[0]

            meter_dict = {}
            meter_dict['loss'] = (loss.item(), coords.shape[0])
            for k, v in loss_out.items():
                meter_dict[k] = (float(v[0]), v[1])
        return loss, preds, visual_dict, meter_dict

    def loss_fn(loss_inp, epoch):
        loss_out = {}
        infos = {}
        arm_regress, poses = loss_inp['arm_regress']
        batch_size = len(arm_regress)
        regression_loss = 0
        gamma = 2

        for i in range(batch_size):

            if type(arm_regress[i]) == int:
                continue

            pose = poses[i*7:(i+1) * 7]
            pose_reshaped = pose.reshape(1, 7)
            arm_reshaped = arm_regress[i].reshape(1, 7)
            regression_loss += regression_criterion(
                arm_reshaped[0, :3],
                pose_reshaped[0, :3])
            regression_loss += (gamma * (1. - cos_regression_criterion(
                arm_reshaped[0, 3:].view(-1),
                pose_reshaped[0, 3:].view(-1)
            )))

        if type(regression_loss) == int:
            pose = poses[0*7:(0+1)*7]
            regression_loss = regression_criterion(pose.reshape(1,7),pose.reshape(1,7))

        loss_out['regression_loss'] = (regression_loss, batch_size)

        loss = regression_loss

        return loss, loss_out, infos

    def get_segmented_scores(scores, fg_thresh=1.0, bg_thresh=0.0):
        '''
        :param scores: (N), float, 0~1
        :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
        '''
        fg_mask = scores > fg_thresh
        bg_mask = scores < bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)

        segmented_scores = (fg_mask > 0).float()
        k = 1 / (fg_thresh - bg_thresh)
        b = bg_thresh / (bg_thresh - fg_thresh)
        segmented_scores[interval_mask] = scores[interval_mask] * k + b

        return segmented_scores

    fn = test_model_fn if test else model_fn

    return fn
