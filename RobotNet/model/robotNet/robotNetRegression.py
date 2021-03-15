import time
import torch
import torch.nn as nn
import spconv
from spconv.modules import SparseModule
import functools
from collections import OrderedDict
import sys
sys.path.append('../../')  # noqa

import open3d as o3d
import numpy as np

from lib.pointgroup_ops.functions import pointgroup_ops
from util import utils

import ipdb

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


class RobotNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        input_c        = cfg.input_channel
        m              = cfg.m
        classes        = cfg.classes
        block_reps     = cfg.block_reps
        block_residual = cfg.block_residual

        self.fc1_hidden = cfg.fc1_hidden
        self.fc2_hidden = cfg.fc2_hidden
        self.regres_dim = cfg.regres_dim

        self.cluster_radius           = cfg.cluster_radius
        self.cluster_meanActive       = cfg.cluster_meanActive
        self.cluster_shift_meanActive = cfg.cluster_shift_meanActive
        self.cluster_npoint_thre      = cfg.cluster_npoint_thre

        self.score_scale     = cfg.score_scale
        self.score_fullscale = cfg.score_fullscale
        self.mode            = cfg.score_mode
        self.max_point_lim   = cfg.max_point_lim
        self.prepare_epochs  = cfg.prepare_epochs

        self.pretrain_path   = cfg.pretrain_path
        self.pretrain_module = cfg.pretrain_module
        self.fix_module      = cfg.fix_module

        self.regression = nn.Sequential(
            nn.Linear(self.max_point_lim*3, self.fc1_hidden),
            nn.ReLU(),
            nn.Linear(self.fc1_hidden, self.fc2_hidden),
            nn.ReLU(),
            # nn.Linear(self.fc2_hidden, self.fc2_hidden),
            # nn.ReLU(),
            nn.Linear(self.fc2_hidden, self.regres_dim),
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

    def freeze_unet(self):
        for param in self.input_conv.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.output_layer.parameters():
            param.requires_grad = False
        for param in self.linear.parameters():
            param.requires_grad = False

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, input, input_map, coords, batch_idxs, batch_offsets,file_names,epoch):
        ipdb.set_trace()

        #Params: input_, p2v_map,  coords_float, coords[:, 0].int(), batch_offsets, epoch

        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda 
            this is locs_float
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        '''
        ret = {}
        output = self.input_conv(input)

        ##### Extracting Arm 
        if epoch > self.prepare_epochs:
            pcd = o3d.geometry.PointCloud()
            arm_regress_list = list()
            for ii in range(len(batch_offsets)-1):
                #print('Current file is: ', file_names[ii])
                batch_coords = coords[batch_offsets[ii]:batch_offsets[ii+1]]
                arm_mask     = semantic_preds[batch_offsets[ii]:batch_offsets[ii+1]] == 1
                arm_points   = batch_coords[arm_mask]
                device       = arm_points.device

                #Moving to CPU
                pcd.points = o3d.utility.Vector3dVector(arm_points.cpu().numpy())
                #o3d.io.write_point_cloud("./arm.pcd", pcd , write_ascii=False, compressed=False, print_progress=True)
                vox_pcd = pcd.voxel_down_sample(voxel_size=0.005)
                length  = np.asarray(vox_pcd.points).shape[0]

                if length > self.max_point_lim:
                    ind      = np.random.choice(length, self.max_point_lim, replace=False)
                    down_pcd = vox_pcd.select_by_index(ind)
                    
                    #o3d.io.write_point_cloud("./down_pcd.pcd", down_pcd , write_ascii=False, compressed=False, print_progress=True)

                #we should update
                else:
                    #print('Passing the file: ', file_names[ii],' ' ,length)
                    arm_regress_list.append(-1)
                    continue
                    #ind      = np.random.choice(length, self.max_point_lim)
                    #down_pcd = vox_pcd.select_by_index(inds)


                points       = np.asarray(down_pcd.points).reshape(-1)

                new_arm_points  = torch.from_numpy(points).float().to(device)
                arm_regress     = self.regression(new_arm_points)
                arm_regress_list.append(arm_regress)

            ret['arm_regress'] = arm_regress_list

        """
        #Onur: Removing unnessary parts
        #### offset
        pt_offsets_feats = self.offset(output_feats)
        pt_offsets       = self.offset_linear(pt_offsets_feats)   # (N, 3), float32
        """

        """
        #Onur: Removing unnessary parts
        ret['pt_offsets'] = pt_offsets
        """

        """
        #Onur: Removing unncessary parts from the model
        if(epoch > self.prepare_epochs):
            #### get prooposal clusters
            object_idxs = torch.nonzero(semantic_preds > -1).view(-1)

            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = utils.get_batch_offsets(batch_idxs_, input.batch_size)
            coords_ = coords[object_idxs] #Here
            pt_offsets_ = pt_offsets[object_idxs]

            semantic_preds_cpu = semantic_preds[object_idxs].int().cpu()

            idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_shift_meanActive)
            proposals_idx_shift, proposals_offset_shift = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre)
            proposals_idx_shift[:, 1] = object_idxs[proposals_idx_shift[:, 1].long()].int()
            # proposals_idx_shift: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset_shift: (nProposal + 1), int

            idx, start_len = pointgroup_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_meanActive)
            proposals_idx, proposals_offset = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx.cpu(), start_len.cpu(), self.cluster_npoint_thre)
            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
            # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int

            proposals_idx_shift[:, 0] += (proposals_offset.size(0) - 1)
            proposals_offset_shift += proposals_offset[-1]
            proposals_idx = torch.cat((proposals_idx, proposals_idx_shift), dim=0)
            proposals_offset = torch.cat((proposals_offset, proposals_offset_shift[1:]))

            #### proposals voxelization again
            input_feats, inp_map = self.clusters_voxelization(proposals_idx, proposals_offset, output_feats, coords, self.score_fullscale, self.score_scale, self.mode)

            #### score
            score = self.score_unet(input_feats)
            score = self.score_outputlayer(score)
            score_feats = score.features[inp_map.long()] # (sumNPoint, C)
            score_feats = pointgroup_ops.roipool(score_feats, proposals_offset.cuda())  # (nProposal, C)
            scores = self.score_linear(score_feats)  # (nProposal, 1)

            ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)
        """
        return ret


def model_fn_decorator(test=False):
    #### config
    from util.config import cfg

    #### criterion
    weights       = [1, 20.0] #as class distribution
    class_weights = torch.FloatTensor(weights).cuda()

    semantic_criterion   = nn.CrossEntropyLoss(weight=class_weights, ignore_index=cfg.ignore_label).cuda()
    regression_criterion = nn.MSELoss(reduction = 'sum').cuda()
    score_criterion      = nn.BCELoss(reduction='none').cuda()

    def test_model_fn(batch, model, epoch):

        ############ CHECK THIS FOR ADDING THE REGRESSION LOSS
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

        """
        Onur: Removing unncessary parts
        pt_offsets = ret['pt_offsets']            # (N, 3), float32, cuda
        if (epoch > cfg.prepare_epochs):
            scores, proposals_idx, proposals_offset = ret['proposal_scores']
        """

        ##### preds
        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores

            if (epoch > cfg.prepare_epochs):
                preds['regression'] = ret['arm_regress']

            """
            Onur: Removing unnessary parts
            preds['pt_offsets'] = pt_offsets
            """

            #preds['unet_time'] = ret['unet_time']
            """
            Onur: removing unncessary parts
            if (epoch > cfg.prepare_epochs):
                
                preds['score'] = scores
                preds['proposals'] = (proposals_idx, proposals_offset)
            """



        return preds


    def model_fn(batch, model, epoch):
        ##### prepare input and forward
        # batch {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        # 'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
        # 'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
        # 'id': tbl, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}
        coords       = batch['locs'].cuda()        # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map      = batch['p2v_map'].cuda()     # (N), int, cuda
        v2p_map      = batch['v2p_map'].cuda()     # (M, 1 + maxActive), int, cuda
        #p2v_map: N points -> M voxels (Use it with p2v_map.cpu().numpy(), its the combination of #batch_size scenes)
        #v2p_map: M_voxels -> N points (Use it with v2p_map.cpu().numpy(), its the combination of #batch_size scenes)

        file_names      = batch['file_names']
        coords_float    = batch['locs_float'].cuda()      # (N, 3), float32, cuda
        feats           = batch['feats'].cuda().float()   # (N, C), float32, cuda
        poses           = batch['poses'].cuda().float()   # (B,7), float32, cuda
        labels          = batch['labels'].cuda()          # (N), long, cuda
        instance_labels = batch['instance_labels'].cuda() # (N), long, cuda, 0~total_nInst, -100

        instance_info     = batch['instance_info'].cuda()          # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), int, cuda

        batch_offsets = batch['offsets'].cuda()                # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1) #Onur: Feats are just RGB
        #We already get the indices of voxels, now we are doing the real voxelization
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda
        #This just generates the sparse convolution object. The object can be used with sparse convolutions.
        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)
        start1 = time.time()
        ret    = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets,file_names, epoch)
        end1   = time.time() - start1

        semantic_scores = ret['semantic_scores'] # (N, nClass) float32, cuda

        """
        #Onur: Removing unnessary parts
        pt_offsets = ret['pt_offsets']           # (N, 3), float32, cuda
        """


        """
        Onur: Removing unnessary parts from the model
        if(epoch > cfg.prepare_epochs):
            scores, proposals_idx, proposals_offset = ret['proposal_scores']
            # scores: (nProposal, 1) float, cuda
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
        """

        loss_inp = {}
        loss_inp['semantic_scores'] = (semantic_scores, labels)

        if(epoch > cfg.prepare_epochs):
            arm_regress = ret['arm_regress']
            loss_inp['arm_regress']     = (arm_regress, poses)

        """
        #Onur: Removing unnessary parts
        loss_inp['pt_offsets']      = (pt_offsets, coords_float, instance_info, instance_labels)
        """

        """
        Onur: Removing unnessary parts from the model
        if(epoch > cfg.prepare_epochs):
            loss_inp['proposal_scores'] = (scores, proposals_idx, proposals_offset, instance_pointnum)
        """

        loss, loss_out, infos = loss_fn(loss_inp, epoch)

        ##### accuracy / visual_dict / meter_dict
        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores

            """
            Onur: Removing unnessary parts
            preds['pt_offsets'] = pt_offsets
            
            if(epoch > cfg.prepare_epochs):
                preds['score'] = scores
                preds['proposals'] = (proposals_idx, proposals_offset)
            """

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
        infos    = {}

        '''semantic loss'''
        semantic_scores, semantic_labels = loss_inp['semantic_scores']
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda

        if epoch <= cfg.prepare_epochs:
            semantic_loss             = semantic_criterion(semantic_scores, semantic_labels)
            loss_out['semantic_loss'] = (semantic_loss, semantic_scores.shape[0])

        else:
            arm_regress, poses          = loss_inp['arm_regress']
            batch_size  = len(arm_regress)
            regression_loss = 0

            for i in range(batch_size):

                if type(arm_regress[i]) == int: #Continue if this pose has lower than max_lim_point point
                    continue

                pose = poses[i*7:(i+1)*7 ]
                regression_loss += regression_criterion(arm_regress[i].reshape(1,7),pose.reshape(1,7))

            if type(regression_loss) == int:
                pose = poses[0*7:(0+1)*7 ]
                regression_loss = regression_criterion(pose.reshape(1,7),pose.reshape(1,7))


            loss_out['regression_loss'] = (regression_loss, batch_size)

        """
        #Onur: Removing unnessary parts
        '''offset loss'''
        pt_offsets, coords, instance_info, instance_labels = loss_inp['pt_offsets']
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 9), float32 tensor (meanxyz, minxyz, maxxyz)
        # instance_labels: (N), long

        gt_offsets       = instance_info[:, 0:3] - coords   # (N, 3)
        pt_diff          = pt_offsets - gt_offsets   # (N, 3)
        pt_dist          = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
        valid            = (instance_labels != cfg.ignore_label).float()
        offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)   # (N), float
        gt_offsets_     = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
        pt_offsets_     = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff  = - (gt_offsets_ * pt_offsets_).sum(-1)   # (N)
        offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

        loss_out['offset_norm_loss'] = (offset_norm_loss, valid.sum())
        loss_out['offset_dir_loss']  = (offset_dir_loss, valid.sum())
        """

        """
        #Onur: Removing unnessary parts of the model
        if (epoch > cfg.prepare_epochs):
            '''score loss'''
            scores, proposals_idx, proposals_offset, instance_pointnum = loss_inp['proposal_scores']
            # scores: (nProposal, 1), float32
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # instance_pointnum: (total_nInst), int

            ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(), instance_labels, instance_pointnum) # (nProposal, nInstance), float
            gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
            gt_scores = get_segmented_scores(gt_ious, cfg.fg_thresh, cfg.bg_thresh)

            score_loss = score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
            score_loss = score_loss.mean()

            loss_out['score_loss'] = (score_loss, gt_ious.shape[0])
        """

        '''total loss'''


        if(epoch > cfg.prepare_epochs):
            loss =  cfg.loss_weight[1] * regression_loss

        else:
            loss = cfg.loss_weight[0] * semantic_loss

        """
        #Onur: Removing unnessary parts
        loss = cfg.loss_weight[0] * semantic_loss + cfg.loss_weight[1] * offset_norm_loss + cfg.loss_weight[2] * offset_dir_loss
        """


        """
        #Onur: Removing unnessary parts of the model
        if(epoch > cfg.prepare_epochs):
            loss += (cfg.loss_weight[3] * score_loss)
        """

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


    if test:
        fn = test_model_fn
    else:
        fn = model_fn
    return fn
