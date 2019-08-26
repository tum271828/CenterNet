import os
import pdb
import torch
import importlib
import torch.nn as nn
import copy
import numpy as np
import cv2

from config import system_configs
from models.py_utils.data_parallel import DataParallel
from utils import crop_image, normalize_
from external.nms import soft_nms, soft_nms_merge

torch.manual_seed(317)

class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()

        self.model = model
        self.loss  = loss

    def forward(self, xs, ys, **kwargs):
        preds = self.model(*xs, **kwargs)
        loss_kp  = self.loss(preds, ys, **kwargs)
        return loss_kp

# for model backward compatibility
# previously model was wrapped by DataParallel module
class DummyModule(nn.Module):
    def __init__(self, model):
        super(DummyModule, self).__init__()
        self.module = model

    def forward(self, *xs, **kwargs):
        return self.module(*xs, **kwargs)

class NetworkFactory(object):
    def __init__(self, db):
        super(NetworkFactory, self).__init__()

        module_file = "models.{}".format(system_configs.snapshot_name)
        print("module_file: {}".format(module_file))
        nnet_module = importlib.import_module(module_file)

        self.model   = DummyModule(nnet_module.model(db))
        self.loss    = nnet_module.loss
        self.network = Network(self.model, self.loss)
        self.network = DataParallel(self.network, chunk_sizes=system_configs.chunk_sizes).cuda()

        total_params = 0
        for params in self.model.parameters():
            num_params = 1
            for x in params.size():
                num_params *= x
            total_params += num_params
        print("total parameters: {}".format(total_params))

        if system_configs.opt_algo == "adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters())
            )
        elif system_configs.opt_algo == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate, 
                momentum=0.9, weight_decay=0.0001
            )
        else:
            raise ValueError("unknown optimizer")

    def cuda(self):
        self.model.cuda()

    def train_mode(self):
        self.network.train()

    def eval_mode(self):
        self.network.eval()

    def train(self, xs, ys, **kwargs):
        xs = [x for x in xs]
        ys = [y for y in ys]

        self.optimizer.zero_grad()
        loss_kp = self.network(xs, ys)
        loss        = loss_kp[0]
        focal_loss  = loss_kp[1]
        pull_loss   = loss_kp[2]
        push_loss   = loss_kp[3]
        regr_loss   = loss_kp[4]
        loss        = loss.mean()
        focal_loss  = focal_loss.mean()
        pull_loss   = pull_loss.mean()
        push_loss   = push_loss.mean()
        regr_loss   = regr_loss.mean()
        loss.backward()
        self.optimizer.step()
        return loss, focal_loss, pull_loss, push_loss, regr_loss

    def validate(self, xs, ys, **kwargs):
        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            ys = [y.cuda(non_blocking=True) for y in ys]

            loss_kp = self.network(xs, ys)
            loss       = loss_kp[0]
            focal_loss = loss_kp[1]
            pull_loss  = loss_kp[2]
            push_loss  = loss_kp[3]
            regr_loss  = loss_kp[4]
            loss = loss.mean()
            return loss

    def test(self, xs, **kwargs):
        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            return self.model(*xs, **kwargs)

    def set_lr(self, lr):
        print("setting learning rate to: {}".format(lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def load_pretrained_params(self, pretrained_model):
        print("loading from {}".format(pretrained_model))
        with open(pretrained_model, "rb") as f:
            params = torch.load(f)
            self.model.load_state_dict(params)

    def load_params(self, iteration):
        cache_file = system_configs.snapshot_file.format(iteration)
        print("loading model from {}".format(cache_file))
        with open(cache_file, "rb") as f:
            params = torch.load(f)
            self.model.load_state_dict(params)

    def save_params(self, iteration):
        cache_file = system_configs.snapshot_file.format(iteration)
        print("saving model to {}".format(cache_file))
        with open(cache_file, "wb") as f:
            params = self.model.state_dict()
            torch.save(params, f) 

    def _rescale_dets(self,detections, ratios, borders, sizes):
        xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
        xs    /= ratios[:, 1][:, None, None]
        ys    /= ratios[:, 0][:, None, None]
        xs    -= borders[:, 2][:, None, None]
        ys    -= borders[:, 0][:, None, None]
        tx_inds = xs[:,:,0] <= -5
        bx_inds = xs[:,:,1] >= sizes[0,1]+5
        ty_inds = ys[:,:,0] <= -5
        by_inds = ys[:,:,1] >= sizes[0,0]+5
        
        np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
        np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)
        detections[:,tx_inds[0,:],4] = -1
        detections[:,bx_inds[0,:],4] = -1
        detections[:,ty_inds[0,:],4] = -1
        detections[:,by_inds[0,:],4] = -1

    def kp_decode(self, images, K, ae_threshold=0.5, kernel=3):
        detections, center = self.test([images], ae_threshold=ae_threshold, K=K, kernel=kernel)
        detections = detections.data.cpu().numpy()
        center = center.data.cpu().numpy()
        return detections, center

    def kp_detection(self, image,db, result_dir, debug=False):
        K             = db.configs["top_k"]
        ae_threshold  = db.configs["ae_threshold"]
        nms_kernel    = db.configs["nms_kernel"]        
        weight_exp    = db.configs["weight_exp"]
        merge_bbox    = db.configs["merge_bbox"]
        categories    = db.configs["categories"]
        nms_threshold = db.configs["nms_threshold"]
        max_per_image = db.configs["max_per_image"]
        nms_algorithm = {
            "nms": 0,
            "linear_soft_nms": 1, 
            "exp_soft_nms": 2
        }[db.configs["nms_algorithm"]]
        top_bboxes = {}
        if True:
            #db_ind = db_inds[ind]
            image_id   = 0
            height, width = image.shape[0:2]

            detections = []
            center_points = []

            if True:
                scale=1
                new_height = int(height * scale)
                new_width  = int(width * scale)
                new_center = np.array([new_height // 2, new_width // 2])

                inp_height = new_height | 127
                inp_width  = new_width  | 127

                images  = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
                ratios  = np.zeros((1, 2), dtype=np.float32)
                borders = np.zeros((1, 4), dtype=np.float32)
                sizes   = np.zeros((1, 2), dtype=np.float32)

                out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
                height_ratio = out_height / inp_height
                width_ratio  = out_width  / inp_width

                resized_image = cv2.resize(image, (new_width, new_height))
                resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width])

                resized_image = resized_image / 255.
                normalize_(resized_image, db.mean, db.std)

                images[0]  = resized_image.transpose((2, 0, 1))
                borders[0] = border
                sizes[0]   = [int(height * scale), int(width * scale)]
                ratios[0]  = [height_ratio, width_ratio]       

                images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
                images = torch.from_numpy(images)
                dets, center = self.kp_decode(images, K, ae_threshold=ae_threshold, kernel=nms_kernel)
                dets   = dets.reshape(2, -1, 8)
                center = center.reshape(2, -1, 4)
                dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
                center[1, :, [0]] = out_width - center[1, :, [0]]
                dets   = dets.reshape(1, -1, 8)
                center   = center.reshape(1, -1, 4)
                
                self._rescale_dets(dets, ratios, borders, sizes)
                center [...,[0]] /= ratios[:, 1][:, None, None]
                center [...,[1]] /= ratios[:, 0][:, None, None] 
                center [...,[0]] -= borders[:, 2][:, None, None]
                center [...,[1]] -= borders[:, 0][:, None, None]
                np.clip(center [...,[0]], 0, sizes[:, 1][:, None, None], out=center [...,[0]])
                np.clip(center [...,[1]], 0, sizes[:, 0][:, None, None], out=center [...,[1]])
                dets[:, :, 0:4] /= scale
                center[:, :, 0:2] /= scale

                if scale == 1:
                    center_points.append(center)
                detections.append(dets)

            detections = np.concatenate(detections, axis=1)
            center_points = np.concatenate(center_points, axis=1)

            classes    = detections[..., -1]
            classes    = classes[0]
            detections = detections[0]
            center_points = center_points[0]
            
            valid_ind = detections[:,4]> -1
            valid_detections = detections[valid_ind]
            
            box_width = valid_detections[:,2] - valid_detections[:,0]
            box_height = valid_detections[:,3] - valid_detections[:,1]
            
            s_ind = (box_width*box_height <= 22500)
            l_ind = (box_width*box_height > 22500)
            
            s_detections = valid_detections[s_ind]
            l_detections = valid_detections[l_ind]
            
            s_left_x = (2*s_detections[:,0] + s_detections[:,2])/3
            s_right_x = (s_detections[:,0] + 2*s_detections[:,2])/3
            s_top_y = (2*s_detections[:,1] + s_detections[:,3])/3
            s_bottom_y = (s_detections[:,1]+2*s_detections[:,3])/3
            
            s_temp_score = copy.copy(s_detections[:,4])
            s_detections[:,4] = -1
            
            center_x = center_points[:,0][:, np.newaxis]
            center_y = center_points[:,1][:, np.newaxis]
            s_left_x = s_left_x[np.newaxis, :]
            s_right_x = s_right_x[np.newaxis, :]
            s_top_y = s_top_y[np.newaxis, :]
            s_bottom_y = s_bottom_y[np.newaxis, :]
            
            ind_lx = (center_x - s_left_x) > 0
            ind_rx = (center_x - s_right_x) < 0
            ind_ty = (center_y - s_top_y) > 0
            ind_by = (center_y - s_bottom_y) < 0
            ind_cls = (center_points[:,2][:, np.newaxis] - s_detections[:,-1][np.newaxis, :]) == 0
            ind_s_new_score = np.max(((ind_lx+0) & (ind_rx+0) & (ind_ty+0) & (ind_by+0) & (ind_cls+0)), axis = 0) == 1
            index_s_new_score = np.argmax(((ind_lx+0) & (ind_rx+0) & (ind_ty+0) & (ind_by+0) & (ind_cls+0))[:,ind_s_new_score], axis = 0)
            s_detections[:,4][ind_s_new_score] = (s_temp_score[ind_s_new_score]*2 + center_points[index_s_new_score,3])/3
        
            l_left_x = (3*l_detections[:,0] + 2*l_detections[:,2])/5
            l_right_x = (2*l_detections[:,0] + 3*l_detections[:,2])/5
            l_top_y = (3*l_detections[:,1] + 2*l_detections[:,3])/5
            l_bottom_y = (2*l_detections[:,1]+3*l_detections[:,3])/5
            
            l_temp_score = copy.copy(l_detections[:,4])
            l_detections[:,4] = -1
            
            center_x = center_points[:,0][:, np.newaxis]
            center_y = center_points[:,1][:, np.newaxis]
            l_left_x = l_left_x[np.newaxis, :]
            l_right_x = l_right_x[np.newaxis, :]
            l_top_y = l_top_y[np.newaxis, :]
            l_bottom_y = l_bottom_y[np.newaxis, :]
            
            ind_lx = (center_x - l_left_x) > 0
            ind_rx = (center_x - l_right_x) < 0
            ind_ty = (center_y - l_top_y) > 0
            ind_by = (center_y - l_bottom_y) < 0
            ind_cls = (center_points[:,2][:, np.newaxis] - l_detections[:,-1][np.newaxis, :]) == 0
            ind_l_new_score = np.max(((ind_lx+0) & (ind_rx+0) & (ind_ty+0) & (ind_by+0) & (ind_cls+0)), axis = 0) == 1
            index_l_new_score = np.argmax(((ind_lx+0) & (ind_rx+0) & (ind_ty+0) & (ind_by+0) & (ind_cls+0))[:,ind_l_new_score], axis = 0)
            l_detections[:,4][ind_l_new_score] = (l_temp_score[ind_l_new_score]*2 + center_points[index_l_new_score,3])/3
            
            detections = np.concatenate([l_detections,s_detections],axis = 0)
            detections = detections[np.argsort(-detections[:,4])] 
            classes   = detections[..., -1]
                    
            keep_inds  = (detections[:, 4] > -1)
            detections = detections[keep_inds]
            classes    = classes[keep_inds]

            top_bboxes[image_id] = {}
            for j in range(categories):
                keep_inds = (classes == j)
                top_bboxes[image_id][j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
                if merge_bbox:
                    soft_nms_merge(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=nms_algorithm, weight_exp=weight_exp)
                else:
                    soft_nms(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=nms_algorithm)
                top_bboxes[image_id][j + 1] = top_bboxes[image_id][j + 1][:, 0:5]

            scores = np.hstack([
                top_bboxes[image_id][j][:, -1] 
                for j in range(1, categories + 1)
            ])
            if len(scores) > max_per_image:
                kth    = len(scores) - max_per_image
                thresh = np.partition(scores, kth)[kth]
                for j in range(1, categories + 1):
                    keep_inds = (top_bboxes[image_id][j][:, -1] >= thresh)
                    top_bboxes[image_id][j] = top_bboxes[image_id][j][keep_inds]

            return top_bboxes[image_id]

        return 0

