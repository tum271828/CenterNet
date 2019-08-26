import os
import cv2
import pdb
import json
import copy
import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib

from tqdm import tqdm
from config import system_configs
from utils import crop_image, normalize_
from external.nms import soft_nms, soft_nms_merge

colours = np.random.rand(80,3)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def _rescale_dets(detections, ratios, borders, sizes):
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

def save_image(data, fn):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data)
    plt.savefig(fn, dpi = height)
    plt.close()

def kp_detection(db, nnet, filelist, result_dir, debug=False):
    debug_dir = os.path.join(result_dir, "debug")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    num_images = len(filelist)

    K             = db.configs["top_k"]
    ae_threshold  = db.configs["ae_threshold"]
    nms_kernel    = db.configs["nms_kernel"]
    
    scales        = db.configs["test_scales"]
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
    for ind in tqdm(range(0, num_images), ncols=80, desc="locating kps"):
        image_id   = ind
        image_path = filelist[ind]
        image_file = os.path.basename(image_path)
        image      = cv2.imread(image_path)

        top_bboxes[image_id]=nnet.kp_detection(image,db, result_dir,debug)

        if debug:
            # why reload? need?
            image      = cv2.imread(image_path)
            im         = image[:, :, (2, 1, 0)]
            fig, ax    = plt.subplots(figsize=(12, 12)) 
            fig        = ax.imshow(im, aspect='equal')
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            #bboxes = {}
            for j in range(1, categories + 1):
                keep_inds = (top_bboxes[image_id][j][:, -1] >= 0.4)
                cat_name  = db.class_name(j)
                for bbox in top_bboxes[image_id][j][keep_inds]:
                  bbox  = bbox[0:4].astype(np.int32)
                  xmin     = bbox[0]
                  ymin     = bbox[1]
                  xmax     = bbox[2]
                  ymax     = bbox[3]
                  #if (xmax - xmin) * (ymax - ymin) > 5184:
                  ax.add_patch(plt.Rectangle((xmin, ymin),xmax - xmin, ymax - ymin, fill=False, edgecolor= colours[j-1], 
                               linewidth=4.0))
                  ax.text(xmin+1, ymin-3, '{:s}'.format(cat_name), bbox=dict(facecolor= colours[j-1], ec='black', lw=2,alpha=0.5),
                          fontsize=15, color='white', weight='bold')

            debug_file = os.path.join(debug_dir, "{}.jpg".format(image_file))
            plt.savefig(debug_file)
            plt.close()

    result_json = os.path.join(result_dir, "results.raw.json")
    with open(result_json, "w") as f:
        json.dump(top_bboxes, f, cls=NumpyEncoder)

    ''''result_json = os.path.join(result_dir, "results.json")
    detections  = db.convert_to_coco(top_bboxes)
    with open(result_json, "w") as f:
        json.dump(detections, f)

    cls_ids   = list(range(1, categories + 1))
    image_ids = [db.image_ids(ind) for ind in db_inds]
    db.evaluate(result_json, cls_ids, image_ids)''''
    return 0

def testing(db, nnet, filelist,result_dir, debug=False):
    return globals()[system_configs.sampling_function](db, nnet,filelist, result_dir, debug=debug)
