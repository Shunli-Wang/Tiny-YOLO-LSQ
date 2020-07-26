from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, norm_flag):
    model.eval()

    # Get dataloader in eval situation.
    # Note that in this case, we don't need data augment and multiscale options.
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False, norm_flag=norm_flag)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred, IoU)
    for batch_i, (paths, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()

        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2] = targets[:, 2] * 512
        targets[:, 3] = targets[:, 3] * 288
        targets[:, 4] = targets[:, 4] * 512
        targets[:, 5] = targets[:, 5] * 288

        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        # np.save('Layer_outputs/' + 'gt_' + str(targets.shape) + '_.npy', targets)
        # np.save('Layer_outputs/' + 'path' + '_.npy', list(paths))
        # para = np.load('Layer_outputs/path_.npy')

        with torch.no_grad():
            outputs = model(imgs)  # 8*10647*17(4position + 1confidence + 12class)
            print(outputs.shape)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

            for output in outputs:
                if output is not None:
                    # One image two objects
                    print(output.shape, output[:, 4])

            print()

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
    print('sample_metrics:', sample_metrics)

    if len(sample_metrics) == 0:
        return np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), 0

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels, IoU_total = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    # Calculate IoU:cb
    IoU_total_ = IoU_total.sum() / IoU_total.shape[0]
    print('IoU_total_', IoU_total_)

    return precision, recall, AP, f1, ap_class, IoU_total_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/dac.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/dac.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.1, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")

    opt = parser.parse_args()
    print(opt)
    opt.weights_path = 'checkpoints/yolov3-tiny_ckpt_99.pth'  # 'weights/darknet53.conv.74'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
