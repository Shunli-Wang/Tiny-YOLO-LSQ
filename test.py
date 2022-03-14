from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import argparse
import tqdm
import yaml
from utils.qan import find_modules_to_quantize, replace_module_by_names

import torch
from torch.autograd import Variable

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, norm_flag):
    model.eval()

    # Get dataloader in eval situation.
    # Note that in this case, we don't need data augment and multiscale options.
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False, norm_flag=norm_flag)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8, 
        collate_fn=dataset.collate_fn
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

        with torch.no_grad():
            outputs = model(imgs)  # 8*10647*17 (4 position + 1 confidence + 12 class)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    if len(sample_metrics) == 0:
        return np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), 0

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels, IoU_total = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    # Calculate IoU:cb
    IoU_total_ = IoU_total.sum() / IoU_total.shape[0]

    return precision, recall, AP, f1, ap_class, IoU_total_

def parser_argument(parser):
    # Model config
    parser.add_argument("--model_def", type=str, default='config/yolov3-tiny.cfg', help="path to model cfg file")
    parser.add_argument("--weights_path", type=str, help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/dac.names", help="path to class label file")
    parser.add_argument("--quan_yaml", type=str, default=None, help="pass a yaml file to activate quan training")  # as a init_weight for Quan-Net
    # Params
    parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--gpu", type=int, default=0, help="assign a gpu to this porject, start from 0")
    # Dataset info
    parser.add_argument("--data_config", type=str, default="config/dac.data", help="path to data config file")
    parser.add_argument("--data_norm", type=bool, default=False, help="/255 for every pixel")
    # Thres
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.1, help="iou thresshold for non-maximum suppression")

    cfg = parser.parse_args()

    return cfg

def create_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiate model
    model = Darknet(cfg.model_def, cfg).to(device)

    return model, device

def quantization(model, cfg):
    # fp-model
    if cfg.quan_yaml is None:
        return model

    # q-model
    with open(cfg.quan_yaml, 'r') as load_f:
        qcfg = yaml.load(load_f, Loader=yaml.FullLoader)

    # Quantize the whole model
    modules_to_replace = find_modules_to_quantize(model, qcfg)
    model = replace_module_by_names(model, modules_to_replace)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cfg = parser_argument(parser)

    model, device = create_model(cfg)
    model = quantization(model, cfg).to(device)

    # Load weights
    if cfg.weights_path.endswith(".pth"):
        model.load_state_dict(torch.load(cfg.weights_path))
        print('+++ Weights loaded from:', cfg.weights_path)

    data_config = parse_data_config(cfg.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class, IoU_total = evaluate(
        model,
        path=valid_path,
        iou_thres=cfg.iou_thres,
        conf_thres=cfg.conf_thres,
        nms_thres=cfg.nms_thres,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        norm_flag=cfg.data_norm
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}, IoU: {IoU_total}")
