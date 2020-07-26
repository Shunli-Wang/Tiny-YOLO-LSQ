from __future__ import division

import os
import argparse

from utils.train_tiny_quan_utils import Quan_train


def parser_argument(parser):
    # Params
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="size of each image batch")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")

    # Model config
    parser.add_argument("--model_def", type=str, default=None, help="path to definition file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")  # as a checkpoint in training
    parser.add_argument("--gpu", type=int, default=0, help="assign a gpu to this porject, start from 0")

    # Eval & Save interval
    parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=2, help="interval evaluations on validation set")
    parser.add_argument("--model_save_path", type=str, default=None)

    # Dataset info
    parser.add_argument("--data_config", type=str, default="config/dac.data", help="path to data config file")
    parser.add_argument("--data_norm", type=bool, default=False, help="/255 for every pixel")

    # Others
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=False, help="allow for multi-scale training")

    # Quantisize Settings:
    parser.add_argument("--quan_weitht_bit", type=int, default=8, help="")
    parser.add_argument("--quan_activation_bit", type=int, default=8, help="")

    # Full Precision model:
    parser.add_argument("--fp_pretrained", type=str, default=None, help="path to full precision model")  # as a init_weight for Quan-Net

    opt = parser.parse_args()
    print(opt)

    return opt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = parser_argument(parser)

    # Re Settings
    opt.name = 'Tiny-YOLO-Release Full-Precision anchor:little '
    opt.hasQ = True

    # Set env GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    Quan_train(opt)

# python train_tiny_qan.py \
#     --gpu 0 \
#     --fp_pretrained 'checkpoints/fp_ckpt/Tiny_yolo_ckpt_90.pth' \
#     --model_save_path 'checkpoints/Quan_model/' \
#     --model_def "config/yolov3-tiny_lsq.cfg" \
#     --quan_weitht_bit 3 \
#     --quan_activation_bit 3
