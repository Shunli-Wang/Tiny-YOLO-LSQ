from __future__ import division
from email.policy import default
import os
import time
import yaml
import datetime
import argparse

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate
from utils.qan import find_modules_to_quantize, replace_module_by_names
from terminaltables import AsciiTable

import torch
from torch.autograd import Variable

metrics = ["grid_size", "loss", "x", "y", "w", "h", "conf", "cls", "cls_acc", "recall50", "recall75", "precision", "conf_obj", "conf_noobj"]

def parser_argument(parser):
    # Model config
    parser.add_argument("--model_def", type=str, default='config/yolov3-tiny.cfg', help="path to model cfg file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")  # as a checkpoint in training

    # Quantisize Settings:
    parser.add_argument("--fp_pretrained", type=str, default=None, help="path to full precision model")  # as a init_weight for Quan-Net
    parser.add_argument("--quan_yaml", type=str, default=None, help="pass a yaml file to activate quan training")  # as a init_weight for Quan-Net

    # Params
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="size of each image batch")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--gpu", type=int, default=0, help="assign a gpu to this porject, start from 0")
    # Dataset info
    parser.add_argument("--data_config", type=str, default="config/dac.data", help="path to data config file")
    parser.add_argument("--data_norm", type=bool, default=False, help="/255 for every pixel")
    # Eval & Save interval
    parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=2, help="interval evaluations on validation set")
    parser.add_argument("--exp_path", type=str)  # !
    # Others
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=False, help="allow for multi-scale training")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")

    cfg = parser.parse_args()

    return cfg

def create_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiate model
    model = Darknet(cfg.model_def, cfg).to(device)
    model.apply(weights_init_normal)

    # Print the whole model
    # for child in model.children():
    #     print(child)

    # Load Pre-Trained model
    if cfg.pretrained_weights:
        if cfg.pretrained_weights.endswith(".pth"):
            checkpoint = torch.load(cfg.pretrained_weights)
            model.load_state_dict(checkpoint)
            print('++++ Load all params to the model.')
        import re
        # !!! This is supposed to be a re-command to extract the epoch num from pre-train model. But it hasn't been establised.
        start_epoch = 0  # 1 + int(re.sub("\D", '', cfg.pretrained_weights)[-10:])
        # return model, device, start_epoch
    else:
        start_epoch = 0

    # Print and save model name & prams
    with open(cfg.exp_path + 'model_params.txt', 'a') as f:
        for name, param in model.named_parameters():
            print(str(name) + '\t\t' + str(param.requires_grad) + '\t\t' + str(param.shape))
            f.write(str(name) + '\t\t' + str(param.requires_grad) + '\t\t' + str(param.shape) + '\n')

    return model, device, start_epoch

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

    print('\n\n')
    # Print and save model name & prams
    with open(cfg.exp_path + 'model_params.txt', 'a') as f:
        print('\nAfter quantization: \n')
        f.write('\nAfter quantization: \n')
        for name, param in model.named_parameters():
            print(str(name) + '\t\t' + str(param.requires_grad) + '\t\t' + str(param.shape))
            f.write(str(name) + '\t\t' + str(param.requires_grad) + '\t\t' + str(param.shape) + '\n')

    return model

def create_dataset(cfg):
    # Get data configuration
    data_config = parse_data_config(cfg.data_config)    # config/dac.data
    train_path = data_config["train"]                   # ./data/train.txt
    valid_path = data_config["valid"]                   # ./data/val.txt
    class_names = load_classes(data_config["names"])    # ./data/dac.names

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=cfg.multiscale_training, norm_flag=cfg.data_norm)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    return dataloader, train_path, valid_path, class_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cfg = parser_argument(parser)

    # Set env GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)

    # Create exp file:
    os.makedirs(cfg.exp_path, exist_ok=True)

    #### Init Step 1: Create Model
    model, device, start_epoch = create_model(cfg)
    model = quantization(model, cfg).to(device)

    #### Init Step 2: Create Dataset
    dataloader, train_path, valid_path, class_names = create_dataset(cfg)

    #### Init Step 3: Create Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training
    for epoch in range(start_epoch, cfg.epochs):
        # Set model in train.
        model.train()

        start_time = time.time()

        #### Epoch Step 1: Train
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            # Load input and target
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            # Train Step 1: Forward pass, get loss
            loss, outputs = model(imgs, targets)

            # Train Step 2: Backward pass, get gradient
            loss.backward()

            # Train Step 3: Optimize params
            if batches_done % cfg.gradient_accumulations:  # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            print('\nEpoch:', epoch, '   Process:', batch_i, len(dataloader), '   Loss:', loss.cpu().data)

            # ----------------
            #   Log 
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, cfg.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j + 1}", metric)]

                tensorboard_log += [("loss", loss.item())]
                # logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        #### Epoch Step 2: Save
        if epoch % cfg.checkpoint_interval == 0:
            torch.save(model.state_dict(), cfg.exp_path + cfg.exp_path.split('/')[1] + f"_%d.pth" % epoch)

        #### Epoch Step 3: Eval
        if epoch % cfg.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class, IoU_total = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=cfg.img_size,
                batch_size=cfg.batch_size,
                norm_flag=cfg.data_norm
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean())
            ]
            # logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            with open(cfg.exp_path + "model_acc_rec.txt", "a+") as f:
                f.write('\n\n----------' + str(epoch) + '----------\n')
                f.write(AsciiTable(ap_table).table)
                f.write(f"---- mAP {AP.mean()}")
                f.write(f"---- IoU {IoU_total}")

                print(AsciiTable(ap_table).table)
                print(f"---- mAP {AP.mean()}")
