from __future__ import division
from models import *
# from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate
from terminaltables import AsciiTable
import os
import time
import datetime
import torch
from torch.autograd import Variable

metrics = ["grid_size", "loss", "x", "y", "w", "h", "conf", "cls", "cls_acc", "recall50", "recall75", "precision", "conf_obj", "conf_noobj"]


def create_model(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiate model
    model = Darknet(opt.model_def, opt).to(device)
    model.apply(weights_init_normal)

    # Print the whole model
    for child in model.children():
        print(child)

    # Load Pre-Trained model
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            checkpoint = torch.load(opt.pretrained_weights)
            model.load_state_dict(checkpoint)
        import re
        # !!! This is supposed to be a re-command to extract the epoch num from pre-train model. But it hasn't been establised.
        start_epoch = 0  # 1 + int(re.sub("\D", '', opt.pretrained_weights)[-10:])
        # return model, device, start_epoch
    else:
        start_epoch = 0

    # Print and save model name & prams
    with open(opt.model_save_path + 'model.txt', 'a') as f:
        for name, param in model.named_parameters():
            print(str(name) + '\t\t' + str(param.requires_grad) + '\t\t' + str(param.shape))
            f.write(str(name) + '\t\t' + str(param.requires_grad) + '\t\t' + str(param.shape) + '\n')

    # Load full precision model (Restore all weights)
    if opt.fp_pretrained is not None:

        quan_layers = [0, 2, 4, 6, 8, 10, 14, 18, 21, 15, 22]  # [2, 4, 6]
        quan_layers_bias = [15, 22]
        import re
        fullp_model = torch.load(opt.fp_pretrained)
        layers_list = list(fullp_model.keys())
        for name in layers_list:
            # Search conv layer
            conv_layer = re.findall(r"conv_\d+.weight", name)

            if len(conv_layer) == 1:
                layer_num = int(re.findall(r"\d+", conv_layer[0])[0])

                # LSQ-Quantize layer load fullp conv layers
                if layer_num in quan_layers:

                    # 1.Get conv weights and bias
                    conv_weight = fullp_model[name]
                    del fullp_model[name]
                    # conv with bias
                    if layer_num in quan_layers_bias:
                        conv_bias = fullp_model[name[:-6] + 'bias']
                        del fullp_model[name[:-6] + 'bias']

                    # 2.Sub conv as Qconv name
                    name = re.sub(r"conv_\d+", r"Q_conv_" + str(layer_num), name)

                    # 3.Add conv weights and bias
                    fullp_model[name] = conv_weight
                    if layer_num in quan_layers_bias:
                        # add bias
                        fullp_model[name[:-6] + 'bias'] = conv_bias

                    # add activation scale
                    fullp_model[name[:-6] + 'quan_a.s'] = torch.ones(1)
                    # add weight scale
                    fullp_model[name[:-6] + 'quan_w.s'] = torch.Tensor([conv_weight.abs().mean()])

        model.load_state_dict(fullp_model, strict=not opt.hasQ)  # , strict=False

    return model, device, start_epoch


def create_dataset(opt):
    # Get data configuration
    data_config = parse_data_config(opt.data_config)  # config/dac.data
    train_path = data_config["train"]  # /home/fair/Dataset/_Detection/DAC_2020_data_training/train.txt
    valid_path = data_config["valid"]  # /home/fair/Dataset/_Detection/DAC_2020_data_training/val.txt
    class_names = load_classes(data_config["names"])  # data/dac.names

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training, norm_flag=opt.data_norm)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    return dataloader, train_path, valid_path, class_names


def Quan_train(opt):
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Init Step 1: Create Model
    model, device, start_epoch = create_model(opt)

    # Init Step 2: Create Dataset
    dataloader, train_path, valid_path, class_names = create_dataset(opt)

    # Init Step 3: Create Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Epoch
    for epoch in range(start_epoch, opt.epochs):
        # Set model in train.
        model.train()

        start_time = time.time()

        # Epoch Step 1: Train
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
            if batches_done % opt.gradient_accumulations:  # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            print(opt.name + '\nEpoch:', epoch, '   Process:', batch_i, len(dataloader), '   Loss:', loss.cpu().data)

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

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

        # Epoch Step 2: Save
        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), opt.model_save_path + f"yolov3-tiny_fullp_ckpt_%d.pth" % epoch)

        # Epoch Step 3: Eval
        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class, IoU_total = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=500,
                norm_flag=opt.data_norm
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            # logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            with open(opt.model_save_path + "log_tiny.txt", "a+") as f:
                f.write('\n\n----------' + str(epoch) + '----------\n')
                f.write(AsciiTable(ap_table).table)
                f.write(f"---- mAP {AP.mean()}")
                f.write(f"---- IoU {IoU_total}")

                print(AsciiTable(ap_table).table)
                print(f"---- mAP {AP.mean()}")