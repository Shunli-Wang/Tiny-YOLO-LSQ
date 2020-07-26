# Tiny-YOLO-LSQ
This is an implementation of YOLO using LSQ network quantization method.

Firstly, you need to download the DAC2020 Dataset and change "train" and "valid" items to point to "train.txt" and "val.txt" respectively. You can download the whole dataset from DAC. 
Note that this dataset contains absolute paths because we encapsulated DAC dataset in the form of Yolo project. You need to run this file ```dac_dataset.py``` yourself to re-generate train.txt  and val.txt which contains right paths in your own computer. We divided the train set and val set according to 7:3 randomly. 
```
classes= 12
train=<PATH to your Dataset>/DAC_2020_data_training/all.txt
valid=<PATH to your Dataset>/_Detection/DAC_2020_data_training/val.txt
names=data/dac.names
backup=backup/
eval=dac
```

Secondly, the whole project is based on [YOLO](https://github.com/eriklindernoren/PyTorch-YOLOv3), so it has the same environment configuration as Yolo. In order to have a better understanding of the project, it is strongly recommended that users should be familiar with Yolo's project first.

Thirdly, If you have finished the dataset preparation and environment configuration, execute the following instructions to train the quantization network:

```shell
python train_tiny_qan.py \
    --gpu 0 \
    --fp_pretrained 'checkpoints/fp_ckpt/Tiny_yolo_ckpt_90.pth' \
    --model_save_path 'checkpoints/Quan_model/' \
    --model_def "config/yolov3-tiny_lsq.cfg" \
    --quan_weitht_bit 3 \
    --quan_activation_bit 3
```
Our Tiny-YOLO framework is defined as follow:
![Tiny-YOLO-LSQ](https://github.com/Shunli-Wang/Tiny-YOLO-LSQ/blob/master/data/Tiny-yolo.png)

## Citation
```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
@article{DBLP:journals/corr/abs-1902-08153,
  title     = {Learned Step Size Quantization},
  author    = {Steven K. Esser and Jeffrey L. McKinstry and Deepika Bablani and Rathinakumar Appuswamy and Dharmendra S. Modha},
  journal   = {CoRR},
  volume    = {abs/1902.08153},
  year      = {2019},
  url       = {http://arxiv.org/abs/1902.08153},
}
```
