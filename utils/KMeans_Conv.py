import torch
from torch.nn import Conv2d
import numpy as np
from sklearn.cluster import KMeans


def pruning_weight(model, s=0.25):
    for name, module in model.named_modules():
        if 'K_conv_' in name:
            threshold = np.std(module.weight.data.cpu().numpy()) * s
            print(f'Pruning with threshold : {threshold} for layer {name}')
            module.prune(threshold)


def frozen_prune_weight(model):
    # Set pruned weight grad to 0 in order to keep 0 in weight
    for name, module in model.named_modules():
        if 'K_conv_' in name:
            device = module.weight.device

            tensor = module.weight.data.cpu().numpy()
            grad_tensor = module.weight.grad.data.cpu().numpy()
            grad_tensor = np.where(tensor == 0, 0, grad_tensor)
            module.weight.grad.data = torch.from_numpy(grad_tensor).to(device)


def km_weight_share_init(model):
    for name, module in model.named_modules():
        if 'K_conv_' in name:
            # Get weight data
            dev = module.weight.device
            weight = module.weight.data.cpu().numpy()
            shape = weight.shape  # [51,512,1,1]

            # Centroids initiate
            min_ = min(weight.reshape(-1, 1))
            max_ = max(weight.reshape(-1, 1))
            space = np.linspace(min_, max_, num=2 ** module.bits)  # Liner init
            kmeans = KMeans(
                n_clusters=len(space),
                init=space.reshape(-1, 1),
                n_init=1,
                precompute_distances=True,
                algorithm="full"
            )
            kmeans.fit(weight.reshape(-1, 1))

            # Log centers & table
            module.KM_centers.data = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(dev)
            module.KM_cor_tables = kmeans.labels_.reshape(shape)


def cluster_center_set_zero(model):
    for name, module in model.named_modules():
        if 'K_conv_' in name:
            # Get weight data
            dev = module.KM_centers.device
            centers = module.KM_centers.data.cpu().numpy()
            tensor_shape = centers.shape  # [?,1]

            centers = centers.reshape((centers.shape[0]))
            min_pos = np.argmin(np.abs(centers))

            centers[min_pos] = 0.
            centers.reshape(tensor_shape)

            module.KM_centers.data[min_pos] = 0.  # torch.from_numpy(centers.astype(np.float32)).to(dev)


class KConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False, bits=5, **kwargs):
        super(KConv2d, self).__init__(in_channels, out_channels, kernel_size, bias=bias, **kwargs)
        # define the bits of quantize
        self.bits = bits
        # set centroid Parameters (wait to be optimize)
        self.KM_centers = torch.nn.Parameter(torch.ones([2 ** self.bits, 1]))  # learnable parameters
        # log tables
        self.KM_cor_tables = []
        # Pruning weights Mask (if 1: valid; 0:pruned)
        self.weight_mask = np.ones(self.weight.data.cpu().numpy().shape, dtype=int)

    def prune(self, threshold):
        weight_dev = self.weight.device

        # Infos:
        print('Before Prune:', 'all:', self.weight_mask.reshape(-1).shape[0])

        # Convert Tensors to numpy and calculate
        weight = self.weight.data.cpu().numpy()
        self.weight_mask = np.where(abs(weight) < threshold, 0, self.weight_mask)

        # Apply new weight and mask
        self.weight.data = torch.from_numpy(weight * self.weight_mask).to(weight_dev)

        # Infos:
        mask = self.weight_mask.reshape(-1)
        print('Before Prune:', 'after:', (mask.shape[0] - sum(mask)) / mask.shape[0] * 100, ' %')

    def forward(self, x):
        dev = self.weight.device
        shape = self.weight.shape  # [51,512,1,1]

        # load quantized weight
        q_weight = self.KM_centers[self.KM_cor_tables].reshape(shape)
        self.weight.data = q_weight.to(dev)

        return self.conv2d_forward(x, self.weight)
