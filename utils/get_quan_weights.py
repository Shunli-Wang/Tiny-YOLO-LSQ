import torch
import numpy as np
from matplotlib import pyplot as plt

pretrained_quan_weights = '/home/fair/Desktop/wsl/_Detection/TinyYOLO_DAC_r/release_checkpoints/V4_5/yolov3-tiny_fullp_ckpt_96.pth'
checkpoint = torch.load(pretrained_quan_weights)
# print(checkpoint['module_list.15.Q_conv_15.quan_a.s'])

for name in checkpoint.keys():
    if 'quan_a.s' in name:
        print(name, checkpoint[name])

for name in checkpoint.keys():
    if 'quan_w.s' in name:
        print(name, checkpoint[name])



for i in checkpoint.keys():
    print(i)

parameters = {}

for i in [0, 2, 4, 6, 8, 10, 13, 14, 15, 18, 21, 22]:
    temp = {}

    # Conv:
    conv_w = 'module_list.' + str(i) + '.Q_conv_' + str(i) + '.weight'
    conv_w_scale = 'module_list.' + str(i) + '.Q_conv_' + str(i) + '.quan_w.s'
    conv_a_scale = 'module_list.' + str(i) + '.Q_conv_' + str(i) + '.quan_a.s'
    temp['conv_w'] = checkpoint[conv_w].cpu().numpy()
    temp['conv_w_scale'] = checkpoint[conv_w_scale].cpu().numpy()
    if i not in [0]:
        temp['conv_a_scale'] = checkpoint[conv_a_scale].cpu().numpy()

    if i in [15, 22]:
        conv_b = 'module_list.' + str(i) + '.Q_conv_' + str(i) + '.bias'
        temp['conv_b'] = checkpoint[conv_b].cpu().numpy()
        parameters[str(i)] = temp
        continue

    # BatchNorm:
    bn_w = 'module_list.' + str(i) + '.batch_norm_' + str(i) + '.weight'
    bn_b = 'module_list.' + str(i) + '.batch_norm_' + str(i) + '.bias'
    bn_run_mean = 'module_list.' + str(i) + '.batch_norm_' + str(i) + '.running_mean'
    bn_run_var = 'module_list.' + str(i) + '.batch_norm_' + str(i) + '.running_var'
    temp['bn_w'] = checkpoint[bn_w].cpu().numpy()
    temp['bn_b'] = checkpoint[bn_b].cpu().numpy()
    temp['bn_run_mean'] = checkpoint[bn_run_mean].cpu().numpy()
    temp['bn_run_var'] = checkpoint[bn_run_var].cpu().numpy()

    parameters[str(i)] = temp

np.save('/home/fair/Desktop/wsl/_Detection/TinyYOLO_DAC_r/utils/weights.npy', parameters)

print(123)



para = np.load('weights.npy').item()
# para.keys()


for i in [12]:
    print('--------------------', i)

    conv_w = para[str(i)]['conv_w']
    conv_w_scale = para[str(i)]['conv_w_scale']
    conv_a_scale = para[str(i)]['conv_a_scale']

    conv_w_r = np.round(conv_w / conv_w_scale)
    conv_w_rc = np.clip(conv_w_r, -3, +3).astype(int)
    conv_w_rc_ex = conv_w_rc.reshape((-1,))

    conv_w_rc_1 = conv_w_rc.reshape((conv_w_rc.shape[0], conv_w_rc.shape[1], 9))
    # conv_w_rc_2 = np.swapaxes(conv_w_rc_1, 1, 2)
    conv_w_rc_3 = conv_w_rc_1.reshape((-1, 9))

    conv_w_rc_p = conv_w_rc_3 + 3

    conv_w_rc_seven = conv_w_rc_p[:, 0] * 7 ** 0 + conv_w_rc_p[:, 1] * 7 ** 1 + \
                      conv_w_rc_p[:, 2] * 7 ** 2 + conv_w_rc_p[:, 3] * 7 ** 3 + \
                      conv_w_rc_p[:, 4] * 7 ** 4 + conv_w_rc_p[:, 5] * 7 ** 5 + \
                      conv_w_rc_p[:, 6] * 7 ** 6 + conv_w_rc_p[:, 7] * 7 ** 7 + conv_w_rc_p[:, 8] * 7 ** 8

    temp = conv_w_rc_seven.sort()

    arr = np.unique(conv_w_rc_seven)

    from collections import Counter
    counter = Counter(conv_w_rc_seven)  # {label:sum(label)}
    sorted_list = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    print(sorted_list)

    count_sum = 0
    for i in range(32):
        count_sum += sorted_list[i][1]

    for i in conv_w_rc_seven:
        print(i, ':')
        count = sum(conv_w_rc_seven == i)
        if count != 1:
            print(i, count)

    count_list = np.array([])
    for i in range(conv_w_rc_seven.shape[0]):
        print(i)

        # Already analysed
        if conv_w_rc_seven[i] == -1:
            continue

        # Not yet analysed
        count = 1
        for j in range(i + 1, conv_w_rc_seven.shape[0]):
            if conv_w_rc_seven[j] == -1:
                continue
            if conv_w_rc_seven[i] == conv_w_rc_seven[j]:
                conv_w_rc_seven[j] = -1
                count += 1

        count_list = np.append(count_list, [conv_w_rc_seven[i], count])
        conv_w_rc_seven[i] = -1

        if i % 20 == 0:
            info = count_list.reshape((-1, 2)).astype(int)
            print(info)

    print('Types:', len(count_list))
    if np.array(count_list).sum() == conv_w_rc_3.shape[0]:
        print('Right')

    # mean_list = np.var(conv_w_rc_3,axis=1)
    # plt.hist(mean_list, rwidth=0.5)
    # plt.show()

    sum_ = 0
    count_list = np.array([])
    for j in range(-3, 4):
        weight_count = sum(conv_w_rc_ex == j)
        sum_ += weight_count
        count_list = np.append(count_list, [weight_count])
    print(count_list, conv_w_rc_ex.shape[0])
    print('Correct:', sum_ == conv_w_rc_ex.shape[0])

    rects = plt.bar(np.arange(-3, 4), count_list)
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, '%2.2f' % (height / conv_w_rc_ex.shape[0] * 100) + '%',
                 ha='center', va='bottom')
    plt.xlabel('Quantisize weight : -3~+3')
    plt.ylabel('Count')
    plt.title(str(i) + 'th Conv Weight')
    # plt.savefig(str(i)+'.jpg')
    # ax.set_title()

    plt.show()

    if i in [15, 22]:
        bn_w = para[str(0)]['bn_w']
        bn_b = para[str(0)]['bn_b']

print(123)
