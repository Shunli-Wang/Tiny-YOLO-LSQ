# Used to creat train.txt and val.txt

import os
import random

path = '/home/fair/Dataset/_Detection/DAC_2020_data_training/DAC_Dataset'
videos_lists = []

train = []
val = []
count = 0

for filename in os.listdir(path):
    p = os.path.join(path, filename)
    videos_lists.append(p)

    # count += 1
    # if count % 10 == 0:
    #     val.append(p)

for video_path in videos_lists:  # 95 files
    file_names = os.listdir(video_path)  # get all img&cml files

    for file_name in file_names:
        if 'jpg' in file_name:  # is img
            if random.random() < 0.7:
                train.append(video_path + '/' + file_name)
            else:
                val.append(video_path + '/' + file_name)

# random.shuffle(train)
# random.shuffle(val)

with open('train.txt', 'w') as f:
    for file in train:
        f.write(file + '\n')
with open('val.txt', 'w') as f:
    for file in val:
        f.write(file + '\n')

print(123)
