from PIL import Image

path = '/home/fair/Dataset/_Detection/DAC_2020_data_training/DAC_Dataset/car3/000638.jpg'

img = Image.open(path).convert('RGB')
print(img.size)

img_ = img.resize((512, 288),Image.NEAREST)    # NEAREST  BILINEAR
print(img_.size)

