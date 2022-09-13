# 将爬取的数据和被初步筛选的图片进行训练测试划分和打马赛克
import os
from PIL import Image, ImageDraw

base_path = "./delle_laion/download/"

download_dirs = os.listdir(base_path)

all_real = []
all_fake = []

for instance in download_dirs:
    if instance != '.DS_Store':
        imgs = os.listdir(os.path.join(base_path, instance))
        temp_fake_imgs = []
        temp_fake_imgsizes = []
        for img in imgs:
            if img == '0.jpeg':
                img_cont = Image.open(os.path.join(base_path, instance, img))
                w, h = img_cont.size
                a = ImageDraw.ImageDraw(img_cont)
                a.rectangle(((w, h), (w - 0.10 * w, h - 0.03 * h)), fill="black", outline='black', width=1)
                all_fake.append(img_cont)
            else:
                img_cont = Image.open(os.path.join(base_path, instance, img))
                w, h = img_cont.size
                center_w = w / 2
                center_h = h / 2
                crop_size = min(img_cont.size) / 2
                cropped = img_cont.crop((center_w - crop_size, center_h - crop_size, center_w + crop_size,
                                    center_h + crop_size))
                a = ImageDraw.ImageDraw(cropped)
                cropped_size_w, cropped_size_h = cropped.size
                a.rectangle(((cropped_size_w, cropped_size_h),
                             (cropped_size_w - 0.10 * cropped_size_w, cropped_size_h - 0.03 * cropped_size_h)),
                            fill="black", outline='black', width=1)
                temp_fake_imgs.append(cropped)
                temp_fake_imgsizes.append(cropped_size_w)
        if len(temp_fake_imgsizes) != 0:
            index = temp_fake_imgsizes.index(max(temp_fake_imgsizes))
            all_real.append(temp_fake_imgs[index])



outputdir = "./delle_laion/processed/"

train_num = int(len(all_fake)*0.5)
train_real = all_real[:train_num]
test_real = all_real[train_num:]
train_fake = all_fake[:train_num]
test_fake = all_fake[train_num:]

for idx, img in enumerate(train_real):
    img.save(os.path.join(outputdir, 'train', '0_real', str(idx)+".jpeg"))

for idx, img in enumerate(train_fake):
    img.save(os.path.join(outputdir, 'train', '1_fake', str(idx)+".jpeg"))

for idx, img in enumerate(test_real):
    img.save(os.path.join(outputdir, 'val', '0_real', str(idx)+".jpeg"))

for idx, img in enumerate(test_fake):
    img.save(os.path.join(outputdir, 'val', '1_fake', str(idx)+".jpeg"))
