# 将爬取的数据和被初步筛选的图片进行训练测试划分和打马赛克

import os
download_dirs = os.listdir("delle_google/download/seleced_delle")


# 1、打马赛克如何打，按照比例来







realimg = []
fakeimg = []






for instance in A_dirs:
    if instance != '.DS_Store':
        imgs = os.listdir(os.path.join("delle_google/download/seleced_delle", instance))
        print(imgs)
        tempimgsize = []
        for img in imgs:
            if img == '0.jpeg':
                realimg.append(os.path.join("delle_google/download/seleced_delle", instance, img))
            else:
                image = Image.open(os.path.join("delle_google/download/seleced_delle", instance, img))
                tempimgsize.append(os.path.join("delle_google/download/seleced_delle", instance, img))
