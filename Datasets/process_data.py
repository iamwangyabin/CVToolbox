# 将爬取的数据和被初步筛选的图片进行训练测试划分和打马赛克

import os



A_dirs = os.listdir("./download/seleced_delle")


realimg = []
fakeimg = []


for instance in A_dirs:
    if instance != '.DS_Store':
        imgs = os.listdir(os.path.join("./download/seleced_delle", instance))
        print(imgs)
        tempimgsize = []
        for img in imgs:
            if img == '0.jpeg':
                realimg.append(os.path.join("./download/seleced_delle", instance, img))
            else:
                image = Image.open(os.path.join("./download/seleced_delle", instance, img))
                tempimgsize.append(os.path.join("./download/seleced_delle", instance, img))
