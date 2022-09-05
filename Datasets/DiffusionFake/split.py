import os
import random
import shutil

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

symmetry = False
train_num = 1500
raw_data_path = "/data/workspace/datasets/diffusion"

del_file("/home/wangyabin/workspace/datasets/DeepFake_Data/CL_data/sd/train/0_real")
del_file("/home/wangyabin/workspace/datasets/DeepFake_Data/CL_data/sd/train/1_fake")
del_file("/home/wangyabin/workspace/datasets/DeepFake_Data/CL_data/sd/val/0_real")
del_file("/home/wangyabin/workspace/datasets/DeepFake_Data/CL_data/sd/val/1_fake")

if symmetry:
    realimages_list = os.listdir(os.path.join(raw_data_path, "real"))
    random.shuffle(realimages_list)
    train_imgs_real = realimages_list[:train_num]
    test_imgs_real = realimages_list[train_num:]

    for img in train_imgs_real:
        shutil.copy(os.path.join(raw_data_path, "real", img),
                    "/home/wangyabin/workspace/datasets/DeepFake_Data/CL_data/sd/train/0_real")
        shutil.copy(os.path.join(raw_data_path, "fake", img),
                    "/home/wangyabin/workspace/datasets/DeepFake_Data/CL_data/sd/train/1_fake")

    for img in test_imgs_real:
        shutil.copy(os.path.join(raw_data_path, "real", img),
                    "/home/wangyabin/workspace/datasets/DeepFake_Data/CL_data/sd/val/0_real")
        shutil.copy(os.path.join(raw_data_path, "fake", img),
                    "/home/wangyabin/workspace/datasets/DeepFake_Data/CL_data/sd/val/1_fake")

else:
    realimages_list = os.listdir(os.path.join(raw_data_path, "real"))
    random.shuffle(realimages_list)
    train_imgs_real = realimages_list[:train_num]
    test_imgs_real = realimages_list[train_num:]

    fakeimages_list = os.listdir(os.path.join(raw_data_path, "fake"))
    random.shuffle(fakeimages_list)
    train_imgs_fake = fakeimages_list[:train_num]
    test_imgs_fake = fakeimages_list[train_num:]

    for img in train_imgs_real:
        shutil.copy(os.path.join(raw_data_path, "real", img),
                    "/home/wangyabin/workspace/datasets/DeepFake_Data/CL_data/sd/train/0_real")

    for img in train_imgs_fake:
        shutil.copy(os.path.join(raw_data_path, "fake", img),
                    "/home/wangyabin/workspace/datasets/DeepFake_Data/CL_data/sd/train/1_fake")

    for img in test_imgs_real:
        shutil.copy(os.path.join(raw_data_path, "real", img),
                    "/home/wangyabin/workspace/datasets/DeepFake_Data/CL_data/sd/val/0_real")

    for img in test_imgs_fake:
        shutil.copy(os.path.join(raw_data_path, "fake", img),
                    "/home/wangyabin/workspace/datasets/DeepFake_Data/CL_data/sd/val/1_fake")



