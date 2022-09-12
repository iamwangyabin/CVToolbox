# coding=utf-8

# modified from https://www.cnblogs.com/willwell/p/google_image_search.html

import base64
import hashlib
import os
import re
import shutil
import time
from multiprocessing import Pool, cpu_count
import numpy as np



import requests
import tqdm
from colorama import Fore
from selenium import webdriver
from selenium.common.exceptions import (ElementNotVisibleException,
                                        StaleElementReferenceException)
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
import io
from PIL import Image

# url = 'https://instagram.fsin4-1.fna.fbcdn.net/v/t51.2885-15/277922118_674380763885312_8005111782321234355_n.jpg?stp=dst-jpg_e35_s640x640_sh0.08&cb=2d435ae8-0fbdf7c6&_nc_ht=instagram.fsin4-1.fna.fbcdn.net&_nc_cat=108&_nc_ohc=4kUHd3HgGWAAX9NQGjZ&edm=APU89FABAAAA&ccb=7-5&oh=00_AT-ANuZAQDSvDQDeF5be2o-X6EEIBEXZPX9_qvzMFA2nog&oe=6321232B&_nc_sid=86f79a'
# driver = webdriver.Chrome(options=webdriver.ChromeOptions())
# driver.set_window_position(0, 0)
# driver.set_window_size(300, 1000)
# driver.get("https://images.google.com/")
# image_button = driver.find_element(By.CLASS_NAME, 'nDcEnd')  # 找到那个摄像机的类
# image_button.send_keys(Keys.ENTER)
# time.sleep(3)
# # 直接传url比较简单，还可以二次校验
# upload_url = driver.find_element(By.CLASS_NAME, 'cB9M7')
# upload_url.send_keys(url)
# time.sleep(3)
# # 点击一下search 之前最好等待一下
# search_button = driver.find_element(By.CLASS_NAME, 'Qwbd3')
# search_button.send_keys(Keys.ENTER)
# time.sleep(3)
# # driver.execute_script("var q=document.documentElement.scrollTop=10000")
# instances = driver.find_element(By.CLASS_NAME, 'Vd9M6')
# SimilarImagesURLs = driver.find_elements(By.TAG_NAME, "img")
# instance.find_element(By.CLASS_NAME, 'UAiK1e')


def start_search_laion(img_url):
    img_content = requests.get(img_url).content
    img_file = io.BytesIO(img_content)
    raw_image = Image.open(img_file)
    raw_image_size = raw_image.size
    selected_urls = []
    selected_imgs = []
    selected_captions = []
    if min(raw_image_size) > 256:


        




def start_search_google(img_url):
    # first download is image
    img_content = requests.get(img_url).content
    img_file = io.BytesIO(img_content)
    raw_image = Image.open(img_file)
    raw_image_size = raw_image.size
    selected_urls = []
    selected_imgs = []
    selected_captions = []
    if min(raw_image_size) > 256:
        driver = webdriver.Chrome(options=webdriver.ChromeOptions())
        driver.set_window_position(0, 0)
        driver.set_window_size(300, 2000)
        driver.get("https://images.google.com/")
        image_button = driver.find_element(By.CLASS_NAME, 'nDcEnd')  # 找到那个摄像机的类
        image_button.send_keys(Keys.ENTER)
        time.sleep(3)
        # 直接传url比较简单，还可以二次校验
        upload_url = driver.find_element(By.CLASS_NAME, 'cB9M7')
        upload_url.send_keys(img_url)
        time.sleep(3)
        # 点击一下search 之前最好等待一下
        search_button = driver.find_element(By.CLASS_NAME, 'Qwbd3')
        search_button.send_keys(Keys.ENTER)
        time.sleep(10)

        # 直接把所有图都扒拉下来

        all_instances = WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.XPATH, "//div[@class='Vd9M6 abDdKd FttNLb  xuQ19b']")))

        SimilarAllURLs = []
        SimilarAllCaptions = []
        for instance in all_instances:
            imgcaption = instance.find_element(By.TAG_NAME, 'a').get_attribute("aria-label")
            imgurl = instance.find_element(By.CLASS_NAME, 'ksQYvb').get_attribute("data-thumbnail-url")
            SimilarAllCaptions.append(imgcaption)
            SimilarAllURLs.append(imgurl)
        # SimilarImagesURLs = driver.find_elements(By.TAG_NAME, "img")
        # SimilarAllURLs = []
        # for image in SimilarImagesURLs:
        #     src = image.get_attribute("src")
        #     SimilarAllURLs.append(src)
        # 开始验证 图太小的不要
        for url, cap in zip(SimilarAllURLs, SimilarAllCaptions):
            if re.match(r'^https?:/{2}\w.+$', url):
                img_content = requests.get(url).content
                img_file = io.BytesIO(img_content)
                image = Image.open(img_file)
                if min(image.size)>=100:
                    selected_urls.append(url)
                    selected_imgs.append(image)
                    selected_captions.append(cap)
        driver.close()
    return raw_image, selected_urls, selected_imgs, selected_captions


# with open(r'./download/delle/delle.txt',"r",encoding='utf-8') as f:
#     all_raw_urls = f.readlines()
#
# save_dir = "./download/delle/"
# all_data = {}
# for idx, raw_url in enumerate(all_raw_urls):
#     raw_url = raw_url.split()[0]
#     try:
#         raw_image, selected_urls, selected_imgs, selected_captions = start_search_google(raw_url)
#         all_data[idx] = {'raw_url':raw_url, 'raw_image':raw_image,
#                          'selected_urls':selected_urls, 'selected_imgs':selected_imgs, 'selected_captions':selected_captions}
#
#         os.mkdir(os.path.join(save_dir, str(idx)))
#
#         raw_image.save(os.path.join(save_dir, str(idx), "0.jpeg"), "JPEG")
#
#         for simg, cap in zip(selected_imgs, selected_captions):
#             if simg.mode != 'RGB':
#                 simg = simg.convert('RGB')
#             intab = r'[?*/\|.:><]'
#             simg.save(os.path.join(save_dir, str(idx), re.sub(intab, "", str(cap))[:20]+".jpeg"), "JPEG")
#     except:
#         print(raw_url)
#
# np.save('delle_processed.npy', all_data)

with open(r'./download/imagen/imagen.txt',"r",encoding='utf-8') as f:
    all_raw_urls = f.readlines()

save_dir = "./download/imagen/"
all_data = {}
for idx, raw_url in enumerate(all_raw_urls):
    raw_url = raw_url.split()[0]
    try:
        raw_image, selected_urls, selected_imgs, selected_captions = start_search_google(raw_url)
        all_data[idx] = {'raw_url':raw_url, 'raw_image':raw_image,
                         'selected_urls':selected_urls, 'selected_imgs':selected_imgs, 'selected_captions':selected_captions}

        os.mkdir(os.path.join(save_dir, str(idx)))

        raw_image.save(os.path.join(save_dir, str(idx), "0.jpeg"), "JPEG")

        for simg, cap in zip(selected_imgs, selected_captions):
            if simg.mode != 'RGB':
                simg = simg.convert('RGB')
            intab = r'[?*/\|.:><]'
            simg.save(os.path.join(save_dir, str(idx), re.sub(intab, "", str(cap))[:20]+".jpeg"), "JPEG")
    except:
        print(raw_url)

np.save('imagen_processed.npy', all_data)






# f = open('delleurls','r')
# alldata = f.readlines()
# urls = []
# for line in alldata:
#     parse=line.split()
#     if len(parse)!=0:
#         urls.append(parse[2].replace('"',''))
#
#
# l=open('delle.txt','w', encoding='utf-8')
#
#
#
# #保存数据txt文件
# with open(r'delle.txt',"w",encoding='utf-8') as l:
#     for var in urls:
#         l.write(str(var) + '\n')
#
# l.close()