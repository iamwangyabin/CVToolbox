# coding=utf-8

# modified from https://www.cnblogs.com/willwell/p/google_image_search.html

import base64
import hashlib
import os
import re
import shutil
import time
from multiprocessing import Pool, cpu_count

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




header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36"}


driver = webdriver.Chrome(options=webdriver.ChromeOptions())
driver.get("https://images.google.com/")

image_button = driver.find_element(By.CLASS_NAME, 'nDcEnd') # 找到那个摄像机的类
image_button.send_keys(Keys.ENTER)


# 直接传url比较简单，还可以二次校验
upload_url = driver.find_element(By.CLASS_NAME, 'cB9M7')


url = 'https://instagram.fsin4-1.fna.fbcdn.net/v/t51.2885-15/277922118_674380763885312_8005111782321234355_n.jpg?stp=dst-jpg_e35_s640x640_sh0.08&cb=2d435ae8-0fbdf7c6&_nc_ht=instagram.fsin4-1.fna.fbcdn.net&_nc_cat=108&_nc_ohc=4kUHd3HgGWAAX9NQGjZ&edm=APU89FABAAAA&ccb=7-5&oh=00_AT-ANuZAQDSvDQDeF5be2o-X6EEIBEXZPX9_qvzMFA2nog&oe=6321232B&_nc_sid=86f79a'
upload_url.send_keys(url)
# 点击一下search 之前最好等待一下
search_button = driver.find_element(By.CLASS_NAME, 'Qwbd3')
search_button.send_keys(Keys.ENTER)

# 调整框框让他全选 suanle
# bbox = driver.find_element(By.CLASS_NAME, 'm8mnA')

# 直接把所有图都扒拉下来
allImages = driver.find_elements(By.TAG_NAME, "img")
uris=[]
for image in allImages:
    src = image.get_attribute("src")
    uris.append(src)
    pos = len(src) - src[::-1].index('/')


# 开始验证 1不是图的不要



img_content = requests.get(uris[0]).content
img_file = io.BytesIO(img_content)
image = Image.open(img_file)





#
# class GoogleSearcher:
#     def __init__(self, download="download", sleep_time=1):
#         super().__init__()
#         self._download = download # 下载文件夹
#         self.sleep_time = sleep_time  # 下载页面时等待时间
#         self.header = {
#             "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36"}
#
#         os.makedirs(self._download, exist_ok=True)  # 创建下载文件夹
#
#         self.option = webdriver.ChromeOptions()
#         # self.option.add_argument("--user-data-dir=" + f"C:/Users/{USERNAME}/AppData/Local/Google/Chrome/User Data/")
#         # self.option.add_argument("headless")  # if use headless, may failed.
#         self.option.add_argument("disable-gpu")
#         self.driver = webdriver.Chrome(options=self.option) # 以上为浏览器对象创建
#
#     def upload_img_get_html(self, file):
#     	# 上传图片并转到图片列表页面
#         print(
#             f"{Fore.GREEN} Begin to upload image {os.path.split(file)[1]} {Fore.RESET}")
#         self.driver.get("https://www.google.com/imghp")
#
#         # 等待相机按钮出现
#         condition_1 = EC.visibility_of_element_located(
#             (By.CLASS_NAME, "Gdd5U"))
#         WebDriverWait(self.driver, timeout=20,
#                       poll_frequency=0.5).until(condition_1)
#         import pdb;pdb.set_trace()
#
#         # 相机按钮出现后点击
#         image_button = self.driver.find_element_by_class_name("Gdd5U")
#         image_button.send_keys(Keys.ENTER)
#         # 等待出现上传图片字样
#         condition_2 = EC.visibility_of_element_located(
#             (By.ID, "DV7the"))
#         WebDriverWait(self.driver, timeout=20, poll_frequency=0.5).until(
#             condition_2)
#
#         # 点击上传图片
#         upload = self.driver.find_element_by_xpath('//*[@id="DV7the"]/div/a')
#         upload.send_keys(Keys.ENTER)
#
#         # 找到上传图片的控件
#         condition_3 = EC.visibility_of_element_located(
#             (By.ID, 'awyMjb'))
#         WebDriverWait(self.driver, timeout=10, poll_frequency=0.5).until(
#             condition_3)
#         input_ = self.driver.find_element_by_id('awyMjb')
#
#         # 因为上传图片的控件是一个input,直接将文件send就行
#         input_.send_keys(file)
#         print(f"{Fore.GREEN} uploaded {Fore.RESET}")
#
#         # 页面转向另一页
#         condition_4 = EC.visibility_of_element_located(
#             (By.XPATH, '//*[@id="top_nav"]'))
#         WebDriverWait(self.driver, timeout=20,
#                       poll_frequency=0.5).until(condition_4)
#         # 等待片刻
#         time.sleep(self.sleep_time)
#
#         # print(driver.current_url)
#         # print(driver.page_source)
#         print(f"{Fore.GREEN} Finish download source code{Fore.RESET}")
#         return self.driver.page_source
#
#     def highlight(self, element):
#         self.driver.execute_script(
#             "arguments[0].setAttribute('style', arguments[1]);", element, "background: yellow; border: 2px solid red;")
#
#     def wait_and_click(self, xpath):
#         #  Sometimes click fails unreasonably. So tries to click at all cost.
#         try:
#             w = WebDriverWait(self.driver, 15)
#             elem = w.until(EC.element_to_be_clickable((By.XPATH, xpath)))
#             elem.click()
#             self.highlight(elem)
#         except Exception as e:
#             print('Click time out - {}'.format(xpath))
#             print('Refreshing browser...')
#             self.browser.refresh()
#             time.sleep(2)
#             return self.wait_and_click(xpath)
#         return elem
#
#     def get_extension_from_link(self, link, default='jpg'):
#     # 获取文件后缀
#         splits = str(link).split('.')
#         if len(splits) == 0:
#             return default
#         ext = splits[-1].lower()
#         if ext == 'jpg' or ext == 'jpeg':
#             return 'jpg'
#         elif ext == 'gif':
#             return 'gif'
#         elif ext == 'png':
#             return 'png'
#         else:
#             return default
#
#     def base64_to_object(self, src):
#     # base64 解码
#         header, encoded = str(src).split(',', 1)
#         data = base64.decodebytes(bytes(encoded, encoding='utf-8'))
#         return data
#
#     def download_images(self, links, download_dir):
#     # 下载图片
#         total = len(links)
#         for index, link in enumerate(links):
#             try:
#                 if len(link) < 100:
#                     print('Downloading {} : {} / {}'.format(link, index + 1, total))
#                 else:
#                     print(
#                         'Downloading {} : {} / {}'.format(link[:100], index + 1, total))
#                         # 链接过长，只打印部分
#                 if str(link).startswith('data:image/jpeg;base64'):
#                 # base64编码的jpg图片
#                     response = self.base64_to_object(src=link)
#                     ext = 'jpg'
#                     is_base64 = True
#                 elif str(link).startswith('data:image/png;base64'):
#                 # base64编码的png图片
#                     response = self.base64_to_object(src=link)
#                     ext = 'png'
#                     is_base64 = True
#                 else:
#                 # 图片超链接
#                     response = requests.get(link, stream=True, timeout=5)
#                     ext = self.get_extension_from_link(link=link)
#                     is_base64 = False
#
#                 path = os.path.join(download_dir, str(index).zfill(4)+"."+ext)
#                 try:
#                     with open(path, "wb") as f:
#                     # base64图片和超链接图片两种保存方法
#                         if is_base64:
#                             f.write(response)
#                         else:
#                             shutil.copyfileobj(response.raw, f)
#                 except Exception as e:
#                     print('Save failed - {}'.format(e))
#
#                 del response
#             except Exception as e:
#                 print('Download failed - ', e)
#                 continue
#
#     def get_full_resolution_links(self):
#         print('[Full Resolution Mode]')
#         time.sleep(1)
#         elem = self.driver.find_element_by_tag_name("body")
#         print('Scraping links')
#         self.wait_and_click('//div[@data-ri="0"]')
#         time.sleep(1)
#         links = []
#         count = 1
#         last_scroll = 0
#         scroll_patience = 0
#         while True:
#             try:
#                 xpath = '//div[@id="islsp"]//div[@class="v4dQwb"]'
#                 div_box = self.driver.find_element(By.XPATH, xpath)
#                 self.highlight(div_box)
#                 xpath = '//img[@class="n3VNCb"]'
#                 img = div_box.find_element(By.XPATH, xpath)
#                 self.highlight(img)
#                 xpath = '//div[@class="k7O2sd"]'
#                 loading_bar = div_box.find_element(By.XPATH, xpath)
#                 # 等待图片加载，如果加载不完，获取到的是 base64 编码的图片
#                 while str(loading_bar.get_attribute('style')) != 'display: none;':
#                     time.sleep(0.1)
#                 src = img.get_attribute('src')
#                 if src is not None:
#                     links.append(src)
#                     if len(src) < 100:
#                         print('%d: %s' % (count, src))
#                     else:
#                         print('%d: %s' % (count, src[:100])) # 如果太长，只打印一部分
#                     count += 1
#             except StaleElementReferenceException:
#                 pass
#             except Exception as e:
#                 print(
#                     '[Exception occurred while collecting links from google_full] {}'.format(e))
#             scroll = self.driver.execute_script("return window.pageYOffset;") # 页面滚动的位置
#             if scroll == last_scroll:
#             # 页面滚动1
#                 scroll_patience += 1
#             else:
#                 scroll_patience = 0
#                 last_scroll = scroll
#             if scroll_patience >= 30:
#             #页面滚动30，停止
#                 break
#             elem.send_keys(Keys.RIGHT)
#         links = list(dict.fromkeys(links)) # 链接去重
#         print('Collect links done. Total: {}'.format(len(links)))
#         return links
#
#     def simple_file_run(self, img):
#         # 上传图片并进行搜索
#         img_name = os.path.splitext(os.path.split(img)[1])[0] # 图片名
#         parent_name = os.path.split(os.path.split(img)[0])[-1] # 图片的父级名字，用来区分图片的类别
#         print("--> Processing image:  {}  ".format(img_name))
#         download_dir = os.path.join(self._download, parent_name, img_name)
#         os.makedirs(download_dir, exist_ok=True)
#         html_source = self.upload_img_get_html(img)  # 上传图片，到搜索结果页
#         similar_img_href = self.driver.find_element_by_xpath(
#             '//div[@class="e2BEnf U7izfe"]/h3/a')
#         similar_img_href.click()  # 查找“类似图片”的链接并点击，进入图片列表页
#         links = self.get_full_resolution_links()  # 将所有图片的大图链接进行收集
#         self.download_images(links, download_dir)  # 下载这些大图
#         print("{}Image {} finished\n{}".format(
#             Fore.GREEN, img_name, Fore.RESET))
#
#
#
# EXTEND = [".bmp", ".jpg", ".jpeg", ".tif", ".tiff",
#           ".jfif", ".png", ".gif", ".iff", ".ilbm"]
#
#
# def is_img(img_path):
#     # 根据后缀判断是否为图片
#     ext = os.path.splitext(img_path)[1]
#     if ext in EXTEND:
#         return True
#     else:
#         return False
#
# def getfilelist(path, filelist):
#     file = os.listdir(path)
#     for im_name in file:
#         if os.path.isdir(os.path.join(path, im_name)):
#             getfilelist(os.path.join(path, im_name), filelist)
#         else:
#             if is_img(im_name):
#                 name = os.path.join(path, im_name)
#                 filelist.append(name)
#
# def partition(ls, size):
#     num_per_list = len(ls)//size
#     result = []
#     if num_per_list*size == len(ls):
#         for i in range(size):
#             result.append(ls[num_per_list*i:num_per_list*(i+1)])
#     else:
#         for i in range(size-1):
#             result.append(ls[num_per_list*i:num_per_list*(i+1)])
#         result.append(ls[num_per_list*(size-1):])
#     return result
#
# def download_task(filelist):
#     searcher = GoogleSearcher(
#         download=r"./download")
#     # import pdb;pdb.set_trace()
#     for file in filelist:
#         searcher.simple_file_run(file)  # 上传单张图并进行以图搜图
#
# def run():
#     num_process = 1 # 进程数设置为cpu核心数
#     # num_process = cpu_count() # 进程数设置为cpu核心数
#     pool = Pool(num_process) # 建立一个进程池
#     filelist = []
#     upload = r"./upload" # 需要进行上传的图片文件夹
#     getfilelist(upload, filelist)  # 递归查找文件夹里面所有的图片文件
#     result = partition(filelist, num_process) # 将图片文件列表平均分为几个list，每个进程跑一部分
#     # import pdb;pdb.set_trace()
#     # download_task(result[0])
#     pool.map_async(download_task, result) # 下载任务丢进进程池
#     pool.close() # 不再允许加入进程池
#     pool.join() # 等待进程完成
#






# from selenium import webdriver
# import os
# import time
#
# # Using Chrome to access web
# driver = webdriver.Chrome(executable_path=os.path.abspath(webdriver.ChromeOptions()))
#
# try:
#     # Open the website
#     driver.get("https://images.google.com/")
#
#     # Find cam button
#     cam_button = driver.find_elements_by_xpath('//div[@aria-label=Search by image and @role=button]')[0]
#     cam_button.click()
#
#     # Find upload tab
#     upload_tab = driver.find_elements_by_xpath('//*[contains(text(), Upload an image)]')[0]
#     upload_tab.click()
#
#     # Find image input
#     upload_btn = driver.find_element_by_name('encoded_image')
#     upload_btn.send_keys(os.getcwd()+/image.png)
#
#     time.sleep(10)
#
# except Exception as e:
#     print(e)
#
# driver.quit()


# if __name__ == '__main__':
#     run()

# https://www.google.com/searchbyimage?site=search&sa=X&image_url=https://imagen.research.google/main_gallery_images/an-art-gallery-displaying-monet-paintings-the-art-gallery-is-flooded-robots.jpg



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