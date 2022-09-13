# pip install clip-retrieval
import re
import os
import requests
import urllib.request

import io
from PIL import Image
import numpy as np
import clip_retrieval
from clip_retrieval.clip_client import ClipClient, Modality

client = ClipClient(
    url="https://knn5.laion.ai/knn-service",
    indice_name="laion5B",
    aesthetic_score=9,
    aesthetic_weight=0.5,
    modality=Modality.IMAGE,
    num_images=20,
)

headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36'
}

with open(r'./delle_laion/delle.txt', "r", encoding='utf-8') as f:
    all_raw_urls = f.readlines()

save_dir = "./delle_laion/download"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

all_data = {}

for idx, raw_url in enumerate(all_raw_urls):
    print(idx)
    try:
        selected_urls = []
        selected_imgs = []
        selected_captions = []
        raw_url = raw_url.split()[0]
        img_content = requests.get(raw_url).content
        img_file = io.BytesIO(img_content)
        raw_image = Image.open(img_file)
        raw_image_size = raw_image.size

        if min(raw_image_size) > 256:
            if not os.path.exists(os.path.join(save_dir, str(idx))):
                os.mkdir(os.path.join(save_dir, str(idx)))
            raw_image.save(os.path.join(save_dir, str(idx), "0.jpeg"), "JPEG")
            results = client.query(image=os.path.join(save_dir, str(idx), "0.jpeg"))

            for instance in results:

                img_url = instance['url']
                img_caption = instance['caption']
                try:
                    intab = r'[?*/\|.:><] '
                    urllib.request.urlretrieve(img_url, os.path.join(save_dir, "tmp.jpeg"))
                    lai_image = Image.open(os.path.join(save_dir, "tmp.jpeg"))
                    lai_image_size = lai_image.size
                    if min(lai_image_size) > 200:
                        lai_image.save(
                            os.path.join(save_dir, str(idx), re.sub(intab, "", str(img_caption))[:20] + ".jpeg"), "JPEG")
                        selected_urls.append(img_url)
                        selected_imgs.append(lai_image)
                        selected_captions.append(img_caption)
                    # laiimg_content = requests.get(img_url, headers=headers, stream=True).content
                    # laiimg_file = io.BytesIO(laiimg_content)
                    # lai_image = Image.open(laiimg_file)
                    # lai_image_size = lai_image.size
                    # if min(lai_image_size) > 200:
                    #     # if lai_image.mode != 'RGB':
                    #     #     lai_image = lai_image.convert('RGB')
                    #     intab = r'[?*/\|.:><] '
                    #     lai_image.save(os.path.join(save_dir, str(idx), re.sub(intab, "", str(img_caption))[:20] + ".jpeg"), "JPEG")
                    #     selected_urls.append(img_url)
                    #     selected_imgs.append(lai_image)
                    #     selected_captions.append(img_caption)
                except:
                    print("error selected: \t" + img_url)

        all_data[idx] = {'raw_url': raw_url, 'raw_image': raw_image,
                         'selected_urls': selected_urls, 'selected_imgs': selected_imgs,
                         'selected_captions': selected_captions}
    except:
        print("error delle image:\t" + raw_url)

np.save('delle_laion.npy', all_data)
