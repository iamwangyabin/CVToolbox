import os
import numpy
import cv2
import tqdm

alllist = os.listdir("/home/wang/d2-net/hpatches_sequences/hpatches-sequences-release")
out = "/home/wang/workspace/lf-net-release/samples"
for i in tqdm.tqdm(alllist):
    for j in range(1,7):
        oldname = os.path.join("/home/wang/d2-net/hpatches_sequences/hpatches-sequences-release",i,str(j)+".jpg")
        im = cv2.imread(oldname)
        img = cv2.resize(im, (640, 480), interpolation=cv2.INTER_CUBIC)
        newname = os.path.join(out, i+"_"+str(j)+".jpg")
        cv2.imwrite(newname,img)