import os 
import shutil

inpath="./fr1/images"
images=os.listdir("./fr1/images")
images = sorted(images)

outputpath='./fr2/images'
os.makedirs(outputpath)
i=0
for p in images:
    if i%10 == 0:
        source=os.path.join(inpath,p)
        target =  os.path.join(outputpath,p)
        shutil.copy(source, target)
    i+=1