import numpy as np
import os

def writ2TXT(data,path):
    with open(path, 'w') as f:
        for line in data:
            sr = str(list(line)).replace('[','').replace(']','').replace(',',' ') + '\n'
            f.write(sr)

def writeNpz(npzPath,outPath):
    for i in os.listdir(npzPath):
        if i[-3:] == 'npz':
            data = np.load(os.path.join(npzPath, i))
            new = np.hstack([data['kpts'], data['descs']])
            writ2TXT(new, os.path.join(outPath, i[:-4])+'.txt')

if __name__ == "__main__":
    npzPath = './rgb_feats_indoor'
    outPath = './rgb_txt'
    writeNpz(npzPath,outPath)