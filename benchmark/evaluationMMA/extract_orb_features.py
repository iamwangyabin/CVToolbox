import cv2
import numpy as np
import imageio
from tqdm import tqdm

image_list_file = "./image_list_hpatches_sequences.txt"
output_extension = '.orb'
output_type = 'npz'


orb = cv2.ORB_create(1000)
cv2.AKAZE_create()
# Process the file
with open(image_list_file, 'r') as f:
    lines = f.readlines()

for line in tqdm(lines, total=len(lines)):
    path = line.strip()
    # path = os.path.join(path)
    image = imageio.imread(path)
    kp, des = orb.detectAndCompute(image, None)
    import pdb
    pdb.set_trace()

    kpx=[]
    kpy=[]
    for i in kp:
        kpx.append(i.pt[0])
        kpy.append(i.pt[1])

    kp = kp.cpu().detach().numpy()
    des = des.cpu().detach().numpy()
    keypoints = np.hstack((kp[:, 2:3], kp[:, 1:2]))

    with open(path + output_extension, 'wb') as output_file:
        np.savez(
            output_file,
            keypoints=keypoints,
            descriptors=des
        )
