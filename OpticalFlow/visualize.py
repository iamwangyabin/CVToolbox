import numpy as np

U = np.loadtxt('/home/wang/workspace/CVToolbox/OpticalFlow/small/HS_U.txt')
V = np.loadtxt('/home/wang/workspace/CVToolbox/OpticalFlow/small/HS_V.txt')

import cv2
cv2.imwrite("HS_cubic_U.jpg",U)
cv2.imwrite("HS_cubic_V.jpg",V)

im = cv2.imread("/home/wang/workspace/CVToolbox/OpticalFlow/small/rubic.1.bmp")
im = cv2.resize(im, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
prevgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

im = cv2.imread("/home/wang/workspace/CVToolbox/OpticalFlow/small/rubic.3.bmp")
im = cv2.resize(im, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
prevgray = gray
vis = draw_flow(gray, flow,step=32)
plt.imshow(vis)


def draw_flow(im, flow, step=8):
    h, w = im.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    # create line endpoints
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    # create image and draw
    vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        if np.linalg.norm(np.array([x1,y1])-np.array([x2,y2]))<2:
            continue
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(vis, (x2, y2), 1, (0, 0, 255), 2)
    return vis

def get_flow(U,V,im):
    h, w = im.shape[:2]
    flow=np.zeros((h,w,2))
    for i in range(h):
        for j in range(w):
            flow[i][j][0] = U[i][j]
            flow[i][j][1] = V[i][j]
    return flow

flow=get_flow(U,V,im)
# gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
vis=draw_flow(gray,flow,step=10)
cv2.imwrite("HS_cubic.png",vis)

