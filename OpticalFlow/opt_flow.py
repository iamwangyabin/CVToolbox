import cv2

cap = cv2.VideoCapture(0)

ret, im = cap.read()
prev_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

while True:
    # get grayscale image
    ret, im = cap.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # compute flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prev_gray = gray
    # plot the flow vectors
    cv2.imshow('Optical flow', draw_flow(gray, flow))
    if cv2.waitKey(10) == 27:
        break


################################################################################################
import numpy as np
from scipy import signal
def optical_flow(I1g, I2g, window_size, tau=1e-2):
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
    w = int(window_size/2) # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g / 255. # normalize pixels
    I2g = I2g / 255. # normalize pixels
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode)+signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, I1g.shape[0]-w):
        for j in range(w, I1g.shape[1]-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
            b = np.reshape(It, (It.shape[0], 1))# get b here
            A = np.vstack((Ix, Iy)).T # get A here
            # if threshold τ is larger than the smallest eigenvalue of A'A:
            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                nu = np.matmul(np.linalg.pinv(A), b) # get velocity here
                u[i,j]=nu[0]
                v[i,j]=nu[1]
    return (u,v)


u,v=optical_flow(prevgray,gray,21)
flow=get_flow(u,v,im)
vis=draw_flow(gray,flow,step=16)
cv2.imwrite("LK_cubic.png",vis)





from scipy.ndimage.filters import convolve as filter2
import numpy as np
from typing import Tuple
#
HSKERN = np.array([[1/12, 1/6, 1/12],
                   [1/6,    0, 1/6],
                   [1/12, 1/6, 1/12]], float)

kernelX = np.array([[-1, 1],
                    [-1, 1]]) * .25  # kernel for computing d/dx

kernelY = np.array([[-1, -1],
                    [1, 1]]) * .25  # kernel for computing d/dy

kernelT = np.ones((2, 2))*.25


def HornSchunck(im1: np.ndarray, im2: np.ndarray, alpha: float=0.001, Niter: int=8):
    """
    im1: image at t=0
    im2: image at t=1
    alpha: regularization constant
    Niter: number of iteration
    """
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    # set up initial velocities
    uInitial = np.zeros([im1.shape[0], im1.shape[1]])
    vInitial = np.zeros([im1.shape[0], im1.shape[1]])

    # Set initial value for the flow vectors
    U = uInitial
    V = vInitial

    # Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

    # if verbose:
    #     from .plots import plotderiv
    #     plotderiv(fx, fy, ft)

#    print(fx[100,100],fy[100,100],ft[100,100])

        # Iteration to reduce error
    for _ in range(Niter):
        # %% Compute local averages of the flow vectors
        uAvg = filter2(U, HSKERN)
        vAvg = filter2(V, HSKERN)
# %% common part of update step
        der = (fx*uAvg + fy*vAvg + ft) / (alpha**2 + fx**2 + fy**2)
# %% iterative step
        U = uAvg - fx * der
        V = vAvg - fy * der

    return U, V


def computeDerivatives(im1: np.ndarray, im2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    fx = filter2(im1, kernelX) + filter2(im2, kernelX)
    fy = filter2(im1, kernelY) + filter2(im2, kernelY)

    # ft = im2 - im1
    ft = filter2(im1, kernelT) + filter2(im2, -kernelT)

    return fx, fy, ft

U,V=HornSchunck(prevgray, gray,Niter=100)