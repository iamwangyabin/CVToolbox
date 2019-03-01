import numpy as np
# from numpy import *
# Rudin-Osher-Fatemi(ROF)去噪模型
def denoise(im,U_init,tolerance=0.1,tau=0.125,tv_weight=100):
    m,n=im.shape
    U=U_init
    Px=im
    Py=im
    error=1

    while(error>tolerance):
        Uold=U
        GradUx=np.roll(U,-1,axis=1)-U
        GradUy=np.roll(U,-1,axis=0)-U
        PxNew=Px+(tau/tv_weight)*GradUx
        PyNew=Py+(tau/tv_weight)*GradUy
        NormNew=np.maximum()