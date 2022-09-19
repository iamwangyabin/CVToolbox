# CameraCalibrate
自己写的相机标定程序

挺奇怪的 这个计算棋盘的程序有时候运行快有时候又很慢很慢
## 主要问题是：
#### 如果说我把图片计算前先把尺寸降低那么算出来的内参矩阵该怎么去调节？
答案：https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix  
但是吧，又有人说这样得到的结果其实还是不对，搜一下其实可以这样：  
https://stackoverflow.com/questions/48892301/can-findchessboardcorners-be-speeded-up  
