import numpy as np 
import cv2

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

from LK import PinholeCamera, VisualOdometry

# # TUM fr1
# # fx		fy		cx		cy		d0		d1		d2		d3		d4
# # 517.3	516.5	318.6	255.3	0.2624	-0.9531	-0.0054	0.0026	1.1633
# # fx		fy		cx		cy		k1		k2		p1		p2		k3

def showGroundtruth():
	f = open("./Data/groundtruth.txt")
	x = []
	y = []
	z = []
	for line in f:
		if line[0] == '#':
			continue
		data = line.split()
		x.append( float(data[1] ) )
		y.append( float(data[2] ) )
		z.append( float(data[3] ) )
	ax = plt.subplot( 111, projection='3d')
	ax.plot(x,y,z)
	plt.show()

def getImageLists():
	imgList=[]
	f = open("./Data/rgb.txt")
	for line in f:
		if line[0] == '#':
			continue
		data = line.split()
		imgList.append(data[1])
	return imgList

def transRo2Qu(t,R):
	trR = R[0][0] + R[1][1] + R[2][2]
	w = np.sqrt(trR + 1) / 2
	x = (R[2][1] - R[1][2]) / (4 * w)
	y = (R[0][2] - R[2][0]) / (4 * w)
	z = (R[1][0] - R[0][1]) / (4 * w)
	return "%.5f %.5f %.5f %.5f %.5f %.5f %.5f" % ( t[0], t[1], t[2], x, y, z, w)

def getVO(data_path,rgb_txt):
	cam = PinholeCamera(640.0, 480.0, 517.3, 516.5, 318.6, 255.3, 0.2624, -0.9531, -0.0054, 0.0026, 1.1633)
	imgList = getImageLists()
	vo = VisualOdometry(cam)
	img = cv2.imread('./Data/'+imgList[0], 0)
	f = open("teat3.txt","w")
	for img_id in range(len(imgList)):
		img = cv2.imread('./Data/'+imgList[img_id], 0)
		vo.update(img, img_id)
		if img_id > 1:
			cur_t = vo.cur_t
			cur_r = vo.cur_R
			name = imgList[img_id].split('/')[1].split('.')[0]
			pose = transRo2Qu(cur_t,cur_r)
			f.write(name+' '+pose+'\n')
		print(img_id)



def showGroundtruth():
	f = open("./teat2.txt")
	x = []
	y = []
	z = []
	for line in f:
		if line[0] == '#':
			continue
		data = line.split()
		x.append( float(data[1] ) )
		y.append( float(data[2] ) )
		z.append( float(data[3] ) )
	ax = plt.subplot( 111, projection='3d')
	ax.plot(x,y,z)
	plt.show()


if __name__ == "__main__":
	getVO("","")
