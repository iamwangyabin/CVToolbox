import numpy as np 
import cv2

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

from visual_odometry import PinholeCamera, VisualOdometry

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
	vo = VisualOdometry(cam,imgList)
	# xyz= np.array([[0],[0],[0]])
	img = cv2.imread('./Data/'+imgList[0], 0)
	f = open("teat2.txt","w")
	for img_id in range(len(imgList)):
		# img = cv2.imread('./Data/'+imgList[img_id], 0)
		vo.update(img, img_id)
		if img_id > 1:
			cur_t = vo.cur_t
			cur_r = vo.cur_R
			# xyz=cur_r.dot(xyz)+cur_t
			name = imgList[img_id].split('/')[1].split('.')[0]
			pose = transRo2Qu(cur_t,cur_r)
			f.write(name+' '+pose+'\n')
			print(img_id)


def showGroundtruth():
	f = open("./teat3.txt")
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

# import os

# imgList = getImageLists()

# def getKptsDescs(feat_path, img_path):
# 	path = img_path.split('/')
# 	npz_path = feat_path + path[1] + '.npz'
# 	data = np.load(npz_path)
# 	des = data['descs']
# 	kpt = data['kpts']
# 	return kpt, des

# def drawMatchLine(imgList):
# 	root_dir = './Data/'
# 	# 读入图片    
# 	Img1 = cv2.imread(root_dir + imgList[106])
# 	Img2 = cv2.imread(root_dir + imgList[107])
# 	feat_path = './Data/rgb_feats/'
# 	k1, d1 = getKptsDescs(feat_path, imgList[0])
# 	k2, d2 = getKptsDescs(feat_path, imgList[1])
# 	# BFMatcher
# 	FLANN_INDEX_KDTREE = 0
# 	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# 	search_params = dict(checks=50)
# 	flann = cv2.FlannBasedMatcher(index_params,search_params)
# 	matchers = flann.knnMatch(d1,d2,k=2)
#     # 相似列表
# 	Match = []
# 	for m,n in matchers:
# 		if m.distance <  0.50*n.distance:
# 			Match.append(m)
# 	# 查看两张图片的宽及高
# 	height1 , width1 = Img1.shape[:2]
# 	height2 , width2 = Img2.shape[:2]
# 	# 像素调整
# 	vis = np.zeros((max(height1, height2), width1 + width2, 3), np.uint8)
# 	vis[:height1, :width1] = Img1
# 	vis[:height2, width1:width1 + width2] = Img2
# 	p1 = [kpp.queryIdx for kpp in Match[:]]
# 	p2 = [kpp.trainIdx for kpp in Match[:]]
# 	post1 = np.int32([k1[pp] for pp in p1])
# 	post2 = np.int32([k2[pp] for pp in p2]) + (width1, 0)
# 	for (x1, y1), (x2, y2) in zip(post1, post2):
# 		cv2.line(vis, (x1, y1), (x2, y2), (0,0,255))
# 	cv2.namedWindow("match",cv2.WINDOW_NORMAL)
# 	cv2.imshow("match", vis)    
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()

# drawMatchLine(imgList)



# def featureTracking_(image_ref, image_cur):
# 	feat_path = './Data/rgb_feats/'
# 	k1, d1 = getKptsDescs(feat_path, image_ref)
# 	k2, d2 = getKptsDescs(feat_path, image_cur)
# 	# BFMatcher
# 	FLANN_INDEX_KDTREE = 0
# 	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# 	search_params = dict(checks=50)
# 	flann = cv2.FlannBasedMatcher(index_params,search_params)
# 	matchers = flann.knnMatch(d1,d2,k=2)
#     # 相似列表
# 	Match = []
# 	for m,n in matchers:
# 		if m.distance <  0.50*n.distance:
# 			Match.append(m)
# 	p1 = [kpp.queryIdx for kpp in Match[:]]
# 	p2 = [kpp.trainIdx for kpp in Match[:]]
# 	post1 = np.int32([k1[pp] for pp in p1])
# 	post2 = np.int32([k2[pp] for pp in p2]) 
# 	return post1, post2

# p1, p2 = featureTracking_(imgList[106], imgList[107])


# E, mask = cv2.findEssentialMat(p1, p2, focal = 517.3, pp=(318.6,255.3), method=cv2.RANSAC, prob=0.999, threshold=1.0)
# _, R, t, mask = cv2.recoverPose(E, p1, p2, focal = 517.3, pp = (318.6,255.3))

# feat_path = './Data/rgb_feats/'
# k1, d1 = getKptsDescs(feat_path, imgList[106])
# k2, d2 = getKptsDescs(feat_path, imgList[107])
# # BFMatcher
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)
# flann = cv2.FlannBasedMatcher(index_params,search_params)
# matchers = flann.knnMatch(d1,d2,k=2)
# # 相似列表
# Match = []
# for m,n in matchers:
# 	if m.distance <  0.80*n.distance:
# 		Match.append(m)
# p1 = [kpp.queryIdx for kpp in Match[:]]
# p2 = [kpp.trainIdx for kpp in Match[:]]
# post1 = np.int32([k1[pp] for pp in p1])
# post2 = np.int32([k2[pp] for pp in p2]) 



# def align_reconstruction_naive_similarity(X, Xp):
#     """Align with GPS and GCP data using direct 3D-3D matches."""
#     # Compute similarity Xp = s A X + b
#     T = tf.superimposition_matrix(X.T, Xp.T, scale=True)
#     A, b = T[:3, :3], T[:3, 3]
#     s = np.linalg.det(A)**(1. / 3)
#     A /= s
#     return s, A, b

# s, A, b=align_reconstruction_naive_similarity(bb, aa)


# new_b=s*A.dot(bb.T).T+b


def getPose(filename):
	f = open(filename)
	datas=[]
	for line in f:
		if line[0] == '#':
			continue
		data = line.split()
		x = float(data[1] )
		y = float(data[2] )
		z = float(data[3] ) 
		datas.append([x,y,z])
	return np.array(datas)

aa=getPose("./teat.txt")
bb=getPose("./teat3.txt")

def showGroundtruth(aa,bb):
	x = []
	y = []
	z = []
	for i in aa:
		x.append( float(i[0] ) )
		y.append( float(i[1] ) )
		z.append( float(i[2] ) )
	x_ = []
	y_ = []
	z_ = []
	for i in bb:
		x_.append( float(i[0]) )
		y_.append( float(i[1]))
		z_.append( float(i[2]))
	ax = plt.subplot( 111, projection='3d')
	ax.plot(x,y,z,color="r")
	ax.plot(x_,y_,z_,color="b")
	plt.show()

showGroundtruth(aa,bb)


# def align(model,data):
#     numpy.set_printoptions(precision=3,suppress=True)
#     model_zerocentered = model - np.array([model.mean(1)]).T
#     data_zerocentered = data - np.array([data.mean(1)]).T
    
#     W = numpy.zeros( (3,3) )
#     for column in range(model.shape[1]):
#         W += numpy.outer(model_zerocentered[:,column],data_zerocentered[:,column])
#     U,d,Vh = numpy.linalg.linalg.svd(W.transpose())
#     S = numpy.matrix(numpy.identity( 3 ))
#     if(numpy.linalg.det(U) * numpy.linalg.det(Vh)<0):
#         S[2,2] = -1
#     rot = U*S*Vh
#     trans =  np.array([data.mean(1)]).T - rot * np.array([model.mean(1)]).T
    
#     model_aligned = rot * model + trans
#     # alignment_error = model_aligned - data
    
#     # trans_error = numpy.sqrt(numpy.sum(numpy.multiply(alignment_error,alignment_error),0)).A[0]
        
#     return rot,trans

# r,t=align(aa.T,bb.T)