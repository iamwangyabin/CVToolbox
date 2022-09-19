import numpy as np 
import cv2

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500

def getKptsDescs(feat_path, img_path):
	path = img_path.split('/')
	npz_path = feat_path + path[1] + '.npz'
	data = np.load(npz_path)
	des = data['descs']
	kpt = data['kpts']
	return kpt, des

# image_ref -- String -- 前一帧
# image_cur
def featureTracking_(image_ref, image_cur):
	feat_path = './Data/rgb_feats/'
	k1, d1 = getKptsDescs(feat_path, image_ref)
	k2, d2 = getKptsDescs(feat_path, image_cur)
	# BFMatcher
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)
	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matchers = flann.knnMatch(d1,d2,k=2)
    # 相似列表
	Match = []
	for m,n in matchers:
		if m.distance <  0.80*n.distance:
			Match.append(m)
	p1 = [kpp.queryIdx for kpp in Match[:]]
	p2 = [kpp.trainIdx for kpp in Match[:]]
	post1 = np.int32([k1[pp] for pp in p1])
	post2 = np.int32([k2[pp] for pp in p2]) 
	return post1, post2


class PinholeCamera:
	def __init__(self, width, height, fx, fy, cx, cy, 
				k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.distortion = (abs(k1) > 0.0000001)
		self.d = [k1, k2, p1, p2, k3]

class VisualOdometry:
	def __init__(self, cam, imgList):
		self.frame_stage = 0
		self.cam = cam
		self.new_frame = None
		self.last_frame = None
		self.imgList = imgList
		self.cur_R = None
		self.cur_t = None
		self.px_ref = None
		self.px_cur = None
		self.focal = cam.fx
		# principal point
		self.pp = (cam.cx, cam.cy)

	def processFirstFrame(self):
		self.frame_stage = STAGE_SECOND_FRAME

	def processSecondFrame(self):
		# self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		self.px_ref, self.px_cur = featureTracking_(self.last_frame, self.new_frame)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		self.frame_stage = STAGE_DEFAULT_FRAME 
		self.px_ref = self.px_cur

	def processFrame(self, frame_id):
		self.px_ref, self.px_cur = featureTracking_(self.last_frame, self.new_frame)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		self.cur_t = self.cur_t + self.cur_R.dot(t) 
		self.cur_R = R.dot(self.cur_R)
		# self.cur_t =t
		# self.cur_R = R

		self.px_ref = self.px_cur

	# 类的入口，输入一张图片以及它的序号
	def update(self, img, frame_id):
		assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		self.new_frame = self.imgList[frame_id]
		if(self.frame_stage == STAGE_DEFAULT_FRAME):
			self.processFrame(frame_id)
		elif(self.frame_stage == STAGE_SECOND_FRAME):
			self.processSecondFrame()
		elif(self.frame_stage == STAGE_FIRST_FRAME):
			self.processFirstFrame()
		self.last_frame = self.new_frame
