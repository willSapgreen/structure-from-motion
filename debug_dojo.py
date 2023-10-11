import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

DOJO_DEBUG = True

# Ref
# https://python.hotexamples.com/examples/cv2/-/findFundamentalMat/python-findfundamentalmat-function-examples.html
# https://stackoverflow.com/posts/62577144/revisions
# https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
# https://answers.opencv.org/question/198880/recoverpose-and-triangulatepoints-3d-results-are-incorrect/

# OpenCV solvePnP vs recoverPose
# https://stackoverflow.com/questions/51914161/solvepnp-vs-recoverpose-by-rotation-composition-why-translations-are-not-same

# OpenCV pose coordinate
# https://stackoverflow.com/questions/37810218/is-the-recoverpose-function-in-opencv-is-left-handed

cur_path = os.path.dirname(__file__)

# test_dataset_path = os.path.join(cur_path, 'test_dataset')
# img_path = os.path.join(test_dataset_path, "image0000001.bmp")

test_dataset_path = os.path.join(cur_path, 'test_dataset/upenn')
img_path = os.path.join(test_dataset_path, "image0000001.bmp")
imgl = cv2.imread(img_path)

img_path = os.path.join(test_dataset_path, "image0000002.bmp")
imgr = cv2.imread(img_path)

sift = cv2.SIFT_create() #cv2.SIFT()

# find the keypoints and descriptors with SIFT
kpl, desl = sift.detectAndCompute(imgl, None)
kpr, desr = sift.detectAndCompute(imgr, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desl, desr, k=2)
good = []
ptsl = []
ptsr = []

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        good.append(m)
        ptsr.append(kpr[m.trainIdx].pt)
        ptsl.append(kpl[m.queryIdx].pt)

print('matched num {}'.format(len(ptsl)))

intrinsicMat = np.array([[568.996140852, 0, 643.21055941],
              [0, 568.988362396, 477.982801038],
              [0, 0, 1]])

ptsr = np.float32(ptsr)
ptsl = np.float32(ptsl)

# pts_l_norm = cv2.undistortPoints(np.expand_dims(ptsl, axis=1), cameraMatrix=intrinsicMat, distCoeffs=None)
# pts_r_norm = cv2.undistortPoints(np.expand_dims(ptsr, axis=1), cameraMatrix=intrinsicMat, distCoeffs=None)
pts_l_norm = ptsl
pts_r_norm = ptsr

F, mask = cv2.findFundamentalMat(pts_l_norm, pts_r_norm, cv2.FM_LMEDS)
E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, intrinsicMat, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# R and t are from destination to reference in the destination coordinate
points, R, t, mask = cv2.recoverPose(E, pts_l_norm, pts_r_norm)

print("Fundamental matrix \n {}".format(F))
print("Essential matrix \n {}".format(E))
print('nubmer used for recover pose {}'.format(points))

print("Rotation matrix \n {}".format(R))
print("translation vector \n {}".format(t))

# The euler angles in the reference coordinate
from scipy.spatial.transform import Rotation as Rot
euler_angle_res = Rot.from_matrix(R.transpose()).as_euler('zyx', degrees=True)
print("euler angles from reference to destination in reference coordinate \n {}".format(euler_angle_res))

# The destination location in the reference coordinate
C = -R.T @ t
print("location C in reference coordinate \n {}".format(C.T))

M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
M_r = np.hstack((R, t))
#
# if DOJO_DEBUG:
#     l_2d_pt = np.array([[522.19512939, 201.47183228]])
#     r_2d_pt = np.array([[512.17010498, 199.26965332]])
#
#     pts_l_norm = np.vstack((l_2d_pt, pts_l_norm))
#     pts_r_norm = np.vstack((r_2d_pt, pts_r_norm))

P_l = np.dot(intrinsicMat,  M_l)
P_r = np.dot(intrinsicMat,  M_r)

# cv2.triangulatePoints() is equal to my linear pnp result
point_4d_hom = cv2.triangulatePoints(P_l, P_r,
                                     np.expand_dims(pts_l_norm, axis=1),
                                     np.expand_dims(pts_r_norm, axis=1))
point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
point_3d = point_4d[:3, :].T
print('1st tri 3d pt \n {}'.format(point_3d[0, :]))

# fig = plt.figure()
# fig.add_subplot(131, facecolor='red')
# fig.add_subplot(132, facecolor='green')
# fig.add_subplot(133, facecolor='blue')
# fig.add_subplot(231, facecolor='black')
# fig.add_subplot(235, facecolor='cyan')
# plt.show()