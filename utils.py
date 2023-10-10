import math
import numpy as np


########################################################################################################################
class KeyPt(np.ndarray):
    """
    Represent 2D point in the homogeneous coordinate
    """
    def __new__(cls, num):
        ret = np.zeros((3, num), dtype=float)
        ret[2:3, :] = 1.0
        return ret.view(cls)


########################################################################################################################
class TriPt(np.ndarray):
    """
    Represent 3D point in the homogeneous coordinate
    """
    def __new__(cls, num):
        ret = np.zeros((4, num), dtype=float)
        ret[3:4, :] = 1.0
        return ret.view(cls)


########################################################################################################################
def convert_rotation_to_quaternion(rot_mat):
    """
    Convert the rotation matrix to the quaternion.

    @Param rot_mat
    A rotation matrix, numpy.array(3, 3)

    @Return quat
    A converted quaternion, numpy.array(4, 1)

    Ref:
    https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
    """

    # Check sanity of the rotation matrix
    if not verify_rotation_mat(rot_mat):
        raise ValueError('{} : Invalid input rotation matrix \n {}'.format(
            "convert_rotation_to_quaternion", rot_mat))

    qw = math.sqrt(1 + rot_mat[0][0] + rot_mat[1][1] + rot_mat[2][2]) / 2.0

    if abs(qw - 0) < 1e-6:
        raise ValueError('{} : Invalid output qw \n {}'.format(
            "convert_rotation_to_quaternion", qw))

    # normalize
    qx = (rot_mat[2][1] - rot_mat[1][2]) / (4 * qw)
    qy = (rot_mat[0][2] - rot_mat[2][0]) / (4 * qw)
    qz = (rot_mat[1][0] - rot_mat[0][1]) / (4 * qw)

    quat = np.array([[qw, qx, qy, qz]])
    quat = quat.T
    return quat


########################################################################################################################
def convert_quaternion_to_rotation(quaternion):
    """
    Convert a quaternion to a rotation matrix

    @Param quaternion
    A quaternion

    @Return rot_mat
    A converted rotation matrix
    """

    q = quaternion
    q_sq = np.square(q)
    w = 0
    x = 1
    y = 2
    z = 3

    rot_mat = np.zeros((3, 3))
    rot_mat[0][0] = 1 - 2 * q_sq[z] - 2 * q_sq[y]
    rot_mat[0][1] = -2 * q[z] * q[w] + 2 * q[y] * q[x]
    rot_mat[0][2] = 2 * q[y] * q[w] + 2 * q[z] * q[x]
    rot_mat[1][0] = 2 * q[x] * q[y] + 2 * q[w] * q[z]
    rot_mat[1][1] = 1 - 2 * q_sq[z] - 2 * q_sq[x]
    rot_mat[1][2] = 2 * q[z] * q[y] - 2 * q[x] * q[w]
    rot_mat[2][0] = 2 * q[x] * q[z] - 2 * q[w] * q[y]
    rot_mat[2][1] = 2 * q[y] * q[z] + 2 * q[w] * q[x]
    rot_mat[2][2] = 1 - 2 * q_sq[y] - 2 * q_sq[x]

    if not verify_rotation_mat(rot_mat):
        raise ValueError('{} : Invalid output rotation matrix \n {}'.format(
            "convert_quaternion_to_rotation", rot_mat))

    return rot_mat


########################################################################################################################
def verify_rotation_mat(rot):
    if rot.shape != (3, 3) or np.linalg.det(rot) - 1 >= 1e-8 or np.any((np.linalg.inv(rot) - rot.T) > 1e-8):
        return False
    else:
        return True


########################################################################################################################
class LMConfig:
    """
    Configruation for Levenberg–Marquardt algorithm (LMA).

    @Param stop_ratio
    The improvement ratio to decide if the process can stop.
    If abs(current error - previous error) / (previous error) < stop_ratio,
    it means we do not see obvious improvement, so we stop the process before
    reaching the number of iteration

    @Param iteration
    The maximum iteration steps the process executes.
    """

    def __init__(self, stop_ratio, iteration):
        self.stop_ratio = stop_ratio
        self.iteration = iteration


########################################################################################################################
class RansacConfig:
    """
    Configuration for RANSAC.

    @Param inlier_threshold
    The threshold to determine if inlier

    @Param subset_confidence
    The probability that at least one of subsets does not include any outlier
    (the desired probability that we at least get a good subset)
    Higher means the result from this model is more reliable.
    Each subset contains sample_num samples

    @Param sample_confidence
    The probability that any selected point is an inlier
    number of inliers in data / number of points in data

    @Param sample_num
    The number of sampled data

    @Param iteration
    The iteration times

    """

    def __init__(self, inlier_threshold, subset_confidence,
                 sample_confidence, sample_num, iteration, is_use_seed=True):
        # TODO - check validity of confidence [0, 1.0]

        self.inlier_threshold = inlier_threshold
        self.subset_confidence = subset_confidence
        self.sample_confidence = sample_confidence
        self.sample_num = int(sample_num)
        self.iteration = int(iteration)
        self.random_seed = -1

        # Confirm if the passed interation is sufficient
        calc_iteration = math.log(1.0 - subset_confidence) / math.log(1.0 - math.pow(sample_confidence, sample_num))
        if calc_iteration > iteration:
            print('RANSAC : iteration increases from {} to {}'.format(iteration, calc_iteration))
            self.iteration = int(calc_iteration)

        # Set up the random seed
        if is_use_seed:
            import random
            random.seed(self.random_seed)


# if __name__ == '__main__':
#     config1 = RansacConfig(8.0, 0.99, 0.8, 6, 100)
#     print('config1 iteration : {}'.format(config1.iteration))
#
#     config2 = RansacConfig(8.0, 0.8, 0.99, 6, 100)
#     print('config2 iteration : {}'.format(config2.iteration))
#
#     print('=== Done ===')


########################################################################################################################
# From ps2
'''
GET_DATA_FROM_TXT_FILE
Arguments:
    filename - a path (str) to the data location
Returns:
    points - a matrix of points where each row is either:
        a) the homogenous coordinates (x,y,1) if the data is 2D
        b) the coordinates (x,y,z) if the data is 3D
    use_subset - use a predefined subset (this is hard coded for now)
'''
def get_data_from_txt_file(filename, use_subset=False):
    with open(filename) as f:
        lines = f.read().splitlines()
    number_pts = int(lines[0])

    import numpy as np
    points = np.ones((number_pts, 3))
    for i in range(number_pts):
        split_arr = lines[i + 1].split()
        if len(split_arr) == 2:
            y, x = split_arr
        else:
            x, y, z = split_arr
            points[i, 2] = z

        points[i, 0] = x
        points[i, 1] = y
    return points


########################################################################################################################
# From ps2
'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    ln1 = F.T.dot(points2.T)
    for i in range(ln1.shape[1]):
        plt.plot([0, im1.shape[1]],
                 [-ln1[2][i] * 1.0 / ln1[1][i], -(ln1[2][i] + ln1[0][i] * im1.shape[1]) * 1.0 / ln1[1][i]], 'r')
        plt.plot([points1[i][0]], [points1[i][1]], 'b*')
    plt.imshow(im1, cmap='gray')

    plt.subplot(1, 2, 2)
    ln2 = F.dot(points1.T)
    for i in range(ln2.shape[1]):
        plt.plot([0, im2.shape[1]], [-ln2[2][i] * 1.0 / ln2[1][i], -(ln2[2][i] + ln2[0][i] * im2.shape[1]) / ln2[1][i]],
                 'r')
        plt.plot([points2[i][0]], [points2[i][1]], 'b*')
    plt.imshow(im2, cmap='gray')


########################################################################################################################
# From ps2
'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    import numpy as np

    l = F.T.dot(points2.T)
    dist_sum = 0.0
    points_num = points1.shape[0]

    for i in range(points_num):
        dist_sum += np.abs(points1[i][0] * l[0][i] + points1[i][1] * l[1][i] + l[2][i]) * 1.0 \
                    / np.sqrt(l[0][i] ** 2 + l[1][i] ** 2)
    return dist_sum / points_num

# class key_point(keypoint):
#     def __init__(self, view_idx, key_idx):
#         self.view_idx = view_idx
#         self.key_idx = key_idx
#
# class triangulated_point:
#     def __init__(self, ref_key_pt, que_key_pt, matched_dist):
#         self.ref_pt = ref_key_pt
#         self.que_pt = que_key_pt
#         self.dist = matched_dist


# import numpy as np
# import cv2 as cv
# from triangulation_processor import ReconKeyPointMatch


#########################################################################################################################
#
#
# class MatchMat:
#    """Represents the match matrix"""
#
#    def __init__(self, qIdx, tIdxL, mat):
#        """
#        :param qIdx: query index (integer)
#        :param tIdxL: train indices (integer list)
#        :param mat: match matrix (numpy.ndarray)
#            Each row represents a key point (key) index in the qIdx-th image.
#            Each column represents the matched key point index in column-idx-th image.
#        """
#
#        if len(tIdxL) != mat.shape[1]:
#            raise AssertionError("tIdxL size is different from mat's column size")
#
#        self.qIdx = qIdx
#        self.tIdxL = tIdxL
#        self.mat = mat
#
#        #  Initialize processed flags
#        self.processedL = [False] * len(tIdxL)
#
#        #  Count the matched key points
#        self.matchedCount = [0] * len(tIdxL)
#        for idx, row in enumerate(mat.T):
#            self.matchedCount[idx] = (row != -1).sum()
#
#########################################################################################################################
#
#
# def buildMatchMat(qIdx, tIdxL, qKeyPtL, matchLL):
#    """
#    :brief: build the match matrix based on query key points and the corresponding matches in each train view
#    :param qKeyPtL: query key points (list of cv2.KeyPoint)
#    :param matchLL: list of matches (list of lists of cv2.DMatch)
#    """
#    rowSize = len(qKeyPtL)
#    colSize = len(matchLL)
#    matrix = np.full((rowSize, colSize), -1).astype(int)
#    for colIdx, matchL in enumerate(matchLL):
#        for matchIdx, match in enumerate(matchL):
#            rowIdx = match.queryIdx
#            matrix[rowIdx, colIdx] = match.trainIdx
#    return MatchMat(qIdx, tIdxL, matrix)
#
#########################################################################################################################
#
#
# def selectInitMatchPair(matchMat):
#    """
#    :brief: select the initial matched pair indices
#    :param matchMat: match matrix (class matchMat)
#    """
#
#    # Current implementation: choose the first element in the match matrix
#    # ToDo: experiment the maximum numIndices method
#    if matchMat:
#        return 0, 1
#    else:
#        raise AssertionError("matchMat cannot be None")
#
#
#########################################################################################################################
#
#
# def GetNextBestView(matchMat, latestProcessedIdx):
#    """Get the best next view index to process"""
#
#    # Current implementation: choose the next consecutive index based on the latest processed index
#    # ToDo: experiment the window search method to find the "next" best view to process
#
#    viewIdx = -1
#    if matchMat.processedL[latestProcessedIdx+1] == False:
#        viewIdx = latestProcessedIdx+1
#    return viewIdx
#
#
#########################################################################################################################
#
#
# def Find2D3DMatch(recon3DPointL, nextMatch):
#    """
#    @brief
#        Find and generate the correspondence list between reconstructed 3D points and match.
#    @param recon3DPointL
#        A reconstructed 3D point list.
#    @param match
#        A match object (class Match), in which the reference view SHOULD be already associated to some 3D points in recon3DPointL.
#    @return
#        The 3D point index list and the associated 2D point index (match::queView) list.
#    """
#
#
#    # Current implementation: increasing view index comparison
#    # ToDo: bidirectional view index comparison
#    point3DCount = 0
#    point3DIdxL = []
#    point2DIdxL = []
#    for pp in recon3DPointL:  # Loop through all reconstructed 3D points
#        for mm in pp.matchL:  # Loop through all corresponded 2D matches which is associated to this 3D point
#            if mm.queView.idx == nextMatch.refView.idx:
#                pp.matchL.append(ReconKeyPointMatch(cc.queViewIdx,
#                                                             cc.queKeyPointIdx,
#                                                             nextMatch.queView.idx,
#                                                             cc.queKeyPointIdx))  # ToDo: DOUBLE CHECK
#                point3DIdxL.append(point3DCount)
#                point2DIdxL.append(cc.queKeyPointIdx)
#        point3DCount += 1
#
#    return point3DIdxL, point2DIdxL
#
#########################################################################################################################
#
