import logging
import numpy as np
import random
from utils import RansacConfig, KeyPt


########################################################################################################################
class EpipolarProcessor:
    """
    Build the epipolar geometry in structure from motion process
    """

    def __init__(self, ransac_config):
        """
        Constructor
        """

        self.fund_mat = np.identity(3)
        self.esse_mat = np.identity(3)
        self.ransac = ransac_config

    #########################################################################
    def determine_fundamental_mat(self, matched_pairs, ransac_config=None):
        """
        Determine fundamental matrix based on the matched pairs

        The left view is the major coordinate.
        The fundamental matrix is used in the IMAGE coordinate
        The essential matrix is used in the NORMALIZED IMAGE coordinate (WHEN BOTH CALIBRATIONS ARE KNOWN)
        The fundamental matrix = right calibration' @ essential matrix @ left calibration
        The 8-point algorithm is implemented here
            a. normalization both left and right matched pairs
            b. SVD to estimate fundamental matrix -> F_head
            c. SVD to optimize F_head to satisfy rank(fun_mat) = 2 -> F_head_head
            d. de-normalization to have the final fundamental matrix
        TODO - improvement : uniformly divide the image into 8x8 inspired by
        "Estimating the Fundamental Matrix by Transforming Points in Projective Space" by Zhengyou Zhang

        @Param matched_pairs
        A list, [ref_pts, que_pts]
        Both ref_pts and que_pts are KeyPt

        @Return inliers-indices
        A list of inlier indices
        """

        matched_pairs_norm, transform_left, transform_right = self.__normalize(matched_pairs)

        if ransac_config is None:
            inliers_indices, fund_mat = self.__estimate_ransac(matched_pairs_norm, self.ransac)
        else:
            inliers_indices, fund_mat = self.__estimate_ransac(matched_pairs_norm, ransac_config)

        self.fund_mat = self.__denormalize(fund_mat, transform_left, transform_right)
        return inliers_indices

    ####################################################################################################################
    def extract_essential_mat(self, left_intrinsic_mat, right_intrinsic_mat):
        """
        Extract the essential matrix from fundamental matrix with left and right calibration
        Considering the normalized camera matrices P = [I | 0] (the left camera) with K and
        P' = [R | t] (the right camera) with K',
        the essential matrix is [t]_x @ R = R[R.transpose() @ t]_x,
        where []_x is skew-symmetric
        Therefore essential matrix = K'.transpose() @ fundamental matrix @ K
        Ref:
        https://www.robots.ox.ac.uk/~vgg/hzbook/hzbook2/HZepipolar.pdf

        @Param left_intrinsic_mat
        Left camera intrinsic matrix, a numpy.array(3, 3)

        @Param right_intrinsic_mat
        Right camera intrinsic matrix, a numpy.array(3, 3)
        """

        # TODO - check left and right intrinsic validity
        left_k = left_intrinsic_mat
        right_k = right_intrinsic_mat

        esse_mat = right_k.transpose() @ self.fund_mat @ left_k

        # Noisy calibration measurement may cause rank(esse_mat) != 2
        # Therefore SVD is applied here
        u, s, vh = np.linalg.svd(esse_mat)
        esse_mat = u @ np.diag([1, 1, 0]) @ vh

        rank = np.linalg.matrix_rank(esse_mat)
        if rank != 2:
            logging.warning('%s : extracted essential matrix is not with rank=2, %d',
                            self.__class__.__name__, rank)
            raise ValueError("esse_mat rank is not equal to 2 : {}".format(rank))

        self.esse_mat = esse_mat / esse_mat[2][2]

    ####################################################################################################################
    def __normalize(self, matched_pairs):
        """
        Normalize the input matched_pairs
            1. the origin of the new coordinate in left/right image coordinate
              is located at the centroid of the all image points (translation)
            2. the mean square distance of the transformed image points from the
              origin is two pixels, [-1, 1] (scaling)
        Ref: https://github.com/Smelton01/8-point-algorithm/blob/master/src/8point.py

        @Param matched_pairs
        A list, [ref_pts, que_pts]
        Both ref_pts and que_pts are KeyPt

        @Return matched_pairs_norm
        Normalized matched pairs

        @Return left_transform
        Normalization transform for left camera

        @Return right_transform
        Normalization transform for right camera
        """
        num_pts = matched_pairs[0].shape[1]
        left_pts = matched_pairs[0][0:2, :].T
        right_pts = matched_pairs[1][0:2, :].T
        left_mean = np.average(left_pts, axis=0)
        right_mean = np.average(right_pts, axis=0)
        left_scale = (2 * num_pts) ** 0.5 / np.sum((np.sum((left_pts - left_mean) ** 2, axis=1)) ** 0.5)
        right_scale = (2 * num_pts) ** 0.5 / np.sum((np.sum((right_pts - right_mean) ** 2, axis=1)) ** 0.5)
        left_transform = np.array([[left_scale, 0, -left_mean[0] * left_scale],
                                   [0, left_scale, -left_mean[1] * left_scale],
                                   [0, 0, 1]])
        right_transform = np.array([[right_scale, 0, -right_mean[0] * right_scale],
                                    [0, right_scale, -right_mean[1] * right_scale],
                                    [0, 0, 1]])
        column_one = np.ones(num_pts).reshape(-1, 1)
        left_pts = np.transpose(np.column_stack((left_pts, column_one)))
        right_pts = np.transpose(np.column_stack((right_pts, column_one)))
        left_pts_norm = np.transpose(left_transform @ left_pts)
        right_pts_norm = np.transpose(right_transform @ right_pts)
        matched_pairs_norm = np.column_stack((left_pts_norm[:, 0:2], right_pts_norm[:, 0:2]))
        return matched_pairs_norm, left_transform, right_transform

    ####################################################################################################################
    def __estimate_eight_pts(self, matched_pairs_8):
        """
        Estimate the fundamental matrix based on 8 pairs of matched points
           through the Lagrangian method
           WF = 0
           where
           W is 8x9, ideally rank(W) = 8
           F is 9x1, ideally rank(F) = 2
           Ref: https://www.cse.psu.edu/~rtc12/CSE486/lecture20_6pp.pdf

        @Param matched_pairs_8
        Eight matched pairs

        @Return f__
        The fundamental matrix
        """
        w = np.zeros((8, 9))
        for idx in range(0, 8):
            x1 = matched_pairs_8[idx][0]
            y1 = matched_pairs_8[idx][1]
            x1_ = matched_pairs_8[idx][2]
            y1_ = matched_pairs_8[idx][3]

            w[idx][0] = x1 * x1_
            w[idx][1] = y1 * x1_
            w[idx][2] = x1_
            w[idx][3] = x1 * y1_
            w[idx][4] = y1 * y1_
            w[idx][5] = y1_
            w[idx][6] = x1
            w[idx][7] = y1
            w[idx][8] = 1.0

        u, s, vh = np.linalg.svd(w)
        vh = vh.transpose()[:, 8]
        f_ = np.reshape(vh, (3, 3))

        # Ideally rank(f_) = 2 (two calibration matrices are rank=3, translation is rank=2, rotation is rank=3)
        # but normally the image points are measured with noise, which causes rank(f_)=3
        # therefore, SVD needs to be applied on f_ to extract the first two eigenvalues and corresponding
        # eigenvectors to construct f__ with rank(f__)=2
        u, s, vh = np.linalg.svd(f_)
        f__ = u @ np.diag([*s[:2], 0]) @ vh

        rank = np.linalg.matrix_rank(f__)
        if rank != 2:
            logging.warning('%s : estimated fundamental matrix is not with rank=2, %d',
                            self.__class__.__name__, rank)
            raise ValueError("f__ rank is not equal to 2 : {}".format(rank))

        f__ = f__ / f__[2][2]
        return f__

    ####################################################################################################################
    def __estimate_ransac(self, matched_pairs, ransac_config):
        """
        Apply RANSAC with eight-point algorithm to find the optimal fundamental matrix

        @Param matched_pairs
        A numpy.array(n,4), where n is the number of pairs
        Each row represents [left_x, left_y, right_x, right_y]

        @Param ransac_config
        A RANSAC configuration setting

        @Return inlier_indices
        A list of inliers' indices

        @Return fund_mat
        The fundamental matrix
        """

        rows, cols = matched_pairs.shape
        fund_mat = np.zeros((3, 3))
        if rows < 8:
            logging.error('%s : number of matched pairs needs equal or more than eight')
            raise ValueError("Insufficient matched pairs : {}".format(rows))
        elif rows == 8:
            fund_mat = self.__estimate_eight_pts(matched_pairs)
            inlier_indices = [0, 1, 2, 3, 4, 5, 6, 7]
        else:  # RANSAC
            max_inlier_num = 0
            inlier_indices = None
            for it in range(0, ransac_config.iteration):

                #  generate the eight indices randomly
                indices = random.sample(range(rows), 8)
                matched_pairs_8 = matched_pairs[indices, :]

                fun_mat_8 = self.__estimate_eight_pts(matched_pairs_8)

                inlier_num = 0
                iindices = []
                for idx in range(0, rows):
                    pt_left = np.array([[matched_pairs[idx][0], matched_pairs[idx][1], 1.0]])
                    pt_right = np.array([[matched_pairs[idx][2], matched_pairs[idx][3], 1.0]])
                    val = abs(pt_right @ fun_mat_8 @ pt_left.T)
                    if val < ransac_config.inlier_threshold:
                        inlier_num += 1
                        iindices.append(idx)

                # Update when the new result is better
                if inlier_num > max_inlier_num:
                    max_inlier_num = inlier_num
                    inlier_indices = iindices
                    fund_mat = fun_mat_8

        return inlier_indices, fund_mat

    ####################################################################################################################
    def __denormalize(self, fund_mat, transform_left, transform_right):
        """
        Denormalize the normalized fundamental matrix

        @Param transform_left
        The left transformation which was applied to normalize the fundamental matrix

        @Param transform_right
        The right transformation which was applied to normalize the fundamental matrix

        @Return ori_fund_mat
        The original fundamental matrix
        """
        ori_fund_mat = transform_right.transpose() @ fund_mat @ transform_left
        ori_fund_mat = ori_fund_mat / ori_fund_mat[2][2]
        return ori_fund_mat


########################################################################################################################
if __name__ == '__main__':
    """ Run the EpipolarProcessor test """

    print('=== Start EpipolarProcessor Unit Test I - Only 8 pts ===')

    # Initialize the testing matched pairs
    # Each row is [leftPtX leftPtY rightPtX rightPtY]
    input_data_ = np.array([[580,  2362, 492,  1803],
                            [2050, 2097, 1381, 1956],
                            [2558, 2174, 1544, 2115],
                            [1395, 1970, 1166, 1752],
                            [2490, 3003, 466,  2440],
                            [3368, 1622, 3320, 2011],
                            [2183, 1500, 2471, 1621],
                            [1972, 1775, 1674, 1736]])

    left_pts_ = KeyPt(8)
    right_pts_ = KeyPt(8)
    left_pts_[0:2, :] = input_data_[:, 0:2].T
    right_pts_[0:2, :] = input_data_[:, 2:4].T
    matched_pairs_ = [left_pts_, right_pts_]

    ITERATION = 200
    THRESHOLD = 1e-3
    ransac_config_ = RansacConfig(inlier_threshold=THRESHOLD,
                                  subset_confidence=0.99,
                                  # the desired probability that the result from this model is 0.99 "reliable"
                                  sample_confidence=0.75,  # (inlier data / total data)
                                  sample_num=8,
                                  iteration=ITERATION)
    ep = EpipolarProcessor(ransac_config_)
    ep.determine_fundamental_mat(matched_pairs_, ransac_config_)

    column_one_ = np.ones(8).reshape(-1, 1)
    lPts = input_data_[:, 0:2]
    lPts = np.column_stack((lPts, column_one_))
    rPts = input_data_[:, 2:4]
    rPts = np.column_stack((rPts, column_one_))

    import cv2 as cv

    [fun_mat_, mask] = cv.findFundamentalMat(lPts, rPts, cv.FM_8POINT)

    np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
    print('    --- fundamental matrix from findFundamentalMat \n{}'.format(ep.fund_mat))
    print('    --- fundamental matrix from OpenCV \n{}'.format(fun_mat_))

    print('    --- Epipolar constraint comparison')
    total_diff = 0
    total_cv = 0
    total_ep = 0
    for idx_ in range(0, 8):
        lPt = lPts[idx_:idx_ + 1, :]
        rPt = rPts[idx_:idx_ + 1, :]
        ep_val = rPt @ ep.fund_mat @ lPt.transpose()
        total_ep += ep_val
        opencv_val = rPt @ fun_mat_ @ lPt.transpose()
        total_cv += opencv_val

        print('    --- point {} : alg value {} vs opencv value {}'.
              format(idx_, abs(ep_val), abs(opencv_val)))
        total_diff += abs(ep_val - opencv_val)

    print('    --- EpipolarProcessor epipolar constraint sum {}'.format(total_ep))
    print('    --- OpenCV epipolar constraint sum {}'.format(total_cv))
    print('    --- constraint sum difference {}'.format(total_diff))
    if total_diff > 1e-2 or total_ep > 1e-2:
        import sys
        print('    --- Fundamental matrix unit test fails')
        sys.exit(-1)
    else:
        print('    --- Fundamental matrix unit test passes')

    logging.debug('%s test pass', __name__)
    print('=== Complete EpipolarProcessor Unit Test I - Only 8 pts ===')

    print('=== Start EpipolarProcessor Unit Test II - Real images ===')

    import os
    from view_processor import ViewProcessor

    cur_path = os.path.dirname(__file__)
    test_dataset_path = os.path.join(cur_path, 'test_dataset', 'epipolar_set')

    # Set up SIFT detector and matcher for the following test
    KEY_TYPE = 'sift'
    IS_CROSS_CHECK = False
    IS_KNN = True
    IS_F_INLINER = False

    # Initialize ViewProcessor and build Views
    vp = ViewProcessor(KEY_TYPE)
    idx_ = 0
    K = np.array([[568.996140852, 0, 643.21055941],
                  [0, 568.988362396, 477.982801038],
                  [0, 0, 1]])

    img_1 = cv.imread(test_dataset_path + '/image1.jpg')
    img_2 = cv.imread(test_dataset_path + '/image2.jpg')

    from utils import get_data_from_txt_file

    points_1 = get_data_from_txt_file(test_dataset_path + '/pt_2D_1.txt')
    points_2 = get_data_from_txt_file(test_dataset_path + '/pt_2D_2.txt')

    points_1_xy = points_1[:, 0:2]
    points_2_xy = points_2[:, 0:2]
    matched_pairs_ = [points_1_xy.T, points_2_xy.T]
    # matched_pairs_ = np.hstack((points_1_xy, points_2_xy))

    ITERATION = 300
    THRESHOLD = 1
    ransac_config_ = RansacConfig(inlier_threshold=THRESHOLD,
                                  subset_confidence=0.99,
                                  # the desired probability that the result from this model is 0.99 "reliable"
                                  sample_confidence=0.75,  # (inlier data / total data)
                                  sample_num=8,
                                  iteration=ITERATION)
    inliers_indices_ = ep.determine_fundamental_mat(matched_pairs_, ransac_config_)
    print('    --- There are {} inliers in {} pts'.format(len(inliers_indices_), matched_pairs_[0].shape[1]))

    [fun_mat_, mask] = cv.findFundamentalMat(points_1, points_2, cv.FM_8POINT)

    print('    --- fundamental matrix from findFundamentalMat \n{}'.format(ep.fund_mat))
    print('    --- fundamental matrix from OpenCV \n{}'.format(fun_mat_))

    total_pFp_ep = np.zeros((1, points_1.shape[0]))
    total_pFp_cv = np.zeros((1, points_1.shape[0]))
    for i in range(points_1.shape[0]):
        pFp_ep = points_2[i] @ ep.fund_mat @ points_1[i].T
        total_pFp_ep[0, i] = np.abs(pFp_ep)

        pFp_cv = points_2[i] @ fun_mat_ @ points_1[i].T
        total_pFp_cv[0, i] = np.abs(pFp_cv)

    """
    print('    --- EpipolarProcessor epipolar constraint average {}'.format(np.average(total_pFp_ep)))
    print('    --- OpenCV epipolar constraint average {}'.format(np.average(total_pFp_cv)))

    print('    --- EpipolarProcessor epipolar constraint std {}'.format(np.std(total_pFp_ep)))
    print('    --- OpenCV epipolar constraint std {}'.format(np.std(total_pFp_cv)))

    print('    --- EpipolarProcessor epipolar constraint max {}'.format(np.max(total_pFp_ep)))
    print('    --- OpenCV epipolar constraint max {}'.format(np.max(total_pFp_cv)))

    print('    --- EpipolarProcessor epipolar constraint min {}'.format(np.min(total_pFp_ep)))
    print('    --- OpenCV epipolar constraint min {}'.format(np.min(total_pFp_cv)))
    """

    if np.average(total_pFp_ep) >= 1:
        print("    --- EpipolarProcessor result average fails : {:.6f}".format(np.average(total_pFp_ep)))
    else:
        print("    --- EpipolarProcessor result average passes : {:.6f}".format(np.average(total_pFp_ep)))

    """
    from utils import compute_distance_to_epipolar_lines
    print("    --- EpipolarProcessor Distance to lines in image 1 for normalized :",
          compute_distance_to_epipolar_lines(points_1, points_2, ep.fund_mat))
    print("    --- EpipolarProcessor Distance to lines in image 2 for normalized :",
          compute_distance_to_epipolar_lines(points_2, points_1, ep.fund_mat.T))

    print("    --- OpenCV Distance to lines in image 1 for normalized :",
          compute_distance_to_epipolar_lines(points_1, points_2, fun_mat_))
    print("    --- OpenCV Distance to lines in image 2 for normalized :",
          compute_distance_to_epipolar_lines(points_2, points_1, fun_mat_.T))
    """

    # for debugging
    # Plotting the epipolar lines
    # import matplotlib.pyplot as plt
    # from utils import plot_epipolar_lines_on_images
    # plt.figure("Epipolar Processor")
    # plot_epipolar_lines_on_images(points_1, points_2, img_1, img_2, ep.fund_mat)
    # plt.figure("OpenCV")
    # plot_epipolar_lines_on_images(points_1, points_2, img_1, img_2, F)
    # plt.show()
    # for debugging

    print('=== Complete EpipolarProcessor Unit Test II - Real images ===')
