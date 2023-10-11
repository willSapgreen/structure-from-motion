import logging

import cv2 as cv
import numpy as np
from epipolar_processor import EpipolarProcessor
from utils import RansacConfig, KeyPt

INVALID_MATCH_VAL = NOT_USED_TRI_VAL = -1
MATCH_ARR_COLS = 4  # ref_x, ref_y, que_x, que_y
RATIO = 0.7


########################################################################################################################
class KeyTrack:
    """
    Generate a key track for a view
    """

    def __init__(self, rows, cols, idx):

        # self.table is a numpy.array((m, n))
        # where m is the number of views
        #       n is the number of key points in this idx-th view
        #       self.table[idx, x] (itself) represents
        #       if this x-th key point is used for triangulation (USED_TRI_VAL) or not
        #       self.table[non_idx, x] represents
        #       if this x-th key point in the idx-th view
        #       has a match (not INVALID_MATCH_VAL value) or not (INVALID_MATCH_VAL) in the non_idx-th view
        self.table = np.empty((rows, cols), dtype='int')
        self.table.fill(INVALID_MATCH_VAL)

        self.idx = idx
        self.key_num = cols

    #########################################################################
    def expand_table(self):
        """
        Only expand vertically
        """
        arr = np.full((1, self.key_num), INVALID_MATCH_VAL, dtype='int')
        self.table = np.append(self.table, arr, 0)

    #########################################################################
    def update_usage(self, used_indices, tri_indices):
        for idx, val in np.ndenumerate(used_indices):
            self.table[self.idx, val] = tri_indices[0, idx[1]]

    #########################################################################
    def extract_unconstructed_points(self):
        indices = np.where(self.table[self.idx, :] == NOT_USED_TRI_VAL)
        indices = np.asarray(indices)
        return indices

    #########################################################################
    def extract_constructed_points(self):
        indices = np.where(self.table[self.idx, :] != NOT_USED_TRI_VAL)
        indices = np.asarray(indices)
        vals = np.take(self.table[self.idx, :], indices)
        return indices, vals


########################################################################################################################
class KeyTracker:
    """
    Generate the key tracking map for views
    Design for the real-time processing,
    which means one new image is given per time.

    """

    def __init__(self, key_type, is_cross_check, is_knn_match, is_fund_inlier, ransac_config):
        """
        Constructor

        @Param key_type
        A key type to be used for key extraction.

        @Param is_cross_check
        A flag to indicate if cross check in OpenCV BFMatcher is enabled or not

        """
        if key_type in ['sift', 'surf']:
            self.matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=is_cross_check)
        else:
            self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=is_cross_check)

        self.key_type = key_type
        self.is_cross_check = is_cross_check
        self.is_knn_match = is_knn_match
        self.is_fund_inlier = is_fund_inlier
        self.ransac_config = ransac_config

        # Store KeyTrack objects
        self.track_list = []  # the matched index map

    ####################################################################################################################
    def add_new_view(self, new_view, views, is_knn_match=None, is_fund_inlier=None, ransac_config=None):
        """
        Add new view

        @Param new_view
        A new view to be added into KeyTracker

        @Param is_knn_match
        A flag to indicate whether KNN match is applied

        @Param is_fund_inlier
        A flag to indicate wheter using the fundamental matrix to extract inlier key point pairs

        @Param ransac_config
        If if_f_inlier is true, ransac_config is used to initialize EpipolorProcessor
        """
        # Use the object's values if users don't provide
        if not is_knn_match:
            is_knn_match = self.is_knn_match
        if not is_fund_inlier:
            is_fund_inlier = self.is_fund_inlier
        if not ransac_config:
            ransac_config = self.ransac_config

        if len(self.track_list) == 0:  # first view
            key_num = len(new_view.key_pts)
            # print('--- key_num : {}'.format(key_num))
            self.track_list.append(KeyTrack(1, key_num, 0))
        else:
            self.__extend_list(new_view, views, is_knn_match, is_fund_inlier, ransac_config)

        # add the new_view
        # self.view_list.append(new_view)

    ####################################################################################################################
    def generate_matched_pairs(self, ref_idx, que_idx, views):
        """
        Return a list [ref_idx_pts, que_idx_pts]
        ref_idx_pts and que_idx_pts are KeyPt(n)
        n is the number of points.

        @Param views
        Views object from the ViewProcessor.

        @Return matched_pairs
        A list of two np.array.
        Each array represents x key points as (3, x).

        @Return r_indices
        A np.array represents the indices of key points in
        views[ref_idx] as (1, x)

        @Return q_indices
        A np.array represents the indices of key points in
        views[que_idx] as (1, x)
        """

        if ref_idx < 0 or que_idx < 0 or ref_idx >= len(self.track_list) or que_idx >= len(self.track_list):
            from inspect import currentframe
            print('{}:{} - invalid ref_idx {} or invalid que_idx {}'.format(
                self.__class__.__name__, currentframe().f_code.co_name,
                ref_idx, que_idx))
            return None
        else:
            row = self.track_list[ref_idx].table[que_idx:que_idx + 1, :]
            indices = np.where(row > 0)[1]
            num = indices.shape[0]
            matched_pairs = []
            ref_pts = KeyPt(num)
            que_pts = KeyPt(num)
            r_indices = np.zeros((1, num), dtype=int)
            q_indices = np.zeros((1, num), dtype=int)
            for idx, r_idx in np.ndenumerate(indices):
                q_idx = row[0, r_idx]
                ref_pts[0, idx] = views[ref_idx].key_pts[r_idx].pt[0]
                ref_pts[1, idx] = views[ref_idx].key_pts[r_idx].pt[1]
                que_pts[0, idx] = views[que_idx].key_pts[q_idx].pt[0]
                que_pts[1, idx] = views[que_idx].key_pts[q_idx].pt[1]

                r_indices[0, idx] = r_idx
                q_indices[0, idx] = q_idx

            matched_pairs.append(ref_pts)
            matched_pairs.append(que_pts)
            return matched_pairs, r_indices, q_indices

    ####################################################################################################################
    def find_best_view(self, input_idx):
        """
        Return the best frame index based on input_idx
        """
        if input_idx < 0 or input_idx >= len(self.track_list):
            from inspect import currentframe
            print('{}:{} - invalid input_idx {}'.format(
                self.__class__.__name__, currentframe().f_code.co_name, input_idx))
            return -1
        else:
            #  For now, return the 1st frame as the best
            return 0

    ####################################################################################################################
    def is_visible(self, view_idx, tri_pt_idx):
        key_idx = np.where(self.track_list[view_idx].table[view_idx, :] == tri_pt_idx)
        if np.any(key_idx):
            key_idx = key_idx[0][0]
        else:
            key_idx = -1
        return key_idx

    ####################################################################################################################
    def clear(self):
        """Clear the existing maps"""
        # self.view_list.clear()
        self.track_list = []

    ####################################################################################################################
    def __extend_list(self, new_view, views, is_knn_match=False, is_fund_inlier=False, ransac_config=None):
        """
        Extend the key map with the new_view

        @Param new_view
        A new view to be added into KeyTracker

        @Param is_knn_match
        A flag to indicate whether KNN match is applied

        @Param is_fund_inlier
        A flag to indicate wheter using the fundamental matrix to extract inlier key point pairs

        @Param ransac_config
        If if_f_inlier is true, ransac_config is used to initialize EpipolorProcessor
        """

        # Get the key numbers in new view,
        # and the current number of tracks in this tracker
        key_num = len(new_view.key_pts)
        new_track_idx = len(self.track_list)

        # Expand the existing tracks to store matched indices with the new track
        for track in self.track_list:
            track.expand_table()

        # Create a new track for the new view
        new_track = KeyTrack(len(self.track_list) + 1, key_num, new_track_idx)

        # Extract key points and descriptors from the new view
        que_key_points = new_view.key_pts
        que_key_descriptors = new_view.key_descriptors

        # Loop through all existing views
        for ref_idx, ref_view in enumerate(views):
            ref_key_points = ref_view.key_pts
            ref_key_descriptors = ref_view.key_descriptors
            ref_track_idx = ref_idx

            # Generate the matched result
            # matches = None
            if is_knn_match:
                k = 2
                if self.is_cross_check:
                    #  k must be 1 when matcher is created with crossCheck=True.
                    k = 1
                matches = self.matcher.knnMatch(que_key_descriptors,
                                                ref_key_descriptors, k=k)
            else:
                matches = self.matcher.match(que_key_descriptors,
                                             ref_key_descriptors)
            # Note: matches size may be smaller than the query descriptors count

            # remove empty tuples or element
            matches = [item for item in matches if item]

            # knnMatch returns tuple-type result
            if is_knn_match:
                matches = self.__process_knn_result(matches)

            # In matches,
            # a trainIdx can be assigned to multiple queryIdx elements.
            # Here we remove the duplicate ones and keep the best one with the shortest distance.
            matches_unique = []
            inlier_indices = []  # By default, all non-duplicate matching pairs are inliers.
            for match_idx in range(len(matches)):
                try:
                    dup_idx = inlier_indices.index(matches[match_idx].trainIdx)

                    # Compare the duplicate pairs and update if necessary
                    if matches[match_idx].distance < matches[dup_idx].distance:
                        inlier_indices[dup_idx] = matches[match_idx].trainIdx
                        matches_unique[dup_idx] = matches[match_idx]
                except ValueError:
                    # Does not find any duplicate one, GOOD
                    matches_unique.append(matches[match_idx])
                    inlier_indices.append(matches[match_idx].trainIdx)
                    pass
            matches = matches_unique

            # Update inlier_indices with the fundamental matrix if enable
            if is_fund_inlier:
                match_arr = self.__build_key_match_arr(ref_key_points,
                                                       que_key_points,
                                                       matches)
                ep = EpipolarProcessor(ransac_config)
                inlier_indices = ep.determine_fundamental_mat(match_arr, ransac_config)

            inlier_ref_incides = []
            for idx in range(len(inlier_indices)):
                inlier_ref_incides.append(matches[idx].trainIdx)

            # Loop through all matches to fill up tracks
            ref_track = self.track_list[ref_idx]
            for i in range(len(matches)):
                ref_key_idx = matches[i].trainIdx

                # Only process when the key point is an inlier
                if ref_key_idx in inlier_ref_incides:
                    que_key_idx = matches[i].queryIdx
                    ref_track.table[new_track_idx, ref_key_idx] = que_key_idx
                    new_track.table[ref_track_idx, que_key_idx] = ref_key_idx

        # Add the new track into the list
        self.track_list.append(new_track)

    ####################################################################################################################
    def __build_key_match_arr(self, ref_key_pts, que_key_pts, matches):
        # match_arr = np.zeros((len(matches), MATCH_ARR_COLS))
        match_arr = []
        ref_pts = KeyPt(len(matches))
        que_pts = KeyPt(len(matches))
        for arr_idx, match in enumerate(matches):
            ref_idx = match.trainIdx
            que_idx = match.queryIdx
            ref_pts[0, arr_idx] = ref_key_pts[ref_idx].pt[0]
            ref_pts[1, arr_idx] = ref_key_pts[ref_idx].pt[1]
            que_pts[0, arr_idx] = que_key_pts[que_idx].pt[0]
            que_pts[1, arr_idx] = que_key_pts[que_idx].pt[1]
        match_arr.append(ref_pts)
        match_arr.append(que_pts)
        return match_arr

    ####################################################################################################################
    def __process_knn_result(self, matches):
        if self.is_cross_check:
            matches = [item[0] for item in matches if item]  # unpack tuple when generated from knnMatch
        else:
            # Use the ratio test to filter out bad matches
            matches = [item[0] for item in matches if (item[0].distance / item[1].distance) < RATIO]

        return matches


########################################################################################################################
if __name__ == '__main__':
    """ Run the KeyTracker test """

    print('=== Start KeyTracker Unit Test ===')

    import os
    from view_processor import ViewProcessor, IMG_EXT

    cur_path = os.path.dirname(__file__)
    test_dataset_path = os.path.join(cur_path, 'test_dataset', 'upenn')

    # Set up testing params
    KEY_TYPE = 'sift'
    ITERATION = 300
    THRESHOLD = 1e-2
    IS_FUND_INLINER = True
    RANSAC_CONFIG = RansacConfig(inlier_threshold=THRESHOLD,
                                 subset_confidence=0.99,
                                 # the desired probability that the result from this model is 0.99 "reliable"
                                 sample_confidence=0.75,  # (inlier data / total data)
                                 sample_num=8,
                                 iteration=ITERATION)

    # Initialize ViewProcessor and build Views
    vp = ViewProcessor(KEY_TYPE)
    idx_ = 0
    K = np.array([[568.996140852, 0, 643.21055941],
                  [0, 568.988362396, 477.982801038],
                  [0, 0, 1]])
    # views = []
    IS_CROSS_CHECK = False
    IS_KNN = True
    kt_no_f = KeyTracker(KEY_TYPE, IS_CROSS_CHECK, IS_KNN, False, RANSAC_CONFIG)
    kt_f = KeyTracker(KEY_TYPE, IS_CROSS_CHECK, IS_KNN, True, RANSAC_CONFIG)
    for file_name in os.listdir(test_dataset_path):
        if file_name.endswith(IMG_EXT):
            img_path = os.path.join(test_dataset_path, file_name)
            img = cv.imread(img_path)

            # Generate the view
            view = vp.generate_view(img, idx_, K)
            kt_no_f.add_new_view(view, vp.view_list, IS_KNN, False, RANSAC_CONFIG)
            kt_f.add_new_view(view, vp.view_list, IS_KNN, True, RANSAC_CONFIG)
            vp.add_view(view)

            # views.append(view)
            idx_ += 1

    # Create a key tracker with inlier disable
    # IS_CROSS_CHECK = False
    # IS_KNN = True
    # IS_FUND_INLINER = False
    # kt_no_f = KeyTracker(KEY_TYPE, IS_CROSS_CHECK, IS_KNN, IS_FUND_INLINER, RANSAC_CONFIG)
    # for view in views:
    #     kt_no_f.add_new_view(view, IS_KNN, IS_FUND_INLINER)

    # Crate a key tracker with inlier enable
    # IS_FUND_INLINER = True
    # kt_f = KeyTracker(KEY_TYPE, IS_CROSS_CHECK, IS_KNN, IS_FUND_INLINER, RANSAC_CONFIG)
    # for view in views:
    #     kt_f.add_new_view(view, IS_KNN, IS_FUND_INLINER, RANSAC_CONFIG)

    # Check if inlier enable produces less matched pairs
    is_inlier_pass = True
    for idx_data in range(len(kt_f.track_list)):
        ft_no_f_inlier_num = np.where(kt_no_f.track_list[idx_data].table > 0)[0].shape[0]
        ft_f_inlier_num = np.where(kt_f.track_list[idx_data].table > 0)[0].shape[0] > 0
        if ft_f_inlier_num >= ft_no_f_inlier_num:
            is_inlier_pass = False

    if is_inlier_pass:
        print('    --- Fundamental inlier test passes')
    else:
        import sys
        print('    --- Fundamental inlier test fails')
        sys.exit(-1)

    # Check if all self indices are invalid
    for idx_data, track in enumerate(kt_f.track_list):
        self_table = track.table[idx_data, :]
        valid_num = np.where(self_table > 0)[0].shape[0]
        if valid_num != 0:
            import sys
            print('    --- Fundamental self indices test fails on {}-th track'.format(idx_data))
            sys.exit(-1)
    print('    --- Fundamental self indices test passes')

    # Check if matched pairs indices are correct
    track_num = len(kt_f.track_list)
    for ref_idx_, ref_track in enumerate(kt_f.track_list):
        ref_table = ref_track.table
        for roundabout in range(1, track_num):
            que_idx_ = (ref_idx_ + roundabout) % track_num
            que_table = kt_f.track_list[que_idx_].table

            ref_row = ref_table[que_idx_:que_idx_ + 1, :]
            que_row = que_table[ref_idx_:ref_idx_ + 1, :]

            for idx_data, idx_val_in_que in np.ndenumerate(ref_row):
                if idx_val_in_que > 0:
                    if que_row[0, idx_val_in_que] != idx_data[1]:
                        import sys
                        print('    --- Fundamental matched pairs indices test fails on {} idx_data in {}-th ref_track'
                              .format(idx_data, ref_idx_))
                        sys.exit(-1)
    print('    --- Fundamental matched pairs indices test passes')

    # Visualize the 1st tracking map
    import matplotlib.pyplot as plt
    from random import randint

    # Assign the KeyTracker to draw
    ft_draw = kt_no_f

    table01 = (ft_draw.track_list[0].table)[1:2, :]
    table02 = (ft_draw.track_list[0].table)[2:3, :]
    table12 = (ft_draw.track_list[1].table)[2:3, :]

    traj_0102 = np.concatenate((vp.view_list[0].img, vp.view_list[1].img, vp.view_list[2].img), axis=1)
    traj_012 = np.concatenate((vp.view_list[0].img, vp.view_list[1].img, vp.view_list[2].img), axis=1)

    # Assume all views are in the same size
    ori_w = vp.view_list[0].img.shape[1]
    ori_h = vp.view_list[0].img.shape[0]
    thickness = 3

    # Plot table01 and table02
    both_valid_indices = np.intersect1d(np.where(table01[0, :] > 0)[0],
                                        np.where(table02[0, :] > 0)[0])
    valid_num = both_valid_indices.shape[0]
    start_range = int(0.4 * valid_num)
    end_range = int(0.45 * valid_num)
    sub_both_valid_indices = both_valid_indices[start_range:end_range]
    radius = 10
    for valid_idx in sub_both_valid_indices:
        val01 = table01[0, valid_idx]
        val02 = table02[0, valid_idx]
        img0_pt = (int(vp.view_list[0].key_pts[valid_idx].pt[0]),
                   int(vp.view_list[0].key_pts[valid_idx].pt[1]))
        img1_pt = (int(vp.view_list[1].key_pts[val01].pt[0]) + ori_w,
                   int(vp.view_list[1].key_pts[val01].pt[1]))
        img2_pt = (int(vp.view_list[2].key_pts[val02].pt[0]) + ori_w * 2,
                   int(vp.view_list[2].key_pts[val02].pt[1]))

        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        color = (r, g, b)
        traj_0102 = cv.line(traj_0102, img0_pt, img1_pt, color, thickness)
        traj_0102 = cv.line(traj_0102, img0_pt, img2_pt, color, thickness)
        cv.circle(traj_0102, img0_pt, radius, color, -1)
        cv.circle(traj_0102, img1_pt, radius, color, -1)
        cv.circle(traj_0102, img2_pt, radius, color, -1)

    # Plot table01 and table12
    valid_indices_0 = np.where(table01[0, :] > 0)
    valid_indices_0 = valid_indices_0[0]
    valid_indices_1 = table01[0, :][valid_indices_0]
    valid_indices_2 = table12[0, :][valid_indices_1]
    valid_num = valid_indices_0.shape[0]
    start_range = int(0.3 * valid_num)
    end_range = int(0.4 * valid_num)
    for idx_ in range(start_range, end_range, 1):
        val00 = valid_indices_0[idx_]
        val01 = valid_indices_1[idx_]
        val02 = valid_indices_2[idx_]

        if val02 > 0:
            img0_pt = (int(vp.view_list[0].key_pts[val00].pt[0]),
                       int(vp.view_list[0].key_pts[val00].pt[1]))
            img1_pt = (int(vp.view_list[1].key_pts[val01].pt[0]) + ori_w,
                       int(vp.view_list[1].key_pts[val01].pt[1]))
            img2_pt = (int(vp.view_list[2].key_pts[val02].pt[0]) + ori_w * 2,
                       int(vp.view_list[2].key_pts[val02].pt[1]))

            r = randint(0, 255)
            g = randint(0, 255)
            b = randint(0, 255)
            color = (r, g, b)
            traj_012 = cv.line(traj_012, img0_pt, img1_pt, color, thickness)
            traj_012 = cv.line(traj_012, img1_pt, img2_pt, color, thickness)
            cv.circle(traj_012, img0_pt, radius, color, -1)
            cv.circle(traj_012, img1_pt, radius, color, -1)
            cv.circle(traj_012, img2_pt, radius, color, -1)

    # Resize
    www = int(traj_0102.shape[1] * 0.5)
    hhh = int(traj_0102.shape[0] * 0.5)
    dim = (www, hhh)
    traj_0102 = cv.resize(traj_0102, dim)
    traj_012 = cv.resize(traj_012, dim)

    # Generate subplot
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    # Convert BGR to RGB
    axs[0].imshow(traj_0102[:, :, ::-1])
    axs[1].imshow(traj_012[:, :, ::-1])
    axs[0].set(title='01 and 02')
    axs[1].set(title='012')
    fig.suptitle('Trajectory', fontsize=18)

    plt.show()  # disable visualization as default

    logging.debug('%s test pass', __name__)
    print('=== Complete KeyTracker Unit Test ===')
