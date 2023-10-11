import numpy as np
import math

from view_processor import ViewProcessor, IMG_EXT
from key_tracker import KeyTracker, KeyTrack
from epipolar_processor import EpipolarProcessor
from triangulation_processor import TriangulationProcessor
from campose_processor import CamposeProcessor
from utils import RansacConfig
from utils import convert_rotation_to_quaternion, convert_quaternion_to_rotation
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.transform import Rotation
from scipy.linalg import block_diag

BA_DEBUG = False

########################################################################################################################


class BaProcessor:
    def __init__(self, view_processor, key_tracker, epi_processor,
                 tri_processor, campose_processor,
                 filter_size=10, iteration=3, damping_factor=5):
        """
        Constructor
        """

        # initialize internal params
        self.curr_data_idx = 0

        # assign the arguments to params
        self.filter_size = filter_size  # the maximum number of data can be stored
        self.iteration = iteration
        self.damping_factor = damping_factor
        self.view_processor = view_processor
        self.key_tracker = key_tracker
        self.epi_processor = epi_processor
        self.tri_processor = tri_processor
        self.campose_processor = campose_processor

    #########################################################################
    def process(self, img, k):
        if self.curr_data_idx >= self.filter_size:
            print('Bundle Adjustment processor is full')
            return
        else:
            # Generate a new view from ViewProcessor
            view = self.view_processor.generate_view(img, self.curr_data_idx, k)

            # Add new view in KeyTracker
            self.key_tracker.add_new_view(view, self.view_processor.view_list)

            # Store the new view in ViewProcessor
            self.view_processor.add_view(view)

            views = self.view_processor.view_list  # for convenience
            curr_view_idx = self.curr_data_idx  # for convenience

            if self.curr_data_idx == 0:
                print('In one image state')
                self.view_processor.view_list[0].is_valid = True
            elif self.curr_data_idx == 1:
                print('In epipolar state')
                # Generate the matched pairs from KeyTracker
                matched_pairs, r_indices, q_indices = \
                    self.key_tracker.generate_matched_pairs(0, 1, self.view_processor.view_list)

                if BA_DEBUG:
                    print('DEBUG: matched pair num between view0 and view1 {}'.
                          format(matched_pairs[0].shape[1]))

                # Run EpipolarProcessor.determine_fundamental_mat() to have fundamental matrix
                self.epi_processor.determine_fundamental_mat(matched_pairs)

                # Run EpipolarProcessor.extract_essential_mat() to have essential matrix
                self.epi_processor.extract_essential_mat(views[0].k, views[1].k)

                # Run CamposeProcessor.extract_cam_pose_from_essential_mat() to have four pairs of camera pose
                r1, r2, c1, c2 = self.campose_processor.extract_cam_pose_from_essential_mat(self.epi_processor.esse_mat)

                # Run CamposeProcessor.disambiguate_cam_pose_four() to have the cam pose of view_1
                r1c1_proj = k @ np.hstack((r1.T, -r1.T @ c1))
                r1c2_proj = k @ np.hstack((r1.T, -r1.T @ c2))
                r2c1_proj = k @ np.hstack((r2.T, -r2.T @ c1))
                r2c2_proj = k @ np.hstack((r2.T, -r2.T @ c2))
                r_candidates = [r1, r1, r2, r2]
                c_candidates = [c1, c2, c1, c2]
                proj_candidates = [r1c1_proj, r1c2_proj, r2c1_proj, r2c2_proj]

                tri_pts_candidates = []
                ref_proj = views[0].cam_proj
                for idx in range(0, 4):
                    projs = [ref_proj, proj_candidates[idx]]
                    linear_3d_pts = self.tri_processor.linear_triangulate(projs, matched_pairs)
                    tri_pts_candidates.append(linear_3d_pts)

                best_idx, valid_indices = self.campose_processor.disambiguate_cam_pose_four(ref_proj,
                                                                                            proj_candidates,
                                                                                            tri_pts_candidates)
                views[1].update_cam_pose(r_candidates[best_idx], c_candidates[best_idx])

                if BA_DEBUG:
                    euler_angle = Rotation.from_matrix(r_candidates[best_idx]).as_euler('zyx', degrees=True)
                    print('DEBUG: view1 euler angles in view0 coordinate\n {}'.format(euler_angle))
                    print('DEBUG: view1 camera pos in view0 coordinate\n {}'.format(c_candidates[best_idx].T))

                # Run TriangulationProcessor.nonlinear_triangulate() to refined triangulated 3D pts
                valid_indices = np.array(valid_indices)  # Make valid indices as np.array
                valid_indices = valid_indices[np.newaxis, :]
                projs = [ref_proj, proj_candidates[best_idx]]
                valid_matched_pairs = [np.take_along_axis(matched_pairs[0], valid_indices, axis=1),
                                       np.take_along_axis(matched_pairs[1], valid_indices, axis=1)]
                valid_tri_pts = np.take_along_axis(tri_pts_candidates[best_idx], valid_indices, axis=1)
                refined_3d_pts = self.tri_processor.nonlinear_triangulate(valid_tri_pts,
                                                                          projs,
                                                                          valid_matched_pairs)

                # Update view0 and view1 key point usage for triangulation
                valid_r_indices = np.take(r_indices, valid_indices)
                valid_q_indices = np.take(q_indices, valid_indices)
                tri_indices = np.arange(0, refined_3d_pts.shape[1], dtype=int)
                tri_indices = tri_indices[np.newaxis, :]
                self.key_tracker.track_list[0].update_usage(valid_r_indices, tri_indices)
                self.key_tracker.track_list[1].update_usage(valid_q_indices, tri_indices)

                # Update the added view validation
                views[1].is_valid = True
                views[1].ref_idx = 0

                # Store the refined triangulated points
                self.tri_processor.add_tri_pt(refined_3d_pts)

                if BA_DEBUG:
                    print('DEBUG: Add {} tri points after {}-th view'.format(refined_3d_pts.shape[1], 1))

            else:
                print('In cam pose state')
                # Steps
                curr_view_idx = self.curr_data_idx
                curr_view = self.view_processor.view_list[curr_view_idx]

                # KeyTracker::find_best_view() - find the best view (the reference) for the new view
                best_view_idx = self.key_tracker.find_best_view(curr_view_idx)
                best_view = self.view_processor.view_list[best_view_idx]
                curr_view.ref_idx = best_view_idx

                # Extract the matched pairs,
                # indices of key points matching in best and current views from KeyTracker.
                matched_pairs, b_key_idx, c_key_idx = \
                    self.key_tracker.generate_matched_pairs(best_view_idx, curr_view_idx,
                                                            self.view_processor.view_list)

                if BA_DEBUG:
                    print('DEBUG: matched pair num between view{} and view{} {}'.
                          format(best_view_idx, curr_view_idx, matched_pairs[0].shape[1]))

                # Extract the indices of key points already used for triangulation in the best view.
                # b_tri_idx represents the index list of key pt.
                # b_tri_val represents the index list of tri pt.
                b_tri_idx, b_tri_val = \
                    self.key_tracker.track_list[best_view_idx].extract_constructed_points()

                # In order to estimate the cam pose of the current view
                # the triangulated points which are projected to the current view are required
                # Therefore, the indices of those triangulated points are the intersection of
                # b_key_idx and b_tri_idx
                # b_key_intersect_idx represents the indices in b_key_idx, not value in b_key_idx
                # b_tri_intersect_idx represents the indices in b_tri_idx, not value in b_tri_idx.
                # TODO:
                # The approach here ONLY works with best_view_idx == 0 for all other views!!!
                if b_tri_idx.shape[1] != self.tri_processor.tri_pts.shape[1]:
                    import sys
                    sys.exit('ERROR: even best_view_idx assumption does not work !!!')

                used_inters_val, b_key_intersect_idx, b_tri_intersect_idx = \
                    np.intersect1d(b_key_idx, b_tri_idx, return_indices=True)
                used_inters_val = used_inters_val[np.newaxis, :]
                b_key_intersect_idx = b_key_intersect_idx[np.newaxis, :]
                b_tri_intersect_idx = b_tri_intersect_idx[np.newaxis, :]

                # Use val which is the index of the tri point.
                b_tri_intersect_val = np.take_along_axis(b_tri_val, b_tri_intersect_idx, axis=1)
                tri_pts = np.take_along_axis(self.tri_processor.tri_pts, b_tri_intersect_val, axis=1)

                # Extract the key points in the current view
                c_key_intersect_idx = b_key_intersect_idx
                c_key_pts = np.take_along_axis(matched_pairs[1], c_key_intersect_idx, axis=1)

                # CamPoseProcessor::estimate_cam_pose_pnp() - estimate new view camera pose
                inlier_ind, c_rot, c_loc = \
                    self.campose_processor.estimate_cam_pose_pnp(c_key_pts, tri_pts, curr_view.k)

                if BA_DEBUG:
                    euler_angle = Rotation.from_matrix(c_rot).as_euler('zyx', degrees=True)
                    print('DEBUG: view{} euler angles in view{} coordinate\n {}'.
                          format(curr_view_idx, best_view_idx, euler_angle))
                    print('DEBUG: view{} camera pos in view{} coordinate\n {}'.
                          format(curr_view_idx, best_view_idx, c_loc.T))

                    import cv2
                    pnp_3d_pts_t = tri_pts[0:3, :].T
                    pnp_2d_pts_t = c_key_pts[0:2, :].T
                    _, Rt_cv, t_cv, _ = cv2.solvePnPRansac(pnp_3d_pts_t[:, np.newaxis], pnp_2d_pts_t[:, np.newaxis],
                                                           curr_view.k,
                                                           None,
                                                           confidence=0.99, reprojectionError=20.0,
                                                           flags=cv2.SOLVEPNP_ITERATIVE)
                    Rt_cv, _ = cv2.Rodrigues(Rt_cv)
                    euler_angle = Rotation.from_matrix(Rt_cv.T).as_euler('zyx', degrees=True)
                    print('DEBUG: OpenCV view{} euler angles in view{} coordinate\n {}'.
                          format(curr_view_idx, best_view_idx, euler_angle))
                    print('DEBUG: OpenCV view{} camera pos in view{} coordinate\n {}'.
                          format(curr_view_idx, best_view_idx, (-Rt_cv.T @ t_cv).T))

                # Update the new view's cam pose
                curr_view.update_cam_pose(c_rot, c_loc)
                curr_view.is_valid = True

                # Extract the indices of key points in the best view
                # are not used for triangulation yet.
                unused_tri_idx = self.key_tracker.track_list[best_view_idx].extract_unconstructed_points()

                # TriangulationProcessor::triangulate() - calculate the triangulated points
                # Prepare projections
                projs = [best_view.cam_proj, curr_view.cam_proj]

                # Prepare the matched pairs which are not used for triangulation yet
                unused_inters_val, b_key_intersect_idx, unused_tri_intersect_idx = \
                    np.intersect1d(b_key_idx, unused_tri_idx,return_indices=True)
                unused_inters_val = unused_inters_val[np.newaxis, :]

                intersect_val, _, _ = \
                    np.intersect1d(unused_inters_val, used_inters_val, return_indices=True)
                if np.any(intersect_val) != False:
                    import sys
                    sys.exit('ERROR: the intersect of unused and used inters sets is NOT empty')

                b_key_intersect_idx = b_key_intersect_idx[np.newaxis, :]
                c_key_intersect_idx = b_key_intersect_idx

                # Triangulate the 3D points
                b_key_pts = np.take_along_axis(matched_pairs[0], b_key_intersect_idx, axis=1)
                c_key_pts = np.take_along_axis(matched_pairs[1], c_key_intersect_idx, axis=1)
                matched_pairs = [b_key_pts, c_key_pts]
                new_tri_pts = self.tri_processor.triangulate(projs, matched_pairs)

                # Update the key point triangulation usage in best view.
                tri_indices = np.arange(self.tri_processor.tri_pts.shape[1],
                                  self.tri_processor.tri_pts.shape[1] + new_tri_pts.shape[1], dtype=int)
                tri_indices = tri_indices[np.newaxis, :]
                best_view_track_table = self.key_tracker.track_list[best_view_idx]
                best_view_track_table.update_usage(unused_inters_val, tri_indices)

                # Update the key point triangulation usage in current view.
                indices = np.take(best_view_track_table.table[curr_view_idx, :], unused_inters_val)
                curr_view_track_table = self.key_tracker.track_list[curr_view_idx]
                curr_view_track_table.update_usage(indices, tri_indices)

                # TriangulationProcessor::add_tri_pt() - add the new triangulated points
                self.tri_processor.add_tri_pt(new_tri_pts)

                if BA_DEBUG:
                    print('Add {} tri points in {}-th view'.format(new_tri_pts.shape[1], curr_view_idx))

                # BaProcessor::__execute_bundle_adjustment()
                self.__execute_bundle_adjustment()

            # increase the index
            self.curr_data_idx += 1

    #########################################################################

    def __execute_bundle_adjustment(self):
        view_num = len(self.view_processor.view_list)
        tri_num = self.tri_processor.tri_pts.shape[1]
        views = self.view_processor.view_list
        tri_pts = self.tri_processor.tri_pts

        if BA_DEBUG:
            print('DEBUG: There are {} tri pts AND {} views for bundle adjustment'.
                  format(tri_num, view_num))

        # Retrieve the initial cam pose from all views
        init_cam_poses = np.zeros((7 * view_num, 1))
        for view_idx in range(0, view_num):
            qua = convert_rotation_to_quaternion(views[view_idx].rot)
            init_cam_poses[7 * view_idx: 7 * (view_idx + 1)] = np.vstack((views[view_idx].loc, qua))
        refined_cam_poses = init_cam_poses.copy()

        # Retrieve the initial tri points
        init_tri_pts = tri_pts[0:3, :].T
        init_tri_pts = init_tri_pts.reshape(3 * tri_num, 1)
        refined_tri_pts = init_tri_pts.copy()

        # Run the bundle adjustment algorithm.
        for iter in range(0, self.iteration):
            j_p = []
            j_x = []
            b = []
            f = []
            d_inv = None
            visibility_count = 0
            for tri_idx in range(0, tri_num):
                d = np.zeros((3, 3))
                for view_idx in range(0, view_num):
                    key_idx = -1
                    view = views[view_idx]
                    key_idx = self.key_tracker.is_visible(view_idx, tri_idx)
                    if key_idx != -1:
                        visibility_count += 1

                        # if BA_DEBUG:
                        #     print('DEBUG: {}-th tri pt is visible in {}-th view'.format(tri_idx, view_idx))

                        # Retrieve the corresponding triangulated and key points
                        tri_pt = np.append(refined_tri_pts[3 * tri_idx:3 * (tri_idx+1)], [[1.0]], axis=0) # tri_pts[:, tri_idx:tri_idx + 1]
                        key_pt = view.key_pts[key_idx]

                        # Retrieve view's camera pose and projection (with intrinsic matrix).
                        loc = refined_cam_poses[(7 * view_idx):(7 * view_idx + 3)]  # views[view_idx].loc
                        qua = refined_cam_poses[(7 * view_idx + 3):(7 * view_idx + 7)]  # views[view_idx].rot
                        rot = convert_quaternion_to_rotation(qua)

                        # Because error is calculated in camera coordinate, not image coordinate,
                        # the intrinsic matrix is not involved with the projection matrix.
                        # It is different from calling TriangulationProcessor::triangulate().
                        proj = np.hstack((rot.T, rot.T @ -loc))

                        j_pp = np.zeros((2, 7*view_num))
                        j_xx = np.zeros((2, 3*tri_num))
                        jac_p = self.campose_processor.construct_jacobian_matrix(rot, loc, tri_pt)
                        jac_x = self.tri_processor.construct_jacobian_matrix(tri_pt, [proj], 1)
                        j_pp[:, 7 * view_idx:7 * (view_idx + 1)] = jac_p
                        j_xx[:, 3 * tri_idx:3 * (tri_idx + 1)] = jac_x

                        # Convert the key point from image coordinate to camera coordinate,
                        # matching campose_processor.construct_jacobian_matrix().
                        key_pt = np.array([[key_pt.pt[0], key_pt.pt[1], 1.0]]).T
                        key_pt = np.linalg.inv(view.k) @ key_pt
                        key_pt /= key_pt[2]  # normalization
                        key_pt = key_pt[0:2, 0]

                        # Generate the projected point in camera coordinate from triangulated point.
                        # (intrinsic matrix not involved)
                        proj = np.hstack((rot.T, rot.T @ -loc))
                        proj_pt = proj @ tri_pt
                        proj_pt /= proj_pt[2]  # normalization
                        proj_pt = proj_pt[0:2, 0]

                        # Append.
                        # Using list::append is faster than numpy.vstack
                        j_p.append(j_pp.tolist())
                        j_x.append(j_xx.tolist())
                        d += (jac_x.T @ jac_x)
                        b.append(key_pt.tolist())
                        f.append(proj_pt.tolist())

                d = d + self.damping_factor * np.eye(3)
                if d_inv is None:
                    d_inv = np.linalg.inv(d)
                else:
                    d_inv = block_diag(d_inv, np.linalg.inv(d))

            # Convert from list to numpy.array
            j_p = np.array(j_p)
            j_p = j_p.reshape(2 * visibility_count, 7 * view_num)
            j_x = np.array(j_x)
            j_x = j_x.reshape(2 * visibility_count, 3 * tri_num)
            b = np.array(b)
            b = b.reshape(2 * visibility_count, 1)
            f = np.array(f)
            f = f.reshape(2 * visibility_count, 1)

            # Calculate delta.
            ep = j_p.T @ (b-f)
            ex = j_x.T @ (b-f)
            A = j_p.T @ j_p + self.damping_factor * np.eye((j_p.T @ j_p).shape[0])
            B = j_p.T @ j_x

            # Update cam pose with delta.
            delta_p = np.linalg.inv(A - B @ d_inv @ B.T) @ (ep - B @ d_inv @ ex)
            refined_cam_poses += delta_p

            # Normalize each quaterion in refined_cam_poses.
            # Normalization guarantees convert_quaternion_to_rotation generate
            # a valid rotation matrix.
            for view_idx in range(0, view_num):
                qua = refined_cam_poses[(7 * view_idx + 3):(7 * view_idx + 7)]
                norm = math.sqrt(np.sum(np.square(qua)))
                qua /= norm
                refined_cam_poses[(7 * view_idx + 3):(7 * view_idx + 7)] = qua

                # if BA_DEBUG:
                #     print('{}-iteration {}-th view loc \n{}'.
                #           format(iter, view_idx,
                #                  refined_cam_poses[(7 * view_idx + 0):(7 * view_idx + 3)].T))
                #     qua = refined_cam_poses[(7 * view_idx + 3):(7 * view_idx + 7)]
                #     rot = convert_quaternion_to_rotation(qua)
                #     euler_angle = Rotation.from_matrix(rot).as_euler('zyx', degrees=True)
                #     print('{}-iteration {}-th view euler angle \n{}'.
                #           format(iter, view_idx, euler_angle))

            # Update tri points with delta.
            delta_x = d_inv @ (ex - B.T @ delta_p)
            refined_tri_pts += delta_x

        # Update the cam poses and tri points.
        for view_idx in range(0, view_num):
            loc = refined_cam_poses[(7 * view_idx):(7 * view_idx + 3)]
            qua = refined_cam_poses[(7 * view_idx + 3):(7 * view_idx + 7)]
            rot = convert_quaternion_to_rotation(qua)
            views[view_idx].update_cam_pose(rot, loc)

        for tri_idx in range(0, tri_num):
            tri_pts[0:3, tri_idx] = refined_tri_pts[3 * tri_idx:3 * (tri_idx + 1), 0]

        if True:
            for view_idx in range(0, view_num):
                init_loc = init_cam_poses[(7 * view_idx):(7 * view_idx + 3)]
                refi_loc = refined_cam_poses[(7 * view_idx):(7 * view_idx + 3)]
                diff_loc = norm = math.sqrt(np.sum(np.square(init_loc - refi_loc)))
                print('DEBUG: {}-th view loc distance changes {} unit'.format(view_idx, diff_loc))

                init_qua = init_cam_poses[(7 * view_idx + 3):(7 * view_idx + 7)]
                refi_qua = refined_cam_poses[(7 * view_idx + 3):(7 * view_idx + 7)]
                init_rot = convert_quaternion_to_rotation(init_qua)
                refi_rot = convert_quaternion_to_rotation(refi_qua)
                init_angle = Rotation.from_matrix(init_rot).as_euler('zyx', degrees=True)
                refi_angle = Rotation.from_matrix(refi_rot).as_euler('zyx', degrees=True)
                diff_angle = np.abs(init_angle - refi_angle)
                print('DEBUG: {}-th view angles changes {} degree'.format(view_idx, diff_angle))

            for tri_idx in range(0, tri_num):
                init_pt = init_tri_pts[(3 * tri_idx):(3 * tri_idx + 3)]
                refi_pt = refined_tri_pts[(3 * tri_idx):(3 * tri_idx + 3)]
                diff_pt = norm = math.sqrt(np.sum(np.square(init_pt - refi_pt)))
                if diff_pt >= 5:
                    print('DEBUG: {}-th pt loc changes more than 5 unit, {} unit'.format(tri_idx, diff_pt))

########################################################################################################################

if __name__ == '__main__':
    """
    Run the Bundle Adjustment Processor unit test
    """
    
    print('=== Start BaProcessor Unit Test ===')

    # Set up the environment
    import os
    import cv2 as cv
    cur_path = os.path.dirname(__file__)
    test_dataset_path = os.path.join(cur_path, 'test_dataset', 'upenn')

    # Set up the params
    K = np.array([[568.996140852, 0, 643.21055941],
                  [0, 568.988362396, 477.982801038],
                  [0, 0, 1]])
    REF_ROT = np.identity(3)
    REF_LOC = np.array([[0.0], [0.0], [0.0]])
    REF_POSE = np.hstack((REF_ROT, REF_LOC))
    ransac_kt = RansacConfig(inlier_threshold=1e-2,
                              subset_confidence=0.99,
                              # the desired probability that the result from this model is 0.99 "reliable"
                              sample_confidence=0.75,  # (inlier data / total data)
                              sample_num=8,
                              iteration=200)
    ransac_ep = RansacConfig(inlier_threshold=1e-3,
                              subset_confidence=0.99,
                              # the desired probability that the result from this model is 0.99 "reliable"
                              sample_confidence=0.75,  # (inlier data / total data)
                              sample_num=8,
                              iteration=300)
    ransac_cp = RansacConfig(inlier_threshold=8.0,
                             subset_confidence=0.99,
                             # the desired probability that the result from this model is 0.99 "reliable"
                             sample_confidence=0.75,  # (inlier data / total data)
                             sample_num=8,
                             iteration=300)

    view_processor = ViewProcessor('sift')
    key_tracker = KeyTracker('sift', False, True, False, ransac_kt)
    epi_processor = EpipolarProcessor(ransac_ep)
    tri_processor = TriangulationProcessor()
    campose_processor = CamposeProcessor(ransac_cp, 5, 300)
    bp_processor = BaProcessor(view_processor, key_tracker, epi_processor,
                               tri_processor, campose_processor)

    # Set up the dataset
    files_ = ["image0000001.bmp", "image0000002.bmp",
              "image0000003.bmp", "image0000004.bmp",
              "image0000005.bmp", "image0000006.bmp"]
    # files_ = ["image0000001.bmp", "image0000002.bmp",
    #           "image0000003.bmp"]
    count = 0

    # Run SFM pipeline
    for file_name in files_:
        img_path = os.path.join(test_dataset_path, file_name)
        print('Processing {} image'.format(file_name))
        img = cv.imread(img_path)
        bp_processor.process(img, K)
        count += 1

    # Plot the result
    color_map = ['r', 'g', 'b', 'y', 'c', 'm']
    fig = plt.figure()
    i = 0
    print('Visualizing {} views'.format(len(bp_processor.view_processor.view_list)))
    loc_all = np.zeros((len(bp_processor.view_processor.view_list), 3))
    rot_all = np.zeros((len(bp_processor.view_processor.view_list), 3, 3))
    for view_ in bp_processor.view_processor.view_list:
        print('{}-th view loc is \n {}'.format(i, view_.loc.T))
        print('{}-th view ori is \n {}'.format(i, Rotation.from_matrix(view_.rot).as_euler('zyx', degrees=True)))

        cam_pose = view_.cam_pose
        C = view_.loc
        R = view_.rot
        R = Rotation.from_matrix(R).as_rotvec()
        R1 = np.rad2deg(R)
        t = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
        t._transform = t.get_transform().rotate_deg(int(R1[1]))
        ax = plt.gca()
        dot = mpl.markers.MarkerStyle(marker=".")
        ax.scatter((C[0]), (C[2]), marker=t, s=250, color=color_map[i])
        ax.scatter((C[0]), (C[2]), marker=dot, s=250, color='black')

        loc_all[i, :] = view_.loc.T
        rot_all[i, :, :] = view_.rot
        i += 1

    tri_pt_num = bp_processor.tri_processor.tri_pts.shape[1]
    print('Visualizing {} triangulated points'.format(tri_pt_num))

    for idx in range(0, tri_pt_num):
        X = bp_processor.tri_processor.tri_pts[:, idx:idx+1]
        ax.scatter(X[0, 0], X[2, 0], s=4, color='darkseagreen', label='tri_pt')

    plt.xlim(-20, 20)
    plt.ylim(-20, 30)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    plt.show()

    print('=== Complete BaProcessor Unit Test ===')
