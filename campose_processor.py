import numpy as np
import logging
import random
import math

from utils import RansacConfig, KeyPt, TriPt
from utils import convert_rotation_to_quaternion, convert_quaternion_to_rotation


########################################################################################################################
class CamposeProcessor:
    def __init__(self, ransac_config, damping_factor, iteration):
        """
        Constructor

        @Param ref_cam_pose
        The reference camera pose.
        numpy.array((3, 4))

        @Param intrinsic_mat
        The intrinsic matrix
        numpy.array((3, 3))
        """
        self.ransac_config = ransac_config
        self.damping_factor = damping_factor
        self.iteration = iteration

    #########################################################################
    def extract_cam_pose_from_essential_mat(self, esse_mat):
        """
        Extract four possible combination of camera pose (R, C) from the essential matrix

        @Param esse_mat
        The essential matrix
        numpy.array((3, 3))

        @Return r1
        One possible camera rotation
        from the reference to destination coordinate
        in the reference coordinate.
        numpy.array((3, 3))

        @Return r2
        One possible camera rotation
        from the reference to destination coordinate
        in the reference coordinate.
        numpy.array((3, 3))

        @Return c1
        One possible camera location
        in the reference coordinate.
        numpy.array((3, 1))

        @Return c2
        One possible camera location
        in the reference coordinate.
        numpy.array((3, 1))
        """

        # TODO
        # why z is not used???
        z = np.array([[0, 1, 0],
                      [-1, 0, 0],
                      [0, 0, 0]])
        w = np.array([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]])

        u, s, vh = np.linalg.svd(esse_mat)

        # last column in u, of which the corresponding eigenvalue is 0.
        # (left null space)
        c1 = u[:, 2].reshape(-1, 1)
        c2 = -c1

        r1 = u @ w @ vh
        r2 = u @ w.T @ vh

        # correct the rotation matrix of which determinant is one
        if np.linalg.det(r1) < 0:
            r1 = -r1
        if np.linalg.det(r2) < 0:
            r2 = -r2

        # 4: four possible combinations
        # 3 x 4: projection matrix dimension
        # cam_pos_four = np.zeros((4, 3, 4))
        # cam_pos_four[0][0:3, 0:3] = r1
        # cam_pos_four[0][:, 3:4] = c1
        # cam_pos_four[1][0:3, 0:3] = r1
        # cam_pos_four[1][:, 3:4] = c2
        # cam_pos_four[2][0:3, 0:3] = r2
        # cam_pos_four[2][:, 3:4] = c1
        # cam_pos_four[3][0:3, 0:3] = r2
        # cam_pos_four[3][:, 3:4] = c2

        r1 = r1.transpose()
        r2 = r2.transpose()
        return r1, r2, c1, c2

    #########################################################################
    def disambiguate_cam_pose_four(self, ref_proj, projs_four, tri_3d_pts_four):
        """
        Determine the best camera pose in the four possible combination.

        @Param projs_four
        A list of four camera projection matrices, len(projs_four) == 4

        @Param tri_3d_pts_four
        A list of TriPts, len(tri_3d_pts_four) == 4

        @Return idx
        The index for best camera pose and the corresponding triangulated 3D
        point set

        @Return best_idx
        The index of the best camera pose

        @Return most_valid_indices
        a numpy.array
        """

        most_valid_count = 0
        best_idx = 0
        most_valid_indices = []
        for idx in range(4):
            valid_indices = self.evalulate_cam_pose_cheirality(ref_proj,
                                                               projs_four[idx],
                                                               tri_3d_pts_four[idx])
            valid_count = len(valid_indices)
            if valid_count > most_valid_count:
                most_valid_count = valid_count
                best_idx = idx
                most_valid_indices = valid_indices

        return best_idx, most_valid_indices

    #########################################################################
    def evalulate_cam_pose_cheirality(self, proj_1, proj_2, tri_3d_pts):
        """
        Determine if any 3D point is in front of self.ref_cam_pose and cam_pose

        @Param cam_pose
        The camera pose, [R | C]
        numpy.array((3, 4))

        @Param tri_3d_pts
        The triangulated 3D points in the homogeneous form
        TriPt((4, num of points))

        @Return indices
        The list of the valid triangulated 3D point indices

        #Reference:
        http://users.cecs.anu.edu.au/~hartley/Papers/cheiral/revision/cheiral.pdf
        """

        num_pts = tri_3d_pts.shape[1]
        indices = []

        # r1 = self.ref_cam_pose[:, 0:3]
        # r1t = r1.T
        # c1 = r1 = self.ref_cam_pose[:, 3:4]
        # p1 = np.hstack((r1t, -r1t @ c1))
        p1 = proj_1

        # r2 = cam_pose[:, 0:3]
        # r2t = r2.T
        # c2 = cam_pose[:, 3:4]
        # p2 = np.hstack((r2t, -r2t @ c2))
        p2 = proj_2

        # TODO: replace loop with the following implementation.
        # prj1_res = p1 @ tri_3d_pts
        # prj2_res = p2 @ tri_3d_pts
        # prj1_res_last_row = prj1_res[-1, :]
        # prj2_res_last_row = prj2_res[-1, :]
        # proj_res = np.multiply(prj1_res[-1, :], prj2_res[-1, :])
        # indices = np.argwhere(proj_res > 0)
        # return indices
        for idx in range(num_pts):
            prj1 = p1 @ tri_3d_pts[:, idx:idx+1]
            prj2 = p2 @ tri_3d_pts[:, idx:idx+1]

            # Check if the projected point is in front of two cameras
            if (prj1[2] > 0) and (prj2[2] > 0):
                indices.append(idx)

        return indices

    #########################################################################
    def estimate_cam_pose_pnp(self, key_2d_pts, tri_3d_pts, intrinsic_mat,
                              ransac_config=None, damping_factor=None, iteration=None):
        """
        Linearly estimate camera pose, (R, C), with RANASC

        @Param key_2d_pts
        2D key points in the image coordinate (homogeneous coordinate)
        KeyPt((3, num of points))

        @Param tri_3d_pts
        3D triangulated points in the world coordinate (homogeneous coordinate)
        TriPt((4, num of points))

        @Param ransac_config
        RANSAC configuration

        @Param intrinsic_mat
        Intrinsic matrix
        nummpy.array((3, 3))

        @Return inlier_indices
        A list of inliers' indices

        @Return rot
        Rotation matrix
        numpy.array((3, 3))

        @Return loc
        Camera position
        numpy.array((3, 1))
        """

        if not ransac_config:
            ransac_config = self.ransac_config
        if not damping_factor:
            damping_factor = self.damping_factor
        if not iteration:
            iteration = self.iteration

        inlier_indices, ini_rot, ini_loc = self.linear_estimate_cam_pose_pnp(key_2d_pts,
                                                                              tri_3d_pts,
                                                                              intrinsic_mat,
                                                                              ransac_config)

        inlier_key_2d_pts = key_2d_pts[:, inlier_indices]
        inlier_tri_3d_pts = tri_3d_pts[:, inlier_indices]

        ref_rot, ref_loc = self.nonlinear_estimate_cam_pose_pnp(inlier_key_2d_pts,
                                                                inlier_tri_3d_pts,
                                                                intrinsic_mat,
                                                                ini_rot, ini_loc,
                                                                damping_factor,
                                                                iteration)

        return inlier_indices, ref_rot, ref_loc

    #########################################################################
    def linear_estimate_cam_pose_pnp(self, key_2d_pts, tri_3d_pts, intrinsic_mat,
                                     ransac_config=None):
        """
        Linearly estimate camera pose, (R, C), with RANASC

        @Param key_2d_pts
        2D key points in the image coordinate
        numpy.array((3, num of points))

        @Param tri_3d_pts
        3D triangulated points in the world coordinate
        numpy.array((4, num of points))

        @Param ransac_config
        RANSAC configuration

        @Param intrinsic_mat
        Intrinsic matrix
        nummpy.array((3, 3))

        @Return inlier_indices
        A list of inliers' indices

        @Return rot
        Rotation matrix
        from the reference to destination coordinate
        in the reference coordinate.
        numpy.array((3, 3))

        @Return loc
        Camera position
        in the reference coordinate.
        numpy.array((3, 1))
        """

        if not ransac_config:
            ransac_config = self.ransac_config

        if key_2d_pts.shape[1] != tri_3d_pts.shape[1]:
            logging.warning('%s : different numbers of key points and of triangulated points',
                            self.__class__.__name__, key_2d_pts.shape[1], tri_3d_pts.shape[1])
            raise ValueError("key pts num - triangulated pts num : {} - {}"
                             .format(key_2d_pts.shape[1], tri_3d_pts.shape[1]))

        num_pts = key_2d_pts.shape[1]
        if num_pts < 6:
            logging.warning('%s : required equal or more than six points {}',
                            self.__class__.__name__, num_pts)
            raise ValueError("required equal or more than six points {}"
                             .format(num_pts))

        # transform from image coordinate to camera coordinate
        rot, loc, inlier_num, inlier_indices = self.__linear_determine_cam_pos(key_2d_pts, tri_3d_pts,
                                                                                 intrinsic_mat,
                                                                                 ransac_config)

        return inlier_indices, rot, loc

    #########################################################################
    def nonlinear_estimate_cam_pose_pnp(self, key_2d_pts, tri_3d_pts, intrinsic_mat,
                                        init_rot, init_loc, damping_factor=None, iteration=None):
        """
        Nonlinearly estimate camera pose, (R, C), through Levenberg-Marquardt algorithm (LMA)

        @Param key_2d_pts
        2D key points in the image coordinate
        numpy.array((3, num of points))

        @Param tri_3d_pts
        3D triangulated points in the world coordinate
        numpy.array((4, num of points))

        @Param init_rot
        Initial camera rotation
        numpy.array((3, 3))

        @Param init_loc
        Initial translation vector
        numpy.array((3, 1))

        @Param intrinsic_mat
        Intrinsic matrix
        nummpy.array((3, 3))

        @Param damping_factor
        The damping factor used in the Levenberg-Marquardt algorithm (LMA)

        @Param iteration
        The iteration times used in the Levenberg_marquardt algorithm (LMA)

        @Return refined_rot
        Refined camera rotation
        numpy.array((3, 3))

        @Return refined_loc
        Refined camera location
        numpy.array((3, 1))
        """

        if not damping_factor:
            damping_factor = self.damping_factor
        if not iteration:
            iteration = self.iteration

        if key_2d_pts.shape[1] != tri_3d_pts.shape[1]:
            logging.warning('%s : different numbers of key points and of triangulated points',
                            self.__class__.__name__, key_2d_pts.shape[1], tri_3d_pts.shape[1])
            raise ValueError("key pts num - triangulated pts num : {} - {}"
                             .format(key_2d_pts.shape[1], tri_3d_pts.shape[1]))

        num_pts = key_2d_pts.shape[1]

        init_qua = convert_rotation_to_quaternion(init_rot)
        norm = math.sqrt(np.sum(np.square(init_qua)))
        init_qua /= norm  # normalize quaternion

        refined_loc = np.copy(init_loc)
        refined_qua = np.copy(init_qua)
        refined_rot = np.copy(init_rot)

        # the parameters to refine are
        # [loc_x, loc_y, loc_z, qua_w, qua_x, qua_y, qua_z]
        params = np.vstack((init_loc, init_qua))

        # for debugging
        # loc_diff = np.zeros((3, iteration))
        # rot_diff = np.zeros((3, iteration))
        # for debugging

        for it in range(iteration):

            jac_rp_all = np.zeros((2 * num_pts, 7))
            err_all = np.zeros((2 * num_pts, 1))
            for pt_idx in range(num_pts):  # Loop through all points
                pt_3d = tri_3d_pts[:, pt_idx:pt_idx + 1]
                pt_2d = key_2d_pts[:, pt_idx:pt_idx + 1]

                # project 3D pt to 2D pt in camera coordinate, NOT image coordinate
                # (no intrinsic matrix involved)
                proj = np.hstack((refined_rot.T, refined_rot.T @ -refined_loc))
                proj_2d = proj @ pt_3d
                proj_2d_norm = proj_2d / proj_2d[2]  # normalization

                # convert 2d pt in image coordinate to camera coordinate
                pt_2d_cam = np.linalg.inv(intrinsic_mat) @ pt_2d
                pt_2d_cam_norm = pt_2d_cam / pt_2d_cam[2]  # normalization

                # construct jacobian matrix w.r.t (loc, qua)
                jac_rp = self.construct_jacobian_matrix(refined_rot, refined_loc, pt_3d)

                # calculate error (difference)
                err = (pt_2d_cam_norm - proj_2d_norm)
                err = err[0:2, 0:1]

                # accumulate the Jacobian matrix and error for each point
                jac_rp_all[pt_idx:pt_idx + 2, :] = jac_rp
                err_all[pt_idx:pt_idx + 2, :] = err

            # based on the Levenberg-Marquardt algorithm,
            # calculate the delta
            delta = np.linalg.inv(jac_rp_all.T.dot(jac_rp_all) + damping_factor * np.identity(7)).dot(jac_rp_all.T).dot(
                err_all)

            # update the params
            params += delta
            refined_loc = params[0:3, 0:1]
            refined_qua = params[3:7, 0:1]

            # normalize the quaternion
            norm = math.sqrt(np.sum(np.square(refined_qua)))
            refined_qua /= norm

            # calculate the rotation matrix from quaternion
            refined_rot = convert_quaternion_to_rotation(refined_qua)

            # for debugging
            # from scipy.spatial.transform import Rotation as R
            # print('          Init loc : {}'.format(init_loc.T))
            # print('Iteration {}-th loc diff : {}'.format(it, refined_loc.T - init_loc.T))

            # euler_angle_init = R.from_matrix(init_rot).as_euler('zyx', degrees=True)
            # print('          Init rot : {}'.format(euler_angle_init))
            # euler_angle = R.from_matrix(refined_rot).as_euler('zyx', degrees=True)
            # print('Iteration {}-th rot diff : {}'.format(it, euler_angle - euler_angle_init))
            # print('======================================')

            # loc_diff[:, it] = np.absolute(refined_loc.T - init_loc.T)
            # rot_diff[:, it] = np.absolute(euler_angle - euler_angle_init)
            # for debugging

        # for debugging
        # import matplotlib.pyplot as plt
        # x_values = np.arange(0, iteration, 1, dtype=int)

        # plt.plot(x_values, loc_diff[0], label='x')
        # plt.plot(x_values, loc_diff[1], label='y')
        # plt.plot(x_values, loc_diff[2], label='z')
        # plt.legend()
        # plt.title('Loc Diff')
        # plt.show()

        # plt.plot(x_values, rot_diff[0], label='yaw')
        # plt.plot(x_values, rot_diff[1], label='roll')
        # plt.plot(x_values, rot_diff[2], label='pitch')
        # plt.legend()
        # plt.title('Rot Diff')
        # plt.show()
        # for debugging

        refined_rot = convert_quaternion_to_rotation(refined_qua)
        return refined_rot, refined_loc

    #########################################################################
    def construct_jacobian_matrix(self, rot, loc, pt_3d):

        qua = convert_rotation_to_quaternion(rot)

        # calculate the Jacobian matrix of rotation matrix w.r.t quaterion
        jac_rq = self.__construct_jacobian_quaternion(qua)

        # calculate the Jacobian matrix of projection matrix (no intrinsic mat) w.r.t quaterion
        jac_pr = self.__construct_jacobian_rotation(rot, loc, pt_3d)

        # apply the chain rule
        jac_pq = jac_pr @ jac_rq

        # calculate the Jacobian matrix of projection matrix (no intrinsic mat) w.r.t location
        jac_pc = self.__construct_jacobian_location(rot, loc, pt_3d)

        # construct the Jacobian matrix of projection matrix (no intrinsic mat)
        # w.r.t location and quaterion
        jac_rp = np.hstack((jac_pc, jac_pq))

        return jac_rp

    #########################################################################
    def __linear_determine_cam_pos(self, key_2d_pts, tri_3d_pts,
                                   intrinsic_mat,
                                   ransac_config):
        """
        Apply direct linear transformation (DLT) with
        random sample consensus (RANSAC) to determine
        the camera pose.

        @Param key_2d_pts
        2D key points
        numpy.array((3, num of points))

        @Param tri_3d_pts
        3D triangulated points
        numpy.array((4, num of points))

        @Param ransac_config
        RANSAC setting

        @Param intrinsic_mat
        Intrinsic matrix

        @Return best_rot
        The estimated camera rotation
        numpy.array((3, 3))

        @Return best_loc
        The estimated camera position
        numpy.array((3, 1))

        @Return max_inlier_num
        The number of inliers

        @Return best_inlier_indices
        The indicies of inliers
        """

        num_pts = key_2d_pts.shape[1]

        max_inlier_num = 0
        best_inlier_indices = []
        best_rot = np.identity(3)
        best_loc = np.zeros((3, 1))
        for it in range(0, ransac_config.iteration):

            # generate the six indices randomly
            indices = random.sample(range(num_pts), 6)
            key_2d_pts_6 = key_2d_pts[:, indices]
            tri_3d_pts_6 = tri_3d_pts[:, indices]

            key_2d_pts_6_in_cam_coord = np.linalg.inv(intrinsic_mat) @ key_2d_pts_6

            rot, loc = self.__estimate_six_pts(key_2d_pts_6_in_cam_coord, tri_3d_pts_6)
            proj = intrinsic_mat @ np.hstack((rot.T, rot.T @ -loc))

            inlier_num = 0
            inlier_indices = []

            # Loop through all points
            for idx in range(num_pts):
                # both pt_2d and proj_2d are in the image coordinate
                pt_2d = key_2d_pts[:, idx:idx + 1]
                pt_3d = tri_3d_pts[:, idx:idx + 1]
                proj_2d = proj @ pt_3d
                proj_2d = proj_2d / proj_2d[2]  # normalization
                err = math.sqrt(np.sum((pt_2d - proj_2d) ** 2))

                if err < ransac_config.inlier_threshold:
                    inlier_num += 1
                    inlier_indices.append(idx)

            if inlier_num > max_inlier_num:
                max_inlier_num = inlier_num
                best_inlier_indices = inlier_indices
                best_rot = rot
                best_loc = loc

        return best_rot, best_loc, max_inlier_num, best_inlier_indices

    #########################################################################
    def __estimate_six_pts(self, key_2d_pts_6, tri_3d_pts_6):
        """
        Determine camera rotation R and camera location C

        @Param key_2d_pts_6
        Six 2D key points (in the camera coordinate) in homogeneous form

        @Param tri_3d_pts_6
        Six 3D triangulated points in homogeneous form

        @Return rot:
        Camera rotation

        @Return loc:
        Camera location
        """

        p2d = key_2d_pts_6
        p3d = tri_3d_pts_6

        w = np.zeros((12, 12))
        for i in range(0, 12, 2):
            pt_idx = int(i / 2)

            w[i][0] = p2d[2][pt_idx] * p3d[0][pt_idx]
            w[i][1] = p2d[2][pt_idx] * p3d[1][pt_idx]
            w[i][2] = p2d[2][pt_idx] * p3d[2][pt_idx]
            w[i][3] = p2d[2][pt_idx]
            w[i][4] = 0
            w[i][5] = 0
            w[i][6] = 0
            w[i][7] = 0
            w[i][8] = -p2d[0][pt_idx] * p3d[0][pt_idx]
            w[i][9] = -p2d[0][pt_idx] * p3d[1][pt_idx]
            w[i][10] = -p2d[0][pt_idx] * p3d[2][pt_idx]
            w[i][11] = -p2d[0][pt_idx]

            w[i + 1][0] = 0
            w[i + 1][1] = 0
            w[i + 1][2] = 0
            w[i + 1][3] = 0
            w[i + 1][4] = p2d[2][pt_idx] * p3d[0][pt_idx]
            w[i + 1][5] = p2d[2][pt_idx] * p3d[1][pt_idx]
            w[i + 1][6] = p2d[2][pt_idx] * p3d[2][pt_idx]
            w[i + 1][7] = p2d[2][pt_idx]
            w[i + 1][8] = -p2d[1][pt_idx] * p3d[0][pt_idx]
            w[i + 1][9] = -p2d[1][pt_idx] * p3d[1][pt_idx]
            w[i + 1][10] = -p2d[1][pt_idx] * p3d[2][pt_idx]
            w[i + 1][11] = -p2d[1][pt_idx]

        # Apply SVD to find the best solution.
        u, s, vh = np.linalg.svd(w)
        vh = vh.transpose()[:, -1]  # the last column
        cam_mat = np.reshape(vh, (3, 4))
        rot_t_noise = cam_mat[:, 0:3]

        # Denoise rotation, R
        uu, ss, vvh = np.linalg.svd(rot_t_noise)
        rot = (uu @ vvh).T  # uu and vvh are orthonormal matrices, of which determinant is +1 or -1

        # Determine location, C
        loc = (rot @ -cam_mat[:, 3:4]) / ss[0]

        # Check the rotation property: det(rotation) is 1
        if np.linalg.det(rot) < 0:
            rot = -rot
            loc = -loc

        return rot, loc

    #########################################################################
    def __construct_jacobian_quaternion(self, quaternion):
        """
        Construct a Jacobian matrix of (rotation matrix)/(quaternion)

        @Param quaternion
        A quaternion
        numpy.array(4, 1)

        @Return jac_rq
        A Jacobian matrix
        numpy.array(9, 4)
        """

        jac_rq = np.zeros((9, 4))
        w = 0
        x = 1
        y = 2
        z = 3
        q2 = quaternion * 2
        q4 = quaternion * 4

        jac_rq[0][0] = 0
        jac_rq[0][1] = 0
        jac_rq[0][2] = -q4[y]
        jac_rq[0][3] = -q4[z]

        jac_rq[1][0] = -q2[z]
        jac_rq[1][1] = q2[y]
        jac_rq[1][2] = q2[x]
        jac_rq[1][3] = -q2[w]

        jac_rq[2][0] = q2[y]
        jac_rq[2][1] = q2[z]
        jac_rq[2][2] = q2[w]
        jac_rq[2][3] = q2[x]

        jac_rq[3][0] = q2[z]
        jac_rq[3][1] = q2[y]
        jac_rq[3][2] = q2[x]
        jac_rq[3][3] = q2[w]

        jac_rq[4][0] = 0
        jac_rq[4][1] = -q4[x]
        jac_rq[4][2] = 0
        jac_rq[4][3] = -q4[z]

        jac_rq[5][0] = -q2[x]
        jac_rq[5][1] = -q2[w]
        jac_rq[5][2] = q2[z]
        jac_rq[5][3] = q2[y]

        jac_rq[6][0] = -q2[y]
        jac_rq[6][1] = q2[z]
        jac_rq[6][2] = -q2[w]
        jac_rq[6][3] = q2[x]

        jac_rq[7][0] = q2[x]
        jac_rq[7][1] = q2[w]
        jac_rq[7][2] = q2[z]
        jac_rq[7][3] = q2[y]

        jac_rq[8][0] = 0
        jac_rq[8][1] = -q4[x]
        jac_rq[8][2] = -q4[y]
        jac_rq[8][3] = 0

        return jac_rq

    #########################################################################
    def __construct_jacobian_rotation(self, rot, loc, pt_3d):
        """
        Construct a Jacobian matrix of (reprojected matrix) / rotation matrix

        @Param rot
        The camera rotation
        numpy.array((3, 3))

        @Param loc
        The camera location
        numpy.array((3, 1))

        @Param pt_3d
        The 3D point
        numpy.array((4, 1))

        @Return jac_rr
        The Jacobian matrix
        numpy.array((2, 9))
        """

        # TODO - sanity check on rot, loc, pt_3d
        proj_mat = np.hstack((rot.T, rot.T @ -loc))
        proj_pt = proj_mat @ pt_3d

        num_u = proj_pt[0:1, :]
        num_v = proj_pt[1:2, :]
        num_w = proj_pt[2:3, :]
        num_w_square = np.square(proj_pt[2:3, :])

        pos_diff = pt_3d[0:3, :] - loc
        x = 0
        y = 1
        z = 2

        jac_rr = np.zeros((2, 9))

        jac_rr[0][0] = num_w * pos_diff[x]
        jac_rr[0][1] = 0
        jac_rr[0][2] = -num_u * pos_diff[x]

        jac_rr[0][3] = num_w * pos_diff[y]
        jac_rr[0][4] = 0
        jac_rr[0][5] = -num_u * pos_diff[y]

        jac_rr[0][6] = num_w * pos_diff[z]
        jac_rr[0][7] = 0
        jac_rr[0][8] = -num_u * pos_diff[z]

        jac_rr[1][0] = 0
        jac_rr[1][1] = num_w * pos_diff[x]
        jac_rr[1][2] = -num_v * pos_diff[x]

        jac_rr[1][3] = 0
        jac_rr[1][4] = num_w * pos_diff[y]
        jac_rr[1][5] = -num_v * pos_diff[y]

        jac_rr[1][6] = 0
        jac_rr[1][7] = num_w * pos_diff[z]
        jac_rr[1][8] = -num_v * pos_diff[z]

        jac_rr /= num_w_square
        return jac_rr

    #########################################################################
    def __construct_jacobian_location(self, rot, loc, pt_3d):
        """
        Construct a Jacobian matrix of (reprojected matrix) / location vector

        @Param rot
        The projection matrix
        numpy.array((3, 1))

        @Param pt_3d
        The 3D point
        numpy.array((4, 1))

        @Return jac_rr
        The Jacobian matrix
        numpy.array((2, 3))
        """

        # TODO - sanity check on rot, loc, pt_3d
        proj_mat = np.hstack((rot.T, rot.T @ -loc))
        proj_pt = proj_mat @ pt_3d

        num_u = proj_pt[0:1, :]
        num_v = proj_pt[1:2, :]
        num_w = proj_pt[2:3, :]
        num_w_square = np.square(proj_pt[2:3, :])

        jac_rc = np.zeros((2, 3))

        jac_rc[0][0] = num_w * -rot[0][0] - num_u * -rot[0][2]
        jac_rc[0][1] = num_w * -rot[1][0] - num_u * -rot[1][2]
        jac_rc[0][2] = num_w * -rot[2][0] - num_u * -rot[2][2]

        jac_rc[1][0] = num_w * -rot[0][1] - num_v * rot[0][2]
        jac_rc[1][1] = num_w * -rot[1][1] - num_v * rot[1][2]
        jac_rc[1][2] = num_w * -rot[2][1] - num_v * rot[2][2]

        jac_rc /= num_w_square

        return jac_rc


########################################################################################################################

if __name__ == '__main__':
    """
    Run the CamposeProcessor test
    """

    print('=== Start CamposeProcessor Unit Test ===')

    import os

    cur_path = os.path.dirname(__file__)
    test_dataset_path = os.path.join(cur_path, 'test_dataset')

    # Load testing data
    ref_r = np.load(test_dataset_path + '/ess_self_r.npy')
    ref_r = ref_r.T  # ref_r is the rotation matrix from destination
                     # to reference in destination coordinate

    ref_c = np.load(test_dataset_path + '/ess_self_c.npy')
    esse_mat_ = np.load(test_dataset_path + '/ess_ess_mat.npy')
    ref_k_ = np.load(test_dataset_path + '/ess_intrinsic_mat.npy')
    key_2d_pts_from1 = np.load(test_dataset_path + '/ess_pixel_pt1.npy')
    key_2d_pts_from2 = np.load(test_dataset_path + '/ess_pixel_pt2.npy')

    key_2d_pts_from1 = key_2d_pts_from1.T
    key_2d_pts_from2 = key_2d_pts_from2.T

    # Load ground truth
    r1_truth = np.load(test_dataset_path + '/ess_r1.npy')
    r2_truth = np.load(test_dataset_path + '/ess_r2.npy')
    r1_truth = r1_truth.T
    r2_truth = r2_truth.T
    # r1_truth and r2_truth are calculated from OpenCV,
    # and they are the rotation matrices from the destination
    # to the reference coordinate in the destination coordinate

    c1_truth = np.load(test_dataset_path + '/ess_c1.npy')
    c2_truth = np.load(test_dataset_path + '/ess_c2.npy')
    r1t1_3d_pts_truth = np.load(test_dataset_path + '/ess_points_3d_r1t1.npy')
    r1t2_3d_pts_truth = np.load(test_dataset_path + '/ess_points_3d_r1t2.npy')
    r2t1_3d_pts_truth = np.load(test_dataset_path + '/ess_points_3d_r2t1.npy')
    r2t2_3d_pts_truth = np.load(test_dataset_path + '/ess_points_3d_r2t2.npy')

    # Initialize the CamposeProcessor, and extract the four possible combinations
    # of rotation and location.

    # The reference camera pose
    # ref_cam_pose_ = np.hstack((ref_r, ref_c))
    ref_proj_ = ref_k_ @ np.hstack((ref_r.T, ref_r.T @ -ref_c))

    # print('    --- Start Camera Pose Extraction From Essential Matrix ---')

    # Set up params
    RANSAC_CONFIG = RansacConfig(inlier_threshold=8.0,
                                 subset_confidence=0.99,
                                 # the desired probability that the result from this model is 0.99 "reliable"
                                 sample_confidence=0.75,  # (inlier data / total data)
                                 sample_num=6,
                                 iteration=300)
    DAMPING_FACTOR = 5
    ITERATION = 200

    # Initialize a CamposeProcessor object
    cp = CamposeProcessor(RANSAC_CONFIG, DAMPING_FACTOR, ITERATION)
    r1_, r2_, c1_, c2_ = cp.extract_cam_pose_from_essential_mat(esse_mat_)

    r1_diff = np.sum(np.absolute(r1_ - r1_truth))
    r2_diff = np.sum(np.absolute(r2_ - r2_truth))
    c1_diff = np.sum(np.absolute(c1_ - c1_truth))
    c2_diff = np.sum(np.absolute(c2_ - c2_truth))

    # print('    --- Camera pose extraction comparing to the ground truth')

    if r1_diff >= 1e-2 or r2_diff >= 1e-2 or c1_diff >= 1e-2 or c2_diff >= 1e-2:
        import sys

        print('    --- Camera pose extraction comparison fails : r1_diff {:.6f}, r2_diff {:.6f}, c1_diff {:.6f},'
              'c2_diff {:.6f}'.format(r1_diff, r2_diff, c1_diff, c2_diff))

        from sys import platform

        if platform == "darwin":
            print('        !!! It is okay because Mac gives different result from Linux !!!')
        else:
            sys.exit()
    else:
        print('    --- Camera pose extraction comparison passes : r1_diff {:.6f}, r2_diff {:.6f}, c1_diff {:.6f}, '
              'c2_diff {:.6f}'.format(r1_diff, r2_diff, c1_diff, c2_diff))

    # Initialize a TriangulationProcessor
    from triangulation_processor import TriangulationProcessor

    tp = TriangulationProcessor()
    damping_factor_ = 10
    r1c1_proj = ref_k_ @ np.hstack((r1_.T, -r1_.T @ c1_))
    r1c2_proj = ref_k_ @ np.hstack((r1_.T, -r1_.T @ c2_))
    r2c1_proj = ref_k_ @ np.hstack((r2_.T, -r2_.T @ c1_))
    r2c2_proj = ref_k_ @ np.hstack((r2_.T, -r2_.T @ c2_))

    proj_candidates = [r1c1_proj, r1c2_proj, r2c1_proj, r2c2_proj]

    # Construct triangulated 3D points set from all projection candidates
    tri_pts_candidates = []
    # for idx in range(4):
    #     projs = [ref_proj, proj_candidates[idx]]
    #     matched_pts = [key_2d_pts_from1.T, key_2d_pts_from2.T]
    #     tri_pts = tp.triangulate(projs, matched_pts, damping_factor_)
    #     tri_pts = np.array(tri_pts)
    #     tri_pts_candidates.append(tri_pts)
    r1t1_3d = np.load(test_dataset_path + '/ess_points_3d_r1t1_result.npy')
    r1t1_3d = r1t1_3d.T[0]
    tri_pts_candidates.append(r1t1_3d)

    r1t2_3d = np.load(test_dataset_path + '/ess_points_3d_r1t2_result.npy')
    r1t2_3d = r1t2_3d.T[0]
    tri_pts_candidates.append(r1t2_3d)

    r2t1_3d = np.load(test_dataset_path + '/ess_points_3d_r2t1_result.npy')
    r2t1_3d = r2t1_3d.T[0]
    tri_pts_candidates.append(r2t1_3d)

    r2t2_3d = np.load(test_dataset_path + '/ess_points_3d_r2t2_result.npy')
    r2t2_3d = r2t2_3d.T[0]
    tri_pts_candidates.append(r2t2_3d)

    best_idx_, most_valid_indices_ = cp.disambiguate_cam_pose_four(ref_proj_,
                                                                   proj_candidates,
                                                                   tri_pts_candidates)

    # In this experiment, R1T2 is the valid combination
    if best_idx_ != 1:
        import sys
        print('    --- Best index should be 1 but {}'.format(best_idx_))
        sys.exit()

    rot_diff = np.sum(np.absolute(r1_truth - r1_))
    loc_diff = np.sum(np.absolute(c2_truth - c2_))
    if rot_diff > 1e-2 or loc_diff > 1e-2:
        import sys

        print('    --- Best camera pose selection from four combination fails : '
              'rot_diff {:.6f}, loc_diff {:.6f}'.format(rot_diff, loc_diff))

        from sys import platform

        if platform == "darwin":
            print('        !!! It is okay because Mac gives different result from Linux !!!')
        else:
            sys.exit()

    # print('    --- Complete Camera Pose Extraction from Essential Matrix ---')

    ####################################################################################################################

    # print('    --- Start Camera Pose From Linear PnP ---')

    is_use_artificial_setup = False

    rot_truth = np.identity(3)
    loc_truth = np.zeros((3, 1))
    tra_truth = np.zeros((3, 1))
    cam_mat_truth = np.zeros((3, 4))
    proj_truth = np.zeros((3, 4))

    pnp_2d_pts = None
    pnp_2d_pts_homogeneous = None

    pnp_3d_pts = np.load(test_dataset_path + '/pnp_points_3d.npy')
    ones = np.ones((1, pnp_3d_pts.shape[0]))

    pnp_3d_pts = pnp_3d_pts.T
    pnp_3d_pts_homogeneous = np.vstack((pnp_3d_pts, ones))

    if is_use_artificial_setup:
        # Use the following website to create an artificial rotation matrix ground truth
        # https://danceswithcode.net/engineeringnotes/rotations_in_3d/demo3D/rotations_in_3d_tool.html
        # rot_truth = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        rot_truth = np.array([[0.433, 0.75, 0.5],
                              [0.25, 0.433, -0.8661],
                              [-0.8661, 0.5, 0.0]])

        # (yaw, pitch, roll) = (-60.287, 65.113, 130.10) in degrees
        # rot_truth = np.array([[0.2085, 0.3654, 0.9071],
        #                       [0.9034, 0.2832, -0.3219],
        #                       [-0.3747, 0.8866, -0.2711]])

        loc_truth = (np.array([[1, 2, -1]])).T
        tra_truth = rot_truth.T @ -loc_truth

        cam_mat_truth = np.hstack((rot_truth.T, tra_truth))
        proj_truth = ref_k_ @ cam_mat_truth

        pnp_2d_pts_homogeneous = proj_truth @ pnp_3d_pts_homogeneous
        pnp_2d_pts_homogeneous = pnp_2d_pts_homogeneous / pnp_2d_pts_homogeneous[2, :]  # normalization
        pnp_2d_pts = pnp_2d_pts_homogeneous[0:2, :]
    else:
        # pnp_rotation.npy and pnp_translation.npy are OpenCV results from another project
        rot_transpose_truth = np.load(test_dataset_path + '/pnp_rotation.npy')
        rot_truth = rot_transpose_truth.T
        tra_truth = np.load(test_dataset_path + '/pnp_translation.npy')
        loc_truth = rot_truth @ -tra_truth

        cam_mat_truth = np.hstack((rot_truth.T, tra_truth))
        proj_truth = ref_k_ @ cam_mat_truth

        pnp_2d_pts = np.load(test_dataset_path + '/pnp_points_2d.npy')
        pnp_2d_pts = pnp_2d_pts.T
        pnp_2d_pts_homogeneous = np.vstack((pnp_2d_pts, ones))

    inlier_indices_, rot_, loc_ = cp.linear_estimate_cam_pose_pnp(pnp_2d_pts_homogeneous,
                                                                   pnp_3d_pts_homogeneous,
                                                                   ref_k_,
                                                                   RANSAC_CONFIG)

    cam_mat_ = np.hstack((rot_.T, rot_.T @ -loc_))

    # Compare with OpenCV result
    # Based on
    # https://stackoverflow.com/questions/14444433/calculate-camera-world-position-with-opencv-python
    # https://stackoverflow.com/questions/14515200/python-opencv-solvepnp-yields-wrong-translation-vector?lq=1
    # https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
    # what solvePnPRansac() returns are the camera matrix
    # So Rt and t(=Rt @ -loc_)
    import cv2

    pnp_3d_pts_t = pnp_3d_pts.T
    pnp_2d_pts_t = pnp_2d_pts.T
    _, Rt_cv, t_cv, _ = cv2.solvePnPRansac(pnp_3d_pts_t[:, np.newaxis], pnp_2d_pts_t[:, np.newaxis], ref_k_,
                                           None,
                                           confidence=0.99, reprojectionError=20.0, flags=cv2.SOLVEPNP_ITERATIVE)
    Rt_cv, _ = cv2.Rodrigues(Rt_cv)
    # Rt_cv is the rotation matrix from destination to reference in the
    # destination coordinate

    # Calculate Euler angles
    from scipy.spatial.transform import Rotation as Rot

    euler_angle_res = Rot.from_matrix(rot_).as_euler('zyx', degrees=True)
    euler_angle_truth = Rot.from_matrix(rot_truth).as_euler('zyx', degrees=True)
    euler_angle_cv = Rot.from_matrix(Rt_cv.T).as_euler('zyx', degrees=True)

    rot_diff = np.absolute(euler_angle_truth - euler_angle_res)
    loc_diff = math.sqrt(np.sum(np.square(loc_truth - loc_)))

    if np.any((rot_diff >= 1)) or loc_diff >= 1e-1:
        import sys

        print('    --- Camera Pose from linear PnP fails : rot_diff ({:.6f}, {:.6f}, {:.6f}), loc_diff {:.6f}'
              .format(rot_diff[0], rot_diff[1], rot_diff[2], loc_diff))
        sys.exit()
    else:
        print('    --- Camera Pose from linear PnP passes : rot_diff ({:.6f}, {:.6f}, {:.6f}), loc_diff {:.6f}'
              .format(rot_diff[0], rot_diff[1], rot_diff[2], loc_diff))

    # print('    --- Complete Camera Pose From Linear PnP ---')

    # print('    --- Start Camera Pose From Nonlinear PnP ---')

    pnp_2d_pts_homogeneous_inlier = pnp_2d_pts_homogeneous[:, inlier_indices_]
    pnp_3d_pts_homogeneous_inlier = pnp_3d_pts_homogeneous[:, inlier_indices_]

    ref_rot_, ref_loc_ = cp.nonlinear_estimate_cam_pose_pnp(pnp_2d_pts_homogeneous_inlier,
                                                            pnp_3d_pts_homogeneous_inlier,
                                                            ref_k_,
                                                            rot_, loc_, DAMPING_FACTOR, ITERATION)

    euler_angle_res = Rot.from_matrix(ref_rot_).as_euler('zyx', degrees=True)
    rot_diff = np.absolute(euler_angle_truth - euler_angle_res)
    loc_diff = math.sqrt(np.sum(np.square(loc_truth - ref_loc_)))

    if np.any((rot_diff >= 1)) or loc_diff >= 1e-1:
        import sys

        print('    --- Camera Pose from nonlinear PnP fails : rot_diff ({:.6f}, {:.6f}, {:.6f}), loc_diff {:.6f}'
              .format(rot_diff[0], rot_diff[1], rot_diff[2], loc_diff))
        sys.exit()
    else:
        print('    --- Camera Pose from nonlinear PnP passes : rot_diff ({:.6f}, {:.6f}, {:.6f}), loc_diff {:.6f}'
              .format(rot_diff[0], rot_diff[1], rot_diff[2], loc_diff))

    ####################################################################################################################
    # Re-create RansacConfig to reset random seed
    RANSAC_CONFIG = RansacConfig(inlier_threshold=8.0,
                                  subset_confidence=0.99,
                                  # the desired probability that the result from this model is 0.99 "reliable"
                                  sample_confidence=0.75,  # (inlier data / total data)
                                  sample_num=6,
                                  iteration=300)
    # _w: mean whole cam pos pno (linear + nonlinear)
    inliers_idx_list, ref_rot_w, ref_loc_w = cp.estimate_cam_pose_pnp(pnp_2d_pts_homogeneous,
                                                                       pnp_3d_pts_homogeneous,
                                                                       ref_k_,
                                                                       RANSAC_CONFIG,
                                                                       DAMPING_FACTOR,
                                                                       ITERATION)

    euler_angle_res_w = Rot.from_matrix(ref_rot_w).as_euler('zyx', degrees=True)
    rot_diff_w = np.absolute(euler_angle_res_w - euler_angle_res)
    loc_diff_w = math.sqrt(np.sum(np.square(ref_loc_w - ref_loc_)))

    if np.any((rot_diff_w >= 1e-5)) or loc_diff_w >= 1e-5:
        import sys

        print('    --- Camera Pose from whole PnP fails : rot_diff ({:.6f}, {:.6f}, {:.6f}), loc_diff {:.6f}'
              .format(rot_diff_w[0], rot_diff_w[1], rot_diff_w[2], loc_diff_w))
        sys.exit()
    else:
        print('    --- Camera Pose from whole PnP passes : rot_diff ({:.6f}, {:.6f}, {:.6f}), loc_diff {:.6f}'
              .format(rot_diff_w[0], rot_diff_w[1], rot_diff_w[2], loc_diff_w))

    print('=== Complete CamposeProcessor Unit Test ===')
