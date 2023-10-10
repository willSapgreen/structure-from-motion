import logging
import numpy as np
from utils import KeyPt, TriPt


########################################################################################################################
class TriangulationProcessor:
    """
    Determine 3D positions from 2D matched key points from TWO images
    """

    def __init__(self, damping_factor=0.5, iteration=100):
        """
        Constructor
        """
        self.damping_factor = damping_factor
        self.iteration = iteration
        self.tri_pts = None

    #########################################################################
    def add_tri_pt(self, tri_pt):
        """
        Add a new TriPt
        """
        if not np.any(self.tri_pts):
            self.tri_pts = tri_pt
        else:
            self.tri_pts = np.hstack((self.tri_pts, tri_pt))

    #########################################################################
    def triangulate(self, projs, matched_pairs, damping_factor=None,
                    iteration=None):
        """
        Triangulate the 2D matched key points in n views to determine 3D points
        through linear triangulation first and then nonlinear triangulation methods

        @Param projs
        A list of projections
        [proj1, proj2]
        projX is a numpy.array(3, 4)

        @Param matched_pairs
        A list of key points in each view
        [key_pts1, key_pts2]
        key_ptsX is a KeyPt(3, m) array, where m is the number of points

        @Param damping_factor
        The damping factor used in the Levenberg-Marquardt algorithm (LMA)

        @Param iteration
        The iteration times used in the Levenberg_marquardt algorithm (LMA)

        @Return refined_3d_pts
        The refined triangulated 3D points
        TriPt(4, m), where m is the number of points
        """

        if not damping_factor:
            damping_factor = self.damping_factor
        if not iteration:
            iteration = self.iteration

        num_projs = len(projs)
        num_views = len(matched_pairs)
        if num_projs != num_views:
            logging.warning('%s : the numbers of views and of projections are different, %d and %d',
                            self.__class__.__name__, num_views, num_projs)
            raise ValueError("different numbers of views and projections : {} - {}"
                             .format(num_views, num_projs))
        if num_projs < 2:
            logging.warning('%s : insufficient projections, %d',
                            self.__class__.__name__, num_projs)
            raise ValueError("insufficient projections : {}"
                             .format(num_projs))

        # Something to know
        # . a projection is a camera matrix, K [transpose(R) transpose(R) @ -translation], a 3x4 matrix
        # Ref:
        # https://docs.opencv.org/4.7.0/d0/dbd/group__triangulation.html
        # https://github.com/alexflint/triangulation

        # For experiment
        # self.__linear_triangulate_two_views_comparison(projs, matched_pairs)

        init_3d_pts = self.linear_triangulate(projs, matched_pairs)
        refined_3d_pts = self.nonlinear_triangulate(init_3d_pts, projs, matched_pairs,
                                                    damping_factor, iteration)
        return refined_3d_pts

    #########################################################################
    def linear_triangulate(self, projs, matched_pairs):
        """
        Linear triangulate - algebra error minimization

        @Param projs
        A list of projections
        [proj1, proj2]
        projX is a 3x4 numpy.array

        @Param matched_pairs
        A list of key points in each view
        [key_pts1, key_pts2]
        key_ptsX is a numpy.array(3, m), where m is the number of points

        @Param num_views
        The number of views

        @Param num_pts
        The number of points to be triangulated

        @Return tri_3d_pts:
        A list of triangulated 3D points
        [tri_pt1, tri_pt2, ..., tri_ptm]
        tri_ptX is a numpy.array(4, 1)
        """

        # Sanity check
        if len(matched_pairs) != len(projs) != 2:
            from inspect import currentframe
            print('{}:{} - num of projs {} and matched pairs {} need to be 2'.format(
                self.__class__.__name__, currentframe().f_code.co_name,
                len(projs), len(matched_pairs)))
            return None

        if matched_pairs[0].shape[1] != matched_pairs[1].shape[1]:
            from inspect import currentframe
            print('{}:{} - matched pairs number does not match {} vs {}'.format(
                self.__class__.__name__, currentframe().f_code.co_name,
                matched_pairs[0].shape[1], matched_pairs[1].shape[1]))
            return None

        num_views = len(matched_pairs)
        num_pts = matched_pairs[0].shape[1]

        tri_pts = TriPt(num_pts)
        for pt_idx in range(num_pts):  # loop through each 3D point
            # SVD with 3D point variables
            # a @ [X, Y, Z, W] = 0
            # Ref: https://amytabb.com/tips/tutorials/2021/10/31/triangulation-DLT-2-3/
            a = np.zeros([2 * num_views, 4])
            for v_idx in range(num_views):  # loop through each 3D point's corresponding projections
                u = matched_pairs[v_idx][0, pt_idx]
                v = matched_pairs[v_idx][1, pt_idx]
                proj = projs[v_idx]
                a[2 * v_idx:2 * v_idx + 1, :] = u * proj[2:3, :] - proj[0:1, :]
                a[2 * v_idx + 1:2 * v_idx + 1 + 1, :] = v * proj[2:3, :] - proj[1:2, :]

            # Apply SVD
            u, s, vh = np.linalg.svd(a)
            vh = vh.transpose()[:, -1]
            svd_3d_x = np.reshape(vh, (4, 1))
            svd_3d_x = (svd_3d_x / svd_3d_x[3][0])  # normalization

            # Store into the result
            tri_pts[:, pt_idx:pt_idx+1] = svd_3d_x[:, :]

        return tri_pts

    #########################################################################
    def nonlinear_triangulate(self, init_3d_pts, projs, matched_pairs,
                              damping_factor=None, iteration=None):
        """
        Nonlinear triangulate - geometric error minimization

        m = num_pts
        n = num_views

        @Return init_3d_pts:
        A list of triangulated 3D points
        TriPt(4, m), where m is the number of points

        @Param projs
        A list of projections
        [proj1, proj2]
        projX is a numpy.array(3, 4)

        @Param matched_pairs
        A list of key points in each view
        [key_pts1, key_pts2]
        key_ptsX is a KeyPt(3, m), where m is the number of points

        @Param damping_factor
        The damping factor used in the Levenberg-Marquardt algorithm (LMA)

        @Param num_views
        The number of views

        @Param num_pts
        The number of points to be triangulated

        @Return refined_3d_pts
        A list of triangulated 3D points.
        TriPt(4, m), where m is the number of points
        """

        # TODO
        # Sanity check: the num of init_3d_pts, projs, matched_pairs[0] and matched_pairs[1]
        #               need to be 2

        if not damping_factor:
            damping_factor = self.damping_factor
        if not iteration:
            iteration = self.iteration

        num_pts = matched_pairs[0].shape[1]
        num_views = len(projs)

        refined_3d_pts = np.copy(init_3d_pts)
        for it_idx in range(iteration):
            for pt_idx in range(num_pts):  # Loop through each 3D point
                tri_3d_pt = refined_3d_pts[:, pt_idx:pt_idx + 1]

                # Extract and accmulate
                # the corresponding projected point (2D) from each view
                m_pts = []
                for v_idx in range(num_views):
                    matched_pt_uv = matched_pairs[v_idx][0:2, pt_idx:pt_idx + 1]
                    m_pts.append(matched_pt_uv)

                err = self.__calculate_reprojection_error(tri_3d_pt,
                                                          projs,
                                                          m_pts,
                                                          num_views)

                jac = self.construct_jacobian_matrix(tri_3d_pt, projs, num_views)

                delta = np.linalg.inv(jac.T.dot(jac) + damping_factor * np.identity(3)).dot(jac.T).dot(err.T)
                tri_3d_pt[0:3] -= delta
        # Ref
        # https://mathoverflow.net/questions/257699/gauss-newton-vs-gradient-descent-vs-levenberg-marquadt-for-least-squared-method

        # print('init_3d_pts {}'.format(init_3d_pts))
        # print('refined_3d_pts {}'.format(refined_3d_pts))
        return refined_3d_pts

    #########################################################################
    def construct_jacobian_matrix(self, tri_3d_pt, projs, num_views):
        """
        Construct the Jacobian matrix of a 3D point

        @Param tri_3d_pt
        The triangulated 3D point in the homogeneous form
        numpy.array(4, 1)

        @Param projs
        A list of projections
        [proj1, proj2, ..., projn]
        projX is a numpy.array(3, 4)

        @Param num_views
        The number of views

        @Return jac:
        numpy.array((num_views x 2) x 3)
        """

        # Because the input tri_3d_pts are already normalized (W = 1),
        # we only need 3 cols
        jac = np.zeros((num_views * 2, 3))

        for v_idx in range(num_views):
            proj = projs[v_idx]
            proj_pt = proj @ tri_3d_pt
            jac_u = (proj_pt[2] * np.array([proj[0, 0], proj[0, 1], proj[0, 2]])
                     - proj_pt[0] * np.array([proj[2, 0], proj[2, 1], proj[2, 2]])) / proj_pt[2] ** 2
            jac_v = (proj_pt[2] * np.array([proj[1, 0], proj[1, 1], proj[1, 2]])
                     - proj_pt[1] * np.array([proj[2, 0], proj[2, 1], proj[2, 2]])) / proj_pt[2] ** 2
            jac[v_idx * 2:v_idx * 2 + 1, :] = jac_u
            jac[v_idx * 2 + 1:v_idx * 2 + 2, :] = jac_v

        return jac

    #########################################################################
    def __calculate_reprojection_error(self, tri_3d_pt, projs, matched_pairs,
                                       num_views):
        """
        Calculate reprojection error (geometric error)

        @Param tri_3d_pt
        The triangulated 3D point in the homogeneous form
        numpy.array(4, 1)

        @Param projs
        A list of projections
        [proj1, proj2, ..., projn]
        projX is a numpy.array(3, 4)

        @Param matched_pairs
        A list of key points in each view
        [key_pt1, key_pt2, ..., key_ptn]
        key_ptX is a 2x1 numpy.array, containing (u, v)

        @Param num_views
        The number of views

        @Return error
        numpy.array(1 x (2 x num_view))
        """
        error = np.zeros((1, 2 * num_views))

        for v_idx in range(num_views):
            proj = projs[v_idx]
            proj_pt = proj @ tri_3d_pt
            proj_pt /= proj_pt[2]  # normalization

            error[:, 2 * v_idx] = (proj_pt[0] - matched_pairs[v_idx][0])
            error[:, 2 * v_idx + 1] = (proj_pt[1] - matched_pairs[v_idx][1])

        return error

    #########################################################################
    # Compare the following linear triangulation methods on two views
    # 1. SVD with only 3D point variables (X, Y, Z, W) <- this one is closest to OpenCV result
    #    A @ [X, Y, Z, W] = 0
    #    Ref: https://amytabb.com/tips/tutorials/2021/10/31/triangulation-DLT-2-3/
    # 2. SVD with 3D point and scaling factor variables (X, Y, Z, W, k, s)
    #    M @ [X, Y, Z, W, k, t, ...] = 0.
    #    where (X,Y,Z,W) is the triangulated 3D point
    #    (k, t, ...) are the scale factors for each 3D -> 2D projection
    #    ex. k * 2d_pt_k1 = proj_k * 3d_pt_1
    #    t * 2d_pt_t1 = proj_t * 3d_pt_1
    #    Ref: https://gist.github.com/davegreenwood/e1d2227d08e24cc4e353d95d0c18c914
    # 3. Normal equation
    # 4. OpenCV
    def __linear_triangulate_two_views_comparison(self, projs, matched_pairs):  # (Abandoned. Need to update)
        """Experiment the results from SVD, normal equations, and OpenCV"""

        # Something to know
        # To resolve the linear least square question,
        # there are major three methods
        # 1. SVD deocmposition (slowest but most accurate)
        # 2. QR decomposition
        # 3. normal equation   (fastest but least accurate)
        # Ref
        # https://amytabb.com/til/2021/12/16/solving-least-squares/
        # https://eigen.tuxfamily.org/dox/group__LeastSquares.html

        # Set up parameters
        (matched_rows, matched_cols) = matched_pairs.shape
        # num_pts = matched_rows
        num_views = int(matched_cols / 3)

        p1 = projs[:, 0:4]
        p2 = projs[:, 4:8]

        u1 = matched_pairs[0, 0]
        v1 = matched_pairs[0, 1]
        u2 = matched_pairs[0, 3]
        v2 = matched_pairs[0, 4]

        # Method I
        a = np.array(
            [[u1 * p1[2, 0] - p1[0, 0], u1 * p1[2, 1] - p1[0, 1], u1 * p1[2, 2] - p1[0, 2], u1 * p1[2, 3] - p1[0, 3]],
             [v1 * p1[2, 0] - p1[1, 0], v1 * p1[2, 1] - p1[1, 1], v1 * p1[2, 2] - p1[1, 2], v1 * p1[2, 3] - p1[1, 3]],
             [u2 * p2[2, 0] - p2[0, 0], u2 * p2[2, 1] - p2[0, 1], u2 * p2[2, 2] - p2[0, 2], u2 * p2[2, 3] - p2[0, 3]],
             [v2 * p2[2, 0] - p2[1, 0], v2 * p2[2, 1] - p2[1, 1], v2 * p2[2, 2] - p2[1, 2], v2 * p2[2, 3] - p2[1, 3]]])

        u, s, vh = np.linalg.svd(a)
        vh = vh.transpose()[:, -1]
        svd_3d_x = np.reshape(vh, (1, 4))
        svd_3d_x = (svd_3d_x / svd_3d_x[0][3])
        print('Method I - SVD with only 3D point variable: {}'.format(svd_3d_x[0]))

        # Method II
        m = np.zeros([3 * num_views, 4 + num_views])
        for v_idx in range(num_views):
            m[3 * v_idx:3 * v_idx + 3, :4] = projs[:, v_idx * 4:v_idx * 4 + 4]
            m[3 * v_idx:3 * v_idx + 3, 4 + v_idx] = -(matched_pairs[0:1, v_idx * 3:v_idx * 3 + 3])

        u, s, vh = np.linalg.svd(m)
        tri_3d_pt = vh[-1, :4]
        tri_3d_pt /= tri_3d_pt[3]
        print('Method II - SVD with 3D point variable and scaling factors: {}'.format(tri_3d_pt))

        # Method III
        a = np.array([[u1 * p1[2, 0] - p1[0, 0], u1 * p1[2, 1] - p1[0, 1], u1 * p1[2, 2] - p1[0, 2]],
                      [v1 * p1[2, 0] - p1[1, 0], v1 * p1[2, 1] - p1[1, 1], v1 * p1[2, 2] - p1[1, 2]],
                      [u2 * p2[2, 0] - p2[0, 0], u2 * p2[2, 1] - p2[0, 1], u2 * p2[2, 2] - p2[0, 2]],
                      [v2 * p2[2, 0] - p2[1, 0], v2 * p2[2, 1] - p2[1, 1], v2 * p2[2, 2] - p2[1, 2]]])

        b = np.array([-(u1 * p1[2, 3] - p1[0, 3]),
                      -(v1 * p1[2, 3] - p1[1, 3]),
                      -(u2 * p2[2, 3] - p2[0, 3]),
                      -(v2 * p2[2, 3] - p2[1, 3])])

        normal_3d_x = np.array([(np.linalg.lstsq(a, b, rcond=None)[0])])
        normal_3d_x = normal_3d_x
        print('Method III - Normal equation: {}'.format(normal_3d_x))

        # opencv
        import cv2
        x1 = np.array([[u1, v1]])
        x2 = np.array([[u2, v2]])
        opencv_3d_x = cv2.triangulatePoints(p1, p2, x1.T, x2.T)
        opencv_3d_x /= opencv_3d_x[3]
        opencv_3d_x = opencv_3d_x.T
        print('Method IV - OpenCv: {}'.format(opencv_3d_x[0]))


########################################################################################################################


if __name__ == '__main__':
    """Run the TriangulationProcessor test"""

    print('=== Start TriangulationProcessor Unit Test ===')

    # print("    --- Starts Linear triangulation vs OpenCV ---")

    # The test sample for testing linear triangulation is from
    # https://gist.github.com/davegreenwood/e1d2227d08e24cc4e353d95d0c18c914
    #

    # Three projection matrices
    p1_ = np.array([[5.010e+03, 0.000e+00, 3.600e+02, 0.000e+00],
                    [0.000e+00, 5.010e+03, 6.400e+02, 0.000e+00],
                    [0.000e+00, 0.000e+00, 1.000e+00, 0.000e+00]])

    p2_ = np.array([[5.037e+03, -9.611e+01, -1.756e+03, 4.284e+03],
                    [2.148e+02, 5.354e+03, 1.918e+02, 8.945e+02],
                    [3.925e-01, 7.092e-02, 9.169e-01, 4.930e-01]])

    p3_ = np.array([[5.217e+03, 2.246e+02, 2.366e+03, -3.799e+03],
                    [-5.734e+02, 5.669e+03, 8.233e+02, -2.567e+02],
                    [-3.522e-01, -5.839e-02, 9.340e-01, 6.459e-01]])

    # Three image points from each projection above
    x1h = np.array([[274.128, 624.409, 1.0]])
    x1h = x1h.T
    x2h = np.array([[239.571, 533.568, 1.0]])
    x2h = x2h.T
    x3h = np.array([[297.574, 549.260, 1.0]])
    x3h = x3h.T

    tp = TriangulationProcessor()
    projections_ = [p1_, p2_]
    matched_pair_ = []

    x1h_ = KeyPt(1)
    x2h_ = KeyPt(1)
    x1h_[:, :] = x1h[:, :]
    x2h_[:, :] = x2h[:, :]
    matched_pair_.append(x1h_)
    matched_pair_.append(x2h_)
    linear_3d_pts = tp.linear_triangulate(projections_, matched_pair_)
    print('    --- Linear result on two views : \n{}'.format(linear_3d_pts.T))

    import cv2

    u1_ = x1h[0, 0]
    v1_ = x1h[1, 0]
    u2_ = x2h[0, 0]
    v2_ = x2h[1, 0]
    x1_ = np.array([[u1_, v1_]])
    x2_ = np.array([[u2_, v2_]])
    opencv_3d_pt = cv2.triangulatePoints(p1_, p2_, x1_.T, x2_.T)
    opencv_3d_pt /= opencv_3d_pt[3]
    print('    --- OpenCV result on two views : \n{}'.format(opencv_3d_pt.T))

    import math

    diff = math.sqrt(np.square(linear_3d_pts- opencv_3d_pt).sum())
    if diff < 1e-10:
        print("    --- Linear triangulation comparing to OpenCV passes : diff {:.6f}".format(diff))
    else:
        import sys
        print("    --- Linear triangulation comparing to OpenCV fails : diff {:.6f}".format(diff))
        sys.exit(-1)

    DAMPING_FACTOR = 0.5
    ITERATION = 300
    whole_3d_pt = tp.triangulate(projections_, matched_pair_, DAMPING_FACTOR, ITERATION)
    print('    --- Whole triangulation (linear + nonlinear) result on two views : \n{}'.format(whole_3d_pt.T))

    # Compare to scipy.optimize (assume this is the ground truth)
    import scipy.optimize as opt

    def proj_error(init_3d_pt, matched_pts_sample, projections):

        init_3d_pt = np.hstack((init_3d_pt, np.ones(1)))
        error = 0.0
        for i in range(len(matched_pts_sample)):
            proj = projections[i] @ init_3d_pt
            proj = proj / proj[2]  # normalization

            err_u = proj[0] - matched_pts_sample[i][0]
            err_v = proj[1] - matched_pts_sample[i][1]

            error += (error + err_u * err_u + err_v * err_v)

        return error


    linear_3d_pts = linear_3d_pts[0:3, :]
    linear_3d_pts = np.reshape(linear_3d_pts, (-1, 3))

    optimized_params = opt.least_squares(fun=proj_error,
                                         x0=linear_3d_pts[0],
                                         method='trf',
                                         args=[matched_pair_,
                                               projections_])
    scipy_3d_pt = optimized_params.x

    print('    --- Triangulation scipy result on two views : \n{}'.format(np.append(scipy_3d_pt, 1)))

    whole_vs_sci = math.sqrt(np.square(whole_3d_pt[0:3, 0] - scipy_3d_pt).sum())
    linear_vs_sci = math.sqrt(np.square(linear_3d_pts - scipy_3d_pt).sum())

    print('    --- Linear_vs_Sci : {:.6f} - Whole_vs_Sci : {:.6f}'.format(linear_vs_sci, whole_vs_sci))
    if whole_vs_sci <= linear_vs_sci:
        print('    --- Whole result is better than linear one')
    else:
        print('    --- Whole result is worse than linear one')
        exit()

    if (whole_vs_sci <= linear_vs_sci) and (whole_vs_sci <= 1e-2):
        print("    --- Whole triangulation passes : {:.6f}".format(whole_vs_sci))
    else:
        import sys
        print("    --- Whole triangulation fails : {:.6f}".format(whole_vs_sci))
        sys.exit(-1)

    ### 3 points test ###
    '''
    projections_ = [p1_, p2_, p3_]
    matched_pair_ = [x1h, x2h, x3h]
    DAMPING_FACTOR = 0.5
    ITERATION = 300

    linear_3d_pt = tp.linear_triangulate(projections_, matched_pair_)
    whole_3d_pt = tp.triangulate(projections_, matched_pair_, DAMPING_FACTOR, ITERATION)
    print('    --- Linear triangulation result on three views : \n{}'.format(linear_3d_pt[0].T))
    print('    --- Whole triangulation (linear + nonlinear) result on three views : \n{}'.format(whole_3d_pt[0].T))

    # Compare to scipy.optimize (assume this is the ground truth)
    import scipy.optimize as opt

    def proj_error(init_3d_pt, matched_pts_sample, projections):

        init_3d_pt = np.hstack((init_3d_pt, np.ones(1)))
        error = 0.0
        for i in range(len(matched_pts_sample)):
            proj = projections[i] @ init_3d_pt
            proj = proj / proj[2]  # normalization

            err_u = proj[0] - matched_pts_sample[i][0]
            err_v = proj[1] - matched_pts_sample[i][1]

            error += (error + err_u * err_u + err_v * err_v)

        return error


    linear_3d_pt = linear_3d_pt[0][0:3, :]
    linear_3d_pt = np.reshape(linear_3d_pt, (-1, 3))

    optimized_params = opt.least_squares(fun=proj_error,
                                         x0=linear_3d_pt[0],
                                         method='trf',
                                         args=[matched_pair_,
                                               projections_])
    scipy_3d_pt = optimized_params.x

    print('    --- Triangulation scipy result on three views : \n{}'.format(np.append(scipy_3d_pt, 1)))

    whole_vs_sci = math.sqrt(np.square(whole_3d_pt[0][0:3, 0] - scipy_3d_pt).sum())
    linear_vs_sci = math.sqrt(np.square(linear_3d_pt - scipy_3d_pt).sum())

    print('    --- Linear_vs_Sci {:.6f} - Whole_vs_Sci {:.6f}'.format(linear_vs_sci, whole_vs_sci))
    if whole_vs_sci <= linear_vs_sci:
        print('    --- Whole result is better than linear one')
    else:
        print('    --- Whole result is worse than linear one')

    if (whole_vs_sci <= linear_vs_sci) and (whole_vs_sci <= 1e-2):
        print("    --- Whole triangulation passes : {:.6f}".format(whole_vs_sci))
    else:
        import sys
        print("    --- Whole triangulation fails : {:.6f}".format(whole_vs_sci))
        sys.exit(-1)

    # print("    --- Complete Whole triangulation (linear + nonlinear) vs Scipy.optimize ---")
    '''

    print('=== Complete TriangulationProcessor Unit Test ===')
