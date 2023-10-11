import os
import sys
import pickle
import cv2 as cv
import numpy as np
import logging

IMG_EXT = ('.png', 'jpg', 'bmp')
VAL_DIFF_THRESHOLD = 1e-5

########################################################################################################################


class View:
    """A struct used to represent the view

    Attributes
    ----------
    img : numpy.ndarray
        The 2D image data
    idx : int
        The view index in a sequence of images
    key_pts : cv2.keypoint
        The key points in the image
    key_descriptors : cv2.descriptor
        The key_descriptors associated with the key points
    rot : numpy.ndarray
        The 3x3 rot matrix from the reference view
    loc: numpy.ndarray
        The 3x1 loc vector from the reference view
    """

    def __init__(self, img, idx, k, key_pts, key_descriptors):
        """Initialization

        Parameters
        ----------
        img : numpy.ndarray
            The 2D image data
        idx : int
            The view index in a sequence of images
        key_pts : list of cv2.keypoint
            The key points in the image
        k: numpy.ndarray
            The 3x3 intrinsic matrix
        """

        self.img = img
        self.idx = idx
        self.ref_idx = idx
        self.key_pts = key_pts
        self.key_descriptors = key_descriptors
        self.rot = np.identity(3, dtype=float)
        self.loc = np.zeros((3, 1), dtype=float)
        self.k = k
        self.cam_pose = np.hstack((self.rot, self.loc))
        self.cam_proj = self.k @ np.hstack((self.rot.T, self.rot.T @ -self.loc))
        self.is_valid = False

    #########################################################################
    def update_cam_pose(self, rot, loc):
        """
        Update camera pose
        """

        self.rot = rot
        self.loc = loc
        self.cam_pose = np.hstack((self.rot, self.loc))
        self.cam_proj = self.k @ np.hstack((self.rot.T, self.rot.T @ -self.loc))

    #########################################################################
    def update_intrinsic_mat(self, k):
        """
        Update intrinsic matrix
        """

        self.k = k
        self.cam_proj = self.k @ np.hstack((self.rot.T, self.rot.T @ -self.loc))

    #########################################################################
    def write_keys(self, key_file_path):
        """ Stores computed keys to pkl files. The files are written inside a keys directory inside the root
        directory """

        if key_file_path[-4:] != '.pkl':
            logging.error('%s : incorrect key file path : %s', self.__class__.__name__,
                          key_file_path)
            return

        temp_array = []
        for idx, point in enumerate(self.key_pts):
            temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id,
                    self.key_descriptors[idx])
            temp_array.append(temp)

        keys_file = open(key_file_path, 'wb')
        pickle.dump(temp_array, keys_file)
        keys_file.close()

########################################################################################################################


class ViewProcessor:
    """A class used to generate views

    Attributes
    ----------
    key_type : str
        The type of key detector
    detector : cv2.detector
        The key_type key detector

    Methods
    -------
    generate_view(image_path, idx, key_path=None)
        Load the image (required), assign the view index, and load keys (if provided)
    """

    def __init__(self, key_type='sift'):
        """Initialization

        Parameters
        ----------
        key_type : str
            The type of key detector
        """

        if key_type == 'sift':
            self.detector = cv.SIFT_create()  # cv.xkeys2d.SIFT_create()
        elif key_type == 'orb':
            self.detector = cv.ORB_create(nkeys=1500)
        # elif key_type == 'surf':
        #    self.detector = cv.xkeys2d.SURF_create()
        else:
            logging.error('%s : admitted key types are SIFT or ORB',
                          self.__class__.__name__)
            sys.exit(0)

        self.key_type = key_type
        self.view_list = []

    #########################################################################
    def add_view(self, view):
        """
        Add a new view
        """
        
        self.view_list.append(view)

    #########################################################################
    def generate_view(self, img, index, k, key_path=None):
        """Generate the view based on the inputs

        Parameters

        """
        # key_pts = []
        # key_descriptors = []
        if not key_path:
            key_pts, key_descriptors = self.__extract_keys(img)
        else:
            key_pts, key_descriptors = self.__read_keys(key_path, img)

        return View(img, index, k, key_pts, key_descriptors)

    #########################################################################
    def __read_keys(self, key_path, img):
        """ Reads keys stored in files. Key files have filenames corresponding to image names without
        extensions """

        # logic to compute keys for images that don't have pkl files
        try:
            if key_path[-4:] != '.pkl':
                logging.error('%s : incorrect key file path : %s', self.__class__.__name__,
                              key_path)
                return

            keys = pickle.load(open(key_path, 'rb'))
            key_pts = []
            key_descriptors = []

            for point in keys:
                keypoint = cv.KeyPoint(x=point[0][0], y=point[0][1], size=point[1], angle=point[2],
                                       response=point[3], octave=point[4], class_id=point[5])
                descriptor = point[6]
                key_pts.append(keypoint)
                key_descriptors.append(descriptor)

            # convert key_descriptors into n x 128 numpy array
            key_descriptors = np.array(key_descriptors)
            return key_pts, key_descriptors

        except FileNotFoundError:
            logging.error('%s : pkl file %s not found ', key_path,
                          self.__class__.__name__)
            return self.__extract_keys(img)

    #########################################################################
    def __extract_keys(self, img):
        """ Extracts keys from the image """
        key_pts, key_descriptors = self.detector.detectAndCompute(img, None)
        return key_pts, key_descriptors


########################################################################################################################

if __name__ == '__main__':
    """ Run the ViewProcessor test """

    print('=== Start ViewProcessor Unit Test ===')
    file_path = os.path.dirname(__file__)  # os.getcwd()
    test_dataset_path = os.path.join(file_path, 'test_dataset', 'upenn')

    # Test key detector creation
    # vp = ViewProcessor('sift')
    # vp = ViewProcessor('orb')
    # vp = ViewProcessor('surf') comment out because SURF is patented

    # Use SIFT detector to test the following test
    vp = ViewProcessor('sift')
    idx_ = 0
    K = np.array([[568.996140852, 0, 643.21055941],
                  [0, 568.988362396, 477.982801038],
                  [0, 0, 1]])
    for file_name in os.listdir(test_dataset_path):
        if file_name.endswith(IMG_EXT):
            img_path = os.path.join(test_dataset_path, file_name)
            img_ = cv.imread(img_path)

            # Generate the view
            ref_view = vp.generate_view(img_, idx_, K)

            # Write the key info
            key_path_ = os.path.join(test_dataset_path, file_name[:-4] + '.pkl')
            ref_view.write_keys(key_path_)

            # Read the written key info
            que_view = vp.generate_view(img_, idx_, K, key_path_)

            if len(ref_view.key_pts) != len(que_view.key_pts):
                logging.error('%s: num of key points are different (%d, %d)', file_name,
                              len(ref_view.key_pts), len(que_view.key_pts))
                sys.exit(-1)

            x_diff = abs(ref_view.key_pts[0].pt[0] - que_view.key_pts[0].pt[0])
            y_diff = abs(ref_view.key_pts[0].pt[1] - que_view.key_pts[0].pt[1])
            if (x_diff >= VAL_DIFF_THRESHOLD) or (y_diff >= VAL_DIFF_THRESHOLD):
                logging.error('%s: x and y are different (%f, %f)', file_name,
                              x_diff, y_diff)
                sys.exit(-1)
            idx_ += 1
            print('    {} passes'.format(file_name))
    print('=== Complete ViewProcessor Unit Test ===')
