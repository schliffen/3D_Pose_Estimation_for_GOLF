#
"""
the main file for 3D pose estimation (full & one time training)
"""
#
# import __init__
#
from src import PoseEstimator
from src.utils import draw_limbs
from src.utils import plot_pose

import cv2
import glob
import matplotlib.pyplot as plt
from os.path import dirname, realpath

import argparse
args = argparse.ArgumentParser()
args.add_argument('-d0', '--DIR_PATH', default=dirname(realpath(__file__)), help='current working directory path')
args.add_argument('-d2', '--SAVED_SESSIONS_DIR', default="/trained_models", help='path to Video directory')
args.add_argument('-d3', '--SESSION_PATH', default="/init_session/init", help='path to Video directory')
args.add_argument('-d4', '--PROB_MODEL_PATH', default="/prob_model/prob_model_params.mat", help='path to Video directory')
args.add_argument('-d5', '--VIDEO_PATH', default="/home/ali/DATA/Videos/v_1/", help='path to Video directory')
args.add_argument('-d6', '--VIDEO_NAME', default="id_304.mov", help='path to Video directory')

ap = args.parse_args()

# DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(ap.DIR_PATH + '/')
#

class data_flow():
    def __init__(flow, path):
        flow.path = path
        flow.cap = cv2.VideoCapture(flow.path)
        if flow.cap.isOpened() is False:
            print('Error opening video stream or file')
            exit(1)
        #
        flow.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        flow.cap.set(cv2.CAP_PROP_FPS, 10)
        flow.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        flow.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Start getting from an specific frame
        flow.cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

    # function for flowing data from video
    def flow_from_video(flow):
        ret = True
        while ret:
            ret, frame = flow.cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield image


    # function for flowing data from folder
    def flow_from_folder(flow):
        img_list = glob.glob(flow.path + '*.jpg')
        #
        for item in img_list:
            image = cv2.cvtColor( cv2.imread(item),  cv2.COLOR_BGR2RGB)
            yield image

    def flow_from_camera(flow):
        ret = True
        while ret:
            ret, frame = flow.cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield image



def main():

    # data flower
    # NOTE: to read data from camera set data flow input to "0"
    dflower = data_flow(ap.VIDEO_PATH + ap.VIDEO_NAME) #
    images = dflower.flow_from_video()#

    # Model settin
    std_shape = (720, 1280, 3)
    pose_estimator = PoseEstimator(std_shape, PROJECT_PATH + ap.SAVED_SESSIONS_DIR + ap.SESSION_PATH, PROJECT_PATH + ap.SAVED_SESSIONS_DIR + ap.PROB_MODEL_PATH)

    # load model
    pose_estimator.initialise()

    # flowing data
    ret = True
    ii = 0
    while ret:
        # create pose estimator
        image = cv2.resize( next( images ), (std_shape[1], std_shape[0]) )

        # image = cv2.cvtColor( cv2.imread( IMAGE_FILE_PATH ), cv2.COLOR_BGR2RGB )

        image_size = image.shape # (720, 1280, 3)

        # estimation
        # in the case of person detection does not work, we can jump to the next frame
        try:
            pose_2d, visibility, pose_3d = pose_estimator.estimate(image)
        except:
            continue
        # Single 2D and 3D poses representation
        ii+=1
        display_results(image, pose_2d, visibility, pose_3d, ii)


    # close model
    pose_estimator.close()

    # Show 2D and 3D poses
    # display_results(image, pose_2d, visibility, pose_3d )


def display_results(in_image, data_2d, joint_visibility, data_3d, i):
    """Plot 2D and 3D poses for each of the people in the image."""
    plt.figure(i)
    draw_limbs(in_image, data_2d, joint_visibility)
    plt.imshow(in_image)
    plt.axis('off')

    # Show 3D poses
    for single_3D in data_3d:
        # or plot_pose(Prob3dPose.centre_all(single_3D))
        plot_pose(single_3D)

    plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(main())
