#
# this script is for debugging the pose estimation model
#
#  yaz ortasu
"""
the main file for 3D pose estimation (full & one time training)
"""
#
# import __init__
#
from lifting import PoseEstimator
# from src.utils import smoothing_prediction
# from src.utils import stats
from lifting.utils import draw_limbs
from lifting.utils import plot_pose
# from src.utils.metric_comparison import get_statistics #to be completed
#
import pickle
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from os.path import dirname, realpath
import argparse
from PIL import Image
#
args = argparse.ArgumentParser()
args.add_argument('-d0', '--DIR_PATH', default=dirname(realpath(__file__)), help='current working directory path')
args.add_argument('-d2', '--SAVED_SESSIONS_DIR', default="/trained_models", help='path to Video directory')
args.add_argument('-d3', '--SESSION_PATH', default="/init_session/init", help='path to Video directory')
args.add_argument('-d4', '--PROB_MODEL_PATH', default="/prob_model/prob_model_params.mat", help='path to Video directory')
args.add_argument('-d5', '--VIDEO_PATH', default="/home/ali/CLionProjects/PoseEstimation/Golf_3D_Pose_my_impl_/input_video/", help='path to Video directory')
args.add_argument('-d5_1', '--RESULTS', default="/home/ali/CLionProjects/PoseEstimation/full_flow/results/", help='path to Video directory')
args.add_argument('-d6_1', '--VIDEO_NAME_1', default="20190716_151154_-0500.mp4", help='path to Video directory')
args.add_argument('-d6_2', '--VIDEO_NAME_2', default="20190716_151154_-0500.mp4", help='path to Video directory')
args.add_argument('-d6_3', '--Vid_dir_1', default="golf_videos_kyle/", help='path to Video directory')
args.add_argument('-d6_4', '--Vid_dir_2', default="selected_01_vids/", help='path to Video directory')
args.add_argument('-d7', '--start', default= 10, help='start frame')
args.add_argument('-d8', '--end', default=100, help='last frame')
args.add_argument('-d9', '--output', default="golf_", help='output name')
args.add_argument('-d10', '--viz', default=False, help='wethere to visualize')
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
        flow.cap.set(cv2.CAP_PROP_FPS, 30)
        flow.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        flow.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Start getting from an specific frame
        flow.cap.set(cv2.CAP_PROP_POS_FRAMES, ap.start)

    # function for flowing data from video
    def flow_from_video(flow):
        ret = True
        while ret:
            ret, frame = flow.cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # in case of needing to rotate the image
            # pil_img = Image.fromarray( image ).rotate(90)
            # image = np.array(pil_img)
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

def display_results(in_image, data_2d, joint_visibility, data_3d, i):
    """
    Plot 2D and 3D poses for each of the people in the image;
    :param in_image:
    :param data_2d:
    :param joint_visibility:
    :param data_3d:
    :param i:
    :return:
    """

    plt.figure(i)
    draw_limbs(in_image, data_2d, joint_visibility, i)
    plt.imshow(in_image)
    # plt.axis('off')

    # Show 3D poses
    for single_3D in data_3d:
        # or plot_pose(Prob3dPose.centre_all(single_3D))
        plot_pose(single_3D)
    plt.show()




#
def main():
    # class definition
    # cmpr_results = get_statistics()
    # data flower
    # NOTE: to read data from camera set data flow input to "0"
    dflower_1 = data_flow(ap.VIDEO_PATH + ap.Vid_dir_1 +  ap.VIDEO_NAME_1)
    # dflower_2 = data_flow(ap.VIDEO_PATH + ap.Vid_dir_2 +  ap.VIDEO_NAME_2)
    images_1 = dflower_1.flow_from_video() # comparing two different results
    # images_2 = dflower_2.flow_from_video()
    # IMAGE_FILE_PATH = '/home/ali/CLionProjects/PoseEstimation/full_flow/data/images/'
    # img_list = glob.glob(IMAGE_FILE_PATH + '*.jpg')

    # Model setting
    std_shape = (1280, 720, 3)
    pose_estimator = PoseEstimator(std_shape, PROJECT_PATH + ap.SAVED_SESSIONS_DIR + ap.SESSION_PATH, PROJECT_PATH + ap.SAVED_SESSIONS_DIR + ap.PROB_MODEL_PATH)

    # load model
    pose_estimator.initialise()

    # loading postprocessors
    # ppop = smoothing_prediction(ap.start, ap.end)
    # stdsp = stats(ap.VIDEO_NAME_1, ap.start, ap.end)

    # flowing data
    ret = True
    ii = 0
    num_keypoints = 17
    pose_2d_s = []
    pose_3d_s = []
    visibility_s = []
    image_s = []
    #
    while ret:
        # create pose estimator
        try:
            while ii < ap.start:
                ii+=1
                next( images_1 )
                # next( images_2 )
                continue
            image_1 = cv2.resize( next( images_1 ), (std_shape[1], std_shape[0]) )
            # image_2 = cv2.resize( next( images_2 ), (std_shape[1], std_shape[0]) )
            ii+=1
        except:
            ret = False
        # for item in  img_list:
        # image = cv2.cvtColor( cv2.resize( cv2.imread( item ), (std_shape[1], std_shape[0]) ) , cv2.COLOR_BGR2RGB )
        # image_size = image_1.shape # (720, 1280, 3)

        # estimation
        try:
            if ii % 2 == 0:
                continue
            pose_2d_1, visibility_1, pose_3d_1 = pose_estimator.estimate(image_1)
            # pose_2d_2, visibility_2, pose_3d_2 = pose_estimator.estimate(image_2)
            #
            pose_2d_s.append(pose_2d_1[0])
            pose_3d_s.append(pose_3d_1[0])
            visibility_s.append(visibility_1)
            image_s.append(image_1)
            # writing images
            # img = Image.fromarray( image_1 )
            # img.save(ap.RESULTS + 'imagespng/' + 'img_frame_' + str(ii) + '.png')
            print('frame number: ', ii)
            #
        except:
            print('No player detected yet... ')
            continue
        # Single 2D and 3D poses representation

        print('Processing the frame {0}'.format(ii))
        if ii> ap.end:
            break
        if ap.viz:
            display_results(image_1, pose_2d_1, visibility_1, pose_3d_1, ii)

        # collecting data
        # cmpr_results.get_result( pose_3d_1[0], pose_3d_2[0] )
            # display_results(image_2, pose_2d_2, visibility_2, pose_3d_2, ii)
            # fig, fig_2 = comparing_result( pose_3d_1[0], pose_3d_2[0] )
            # fig.show()
            # fig_2.show()
        # working on 2D and 3D correspondness
        # postprocessing the poses
        # getting the data
        # stdsp.get_data()


    # optimized_pose = []
    # posposj = [pose_3d_s[i][:,13] for i in range(len(pose_3d_s))]
    # ppop.optimize_pose(13, posposj)
    # for j in range(num_keypoints):
    #     posposj = [pose_3d_s[i][:,j] for i in range(len(pose_3d_s))]
    #     optimized_pose.append( ppop.optimize_pose(j, posposj) )

    # ppop.do_post_process(optimized_pose)
    # stdsp.update_state(optimized_pose)
    # stdsp.save_stats()


    # saving images and 2d poses
    with open(ap.RESULTS + 'pickles/' + ap.VIDEO_NAME_1.split('.')[0] + '_pose2Ds.pickle'  , 'wb') as f:
        pickle.dump(pose_2d_s, f)
    with open(ap.RESULTS + 'pickles/' + ap.VIDEO_NAME_1.split('.')[0] + '_pose3Ds.pickle'  , 'wb') as f:
        pickle.dump(pose_3d_s, f)
    with open(ap.RESULTS + 'pickles/' + ap.VIDEO_NAME_1.split('.')[0] + '_image_s.pickle'  , 'wb') as f:
        pickle.dump(image_s, f)
    #
    # optimized_pose = np.array(optimized_pose).transpose(2,0,1)

    # plotting the smoothed poses ...
    # plotting frame by frame
    #path_to_ref_pos = ['golf_03_best_pos.pickle', 'golf_04_best_pos.pickle']
    # stdsp.load_opt_stat(path_to_ref_pos)
    # stdsp.comparing_poses(optimized_pose)
    #indx = []
    # stdsp.draw_3D_smooth( optimized_pose, pose_2d_s, image_s )

    # close model
    pose_estimator.close()
    # cmpr_results.disply_results()

    # with open('data/pose_2d_' + ap.output +  ap.VIDEO_NAME.split('.')[0] + '.pickle', 'wb') as f:
    #     pickle.dump(pose_2d_s, f)
    # with open('data/pose_3d_' + ap.output + ap.VIDEO_NAME.split('.')[0] + '.pickle', 'wb') as f:
    #     pickle.dump(pose_3d_s, f)
    # with open('data/visibility_' + ap.output + ap.VIDEO_NAME.split('.')[0] + '.pickle', 'wb') as f:
    #     pickle.dump(visibility_s, f)
    # with open('data/frame_' + ap.output + ap.VIDEO_NAME.split('.')[0] + '.pickle', 'wb') as f:
    #     pickle.dump(image_s, f)

    # Show 2D and 3D poses
    # display_results(image, pose_2d, visibility, pose_3d )

if __name__ == '__main__':
    # importing the compare metrics

    #
    main()


