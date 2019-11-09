#
#
import cv2
import time
import pickle
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from os.path import dirname, realpath
# importing related codes
from src import smoothing_prediction
from numpy.polynomial import legendre as leg
from scipy import signal

import argparse

args = argparse.ArgumentParser()
args.add_argument('-d0', '--root', default=dirname(realpath(__file__)), help='current working directory path')
args.add_argument('-d1', '--data', default='/results/pickles/', help='path to the using data' )
# args.add_argument('-d1', '--', default='', help='' )
# args.add_argument('-d2', '--', default='', help='' )
# args.add_argument('-d3', '--', default='', help='' )
# args.add_argument('-d4', '--', default='', help='' )
# args.add_argument('-d5', '--', default='', help='' )
ap = args.parse_args()


def estimate_start_end(pose_3d):
    #
    time_length = len(pose_3d)
    # decision factors (z position of hands)
    z_axis_lh = [pose_3d[i][2, 13] for i in range(time_length) ]
    z_axis_rh = [pose_3d[i][2, 13] for i in range(time_length) ]
    dz_lh = [a - b for a, b in zip(z_axis_lh[2:], z_axis_lh[0:-2])]
    dz_rh = [a - b for a, b in zip(z_axis_rh[2:], z_axis_rh[0:-2])]
    # rescaling
    init_lh = np.where( np.array(dz_lh) > 100 )[0][0]
    init_rh = np.where( np.array(dz_rh) > 100 )[0][0]
    end_lh = np.argmin(dz_lh)
    end_rh = np.argmin(dz_rh)
    #
    gs_start = int((init_lh + init_rh)/2) - 1
    gs_end = int((end_lh + end_rh)/2) + 3
    return gs_start, gs_end

class estimate_smooth_pose():
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.approx_deg = 4
        self.resamp = 5
        self.num_joints = 17
        self.aspect_ratio = 1.4
    
    def pose_postprocess( self, pose_3d ):
        #
        focus_time = None
        time_length = len(pose_3d)
        t_axis = np.linspace(0, time_length, time_length + 1)
        #
        nsamples = self.end - self.start
        # The independent variable sampled at regular intervals
        xs=np.linspace(-1, 1, nsamples)
        # The Legendre Vandermonde matrix
        V=leg.legvander(xs, self.approx_deg)
        # Generate some data to fit
        ys=xs*np.pi
        # Do the fit for all 17 poses
        resampledpose = {}
        for ii in range(self.num_joints):
            respsoexyz = []
            for ij in range(3):
                jnt_coord = [pose_3d[i][ij, ii] for i in range(time_length)]
                #
                coeffs=np.linalg.lstsq(V, jnt_coord[self.start:self.end], rcond=None)[0]
                # Evaluate the fit for plotting purposes
                g = leg.legval(xs, coeffs)
                focus_val, _time = signal.resample(g, len(g)*self.resamp, t_axis[self.start:self.end])
                focus_val  = list(focus_val)
                if focus_time == None:
                    focus_time = list(_time)
                    for ti in range(self.start):
                        focus_time.insert(0, ti)
                # adding the inactive times
                for ti in range(self.start):
                    focus_val.insert(0, focus_val[0])

                respsoexyz.append(focus_val)

            #
            resampledpose.update( {'joint_' + str(ii): np.array(respsoexyz) } )

        return resampledpose, focus_time

    def get_3d_line(self, p0, p1, z):
        #
        m = p1 - p0
        x = [(z[j]-p0[2][j])*m[0][j]/m[2][j] + p0[0][j] for j in range(z.shape[0])]
        y = [(z[j]-p0[2][j])*m[1][j]/m[2][j] + p0[1][j] for j in range(z.shape[0])]

        return [x,y]

    def joint_constrain(self, pose_dict):
        ## IMPORTANT PART <correcting joints>
        num_frames = pose_dict['joint_13'].shape[1]
        # Important part
        pose_dict['joint_13'] = (pose_dict['joint_13'] + pose_dict['joint_16'])/2
        pose_dict['joint_16'] = (pose_dict['joint_13'] + pose_dict['joint_16'])/2
        #
        #
        for i in range(3):
            pose_dict['joint_3'][i] = [pose_dict['joint_3'][i].mean()] * num_frames
            pose_dict['joint_6'][i] = [pose_dict['joint_6'][i].mean()] * num_frames
            # to take these with care
            if i<2:
                # stomach
                pose_dict['joint_0'][i] = [ (pose_dict['joint_0'][i][j]/3 + pose_dict['joint_1'][i][j]/3 + pose_dict['joint_4'][i][j]/3)
                                            for j in range(num_frames) ]
                # knees
                pose_dict['joint_2'][i] = [ (pose_dict['joint_2'][i][j]/3 + pose_dict['joint_3'][i][j]/3 + pose_dict['joint_14'][i][j]/3)
                                             for j in range(num_frames) ]
                pose_dict['joint_5'][i] = [ (pose_dict['joint_5'][i][j]/3 + pose_dict['joint_6'][i][j]/3 + pose_dict['joint_11'][i][j]/3)
                                            for j in range(num_frames) ]
                # pevliks
                pose_dict['joint_1'][i] = [ (pose_dict['joint_1'][i][j]/3 + pose_dict['joint_3'][i][j]/3 + pose_dict['joint_14'][i][j]/3)
                                            for j in range(num_frames) ]
                pose_dict['joint_4'][i] = [ (pose_dict['joint_4'][i][j]/3 + pose_dict['joint_6'][i][j]/3 + pose_dict['joint_11'][i][j]/3)
                                            for j in range(num_frames) ]
                # elbows
                # elbow_reg = self.get_3d_line(pose_dict['joint_11'], pose_dict['joint_13'], pose_dict['joint_12'][2])
                # + elbow_reg[i][j] * abs(j- num_frames/2)/num_frames
                pose_dict['joint_12'][i] = [ (pose_dict['joint_12'][i][j] + pose_dict['joint_15'][i][j]/2 + pose_dict['joint_4'][i][j]/3 )
                                            for j in range(num_frames) ]
                # elbow_reg = self.get_3d_line(pose_dict['joint_14'], pose_dict['joint_16'], pose_dict['joint_15'][2])
                # + elbow_reg[i][j] * abs(j- num_frames/2)/num_frames
                pose_dict['joint_15'][i] = [ (pose_dict['joint_15'][i][j] + pose_dict['joint_12'][i][j]/2 + pose_dict['joint_1'][i][j]/3 )
                                            for j in range(num_frames) ]



        # ration and height standardization
        zero_height = min( pose_dict['joint_3'][2,0], pose_dict['joint_6'][2,0])
        for item in pose_dict.keys():
            pose_dict[item][2,:] -= zero_height
            pose_dict[item][2,:] *= self.aspect_ratio
        joints_3d = np.array([[pose_dict['joint_' + str(j)][:,i] for j in range(len(pose_dict.keys()))] \
                              for i in range(num_frames)]).transpose(0, 2, 1)
        return joints_3d


class joint_spect():
    def __init__(self):
        self.pad = (243 - 1) // 2 # Padding on each side
        self.causal_shift = 0
        self.num_kpoints = 17
        self.keypoints_symmetry = [[1, 3, 5, 7, 9, 11, 13, 15],[2, 4, 6, 8, 10, 12, 14, 16]]
        self.rot = np.array([ 0.14070565, -0.15007018, -0.7552408 ,  0.62232804], dtype=np.float32)
        self.skeleton_parents =  np.array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15])
        self.pairs = [(1,2), (5,4), (6,5), (8,7), (8,9), (10,1),\
                      (11,10), (12,11), (13,1), (14,13), (15,14),\
                      (16,2), (16,3), (16,4), (16,7)]
        self.kps_left, kps_right = list(self.keypoints_symmetry[0]), list(self.keypoints_symmetry[1])
        self.joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])
        self.joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
                            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                            [5, 11], [6, 12], [11, 12],
                            [11, 13], [12, 14], [13, 15], [14, 16]]
    #
    def joint_name_3d(self, joint):
        if joint == 0:   return 'stomach'
        elif joint == 1: return 'right_pelvic'
        elif joint == 2: return 'right_knee'
        elif joint == 3: return 'right_heel'
        elif joint == 4: return 'left_pelvic'
        elif joint == 6: return 'left_knee'
        elif joint == 7: return 'left_heel'
        elif joint == 8: return 'neck'
        elif joint == 9: return 'jaw'
        elif joint == 10:return 'head'
        elif joint == 11:return 'left_shoulder'
        elif joint == 12:return 'left_elbow'
        elif joint == 13:return 'left_hand'
        elif joint == 14:return 'right_shoulder'
        elif joint == 15:return 'right_elbow'
        elif joint == 16:return 'right_hand'
    #
    def joint_name_2d(self, joint):
        if joint == 'head':  return 1, # 2, 3, 4
        elif joint == 'right_pelvic': return 11
        elif joint == 'left_pelvic': return 12
        elif joint == 'right_knee': return 13
        elif joint == 'left_knee': return 14
        elif joint == 'right_heel': return 15
        elif joint == 'left_heel': return 16
        elif joint == 'right_shoulder': return 5
        elif joint == 'left_shoulder': return 6
        elif joint == 'right_elbow': return 7
        elif joint == 'leftt_elbow': return 8
        elif joint == 'jaw': return 0
        elif joint == 'right_hand': return 9
        elif joint == 'left_hand': return 10

    def joint_color(self, j):
        """
        TODO: 'j' shadows name 'j' from outer scope
        """
        colors = [
            (10, 250, 255), (250, 10, 0), (0, 250, 10),
            (5, 10, 10), (255, 5, 250), (5, 10, 250),]
        _c = 0
        if j in range(1, 4):  _c = 1
        if j in range(4, 7):  _c = 2
        if j in range(9, 11): _c = 3
        if j in range(11, 14):_c = 4
        if j in range(14, 17):_c = 5
        return colors[_c]

class visualization(joint_spect):
    def __init__(self, video_name, series = '07'):
        self.video_name = video_name
        self.save_dir = '/home/ali/CLionProjects/PoseEstimation/Golf_3D_Pose_my_impl_/results/kyle/'
        self.NORMALISATION_COEFFICIENT = 1280*720
        self.JOINT_DRAW_SIZE = 3
        self.LIMB_DRAW_SIZE = 2
        self._COLORS = [
            [0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0],
            [170, 255, 0], [255, 170, 0], [255, 0, 0], [255, 0, 170],
            [170, 0, 255]
        ]
        self._LIMBS = np.array([0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9,
                                9, 10, 11, 12, 12, 13]).reshape((-1, 2))

        self.init_pose = []
        self.historic_data = {}



    def joint_color(self, j):
        """
        TODO: 'j' shadows name 'j' from outer scope
        """
        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255),
                  (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if j in range(1, 4):
            _c = 1
        if j in range(4, 7):
            _c = 2
        if j in range(9, 11):
            _c = 3
        if j in range(11, 14):
            _c = 4
        if j in range(14, 17):
            _c = 5
        return colors[_c]

    def vis_2d(self, image, pose_2d):
        #
        _NORMALISATION_FACTOR = int(math.floor(math.sqrt(image.shape[0] * image.shape[1] / self.NORMALISATION_COEFFICIENT)))

        # for oid in range(pose_2d.shape[0]):
        for lid, (p0, p1) in enumerate(self._LIMBS):
            # if not (visible[oid][p0] and visible[oid][p1]):
            #     continue
            y0, x0 = pose_2d[p0]
            y1, x1 = pose_2d[p1]
            cv2.circle(image, (x0, y0), self.JOINT_DRAW_SIZE * _NORMALISATION_FACTOR , self._COLORS[lid], -1)
            cv2.circle(image, (x1, y1), self.JOINT_DRAW_SIZE * _NORMALISATION_FACTOR , self._COLORS[lid], -1)
            cv2.line(image, (x0, y0), (x1, y1),
                     self._COLORS[lid], self.LIMB_DRAW_SIZE * _NORMALISATION_FACTOR , 16)
            # time.sleep(5)

        return image

    def translate_visibility(self, jname):
        if jname == 'Hands':
            return [13, 16]

        return -1

    def vis_3d(self, pose_3d, ax, visible = None, cfrm = 0, keep_tr = False):
        import mpl_toolkits.mplot3d.axes3d as p3

        _CONNECTION = [
            [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
            [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15],
            [15, 16]]

        assert (pose_3d.ndim == 2)
        assert (pose_3d.shape[0] == 3)

        # if pose_3d.max() > 2500 or pose_3d.min() < -2000:
        #     return None

        # translating names to joints for illustration
        # visible = self.translate_visibility( visible )
        # preparing the ax in a proper way
        # computing the radious
        # radius = 100
        # #
        # # ax.view_init(elev=20., azim=-60.)
        # ax.set_xlabel('length')
        # ax.set_xlim3d([-radius/2, radius/2])
        # ax.set_zlabel('height')
        # ax.set_zlim3d([0, 16*radius])
        # ax.set_ylabel('width')
        # ax.set_ylim3d([-radius/2, radius/2])
        # ax.set_aspect('equal')
        # #
        # ax.set_xticklabels(list(np.array([-1., -.8 , -.6, -.4, -.2, 0.0, .2, .4, .6, .8, 1.])*radius))
        # ax.set_yticklabels(list(np.array([-1., -.8 , -.6, -.4, -.2, 0.0, .2, .4, .6, .8, 1.])*radius))
        # ax.set_zticklabels(list(np.array([0.0, .2, .4, .6, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.0])*radius*2))
        ax.dist = 6.5 # 7.5

        # modifying the representation joints
        if visible == None or visible == []:
            print('There is noting to show - select joints to show or set signal = [-1] to show all!')
            return -1

        if (cfrm == 0 and visible[0] != -1):
                self.historic_data = {}
                self.init_pose = pose_3d

        if visible[0] != -1:
            for item in visible:
                self.init_pose[:, item] = pose_3d[:, item]
                if keep_tr:
                    try:
                        self.historic_data[str(item)].append(pose_3d[:, item])
                    except:
                        self.historic_data.update({str(item):[pose_3d[:, item]]})

            pose_3d = self.init_pose


        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        for c in _CONNECTION:
            # if not (c[0] in visible or c[1] in visible):
            #     continue
            col = '#%02x%02x%02x' % self.joint_color(c[0])
            ax.plot([pose_3d[0, c[0]], pose_3d[0, c[1]]],
                    [pose_3d[1, c[0]], pose_3d[1, c[1]]],
                    [pose_3d[2, c[0]], pose_3d[2, c[1]]], c=col)

        for j in range(pose_3d.shape[1]):
            # if not j in visible:
            #     continue
            col = '#%02x%02x%02x' % self.joint_color(j)
            ax.scatter(pose_3d[0, j], pose_3d[1, j], pose_3d[2, j],
                       c=col, marker='o', edgecolor=col)

        # plotting previous points if it was selected
        if keep_tr:
            for i1 in list(self.historic_data.keys()):
                for i2 in self.historic_data[i1]:
                    ax.scatter(i2[0], i2[1], i2[2],
                       c=col, marker='>', edgecolor='#%02x%02x%02x' % self.joint_color(int(i1)))

        # smallest = pose_3d.min()
        # largest = pose_3d.max()
        # ax.set_xlim3d(smallest, largest)
        # ax.set_ylim3d(smallest, largest)
        # ax.set_zlim3d(smallest, largest)
        ax.set_xlim3d(-1000, 1000)
        ax.set_ylim3d(-1000, 1000)
        ax.set_zlim3d(0, 3000)

        return ax



if __name__ == '__main__':
    # importing pickled data
    # from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import mpl_toolkits.mplot3d.axes3d as p3

    #
    data_address_1 = {'images': 'golf_01_image_s.pickle', 'pose_2ds':'golf_01_pose2Ds.pickle', 'pose_3ds': 'golf_01_pose3Ds.pickle'}
    data_address_2 = {'images': 'golf_02_image_s.pickle', 'pose_2ds':'golf_02_pose2Ds.pickle', 'pose_3ds': 'golf_02_pose3Ds.pickle'}
    data_address_3 = {'images': 'golf_03_image_s.pickle', 'pose_2ds':'golf_03_pose2Ds.pickle', 'pose_3ds': 'golf_03_pose3Ds.pickle'}
    data_address_4 = {'images': 'golf_04_image_s.pickle', 'pose_2ds':'golf_04_pose2Ds.pickle', 'pose_3ds': 'golf_04_pose3Ds.pickle'}
    data_address = [data_address_1, data_address_2, data_address_3, data_address_4]

    with open( ap.root + ap.data + data_address[0]['pose_3ds'], 'rb' ) as f:
        pose_3d = pickle.load(f) #
    with open( ap.root + ap.data + data_address[0]['pose_2ds'], 'rb' ) as f:
        pose_2d = pickle.load(f) # frm number x joints x coordinates
    with open( ap.root + ap.data + data_address[0]['images'], 'rb' ) as f:
        images = pickle.load(f) # frm number x image coord
    #
    gs_start, gs_end = estimate_start_end(pose_3d)
    posesmoothing = estimate_smooth_pose(gs_start, gs_end)
    #
    processed_pose_3d, focus_time = posesmoothing.pose_postprocess( pose_3d )
    # processed_pose_3d # dictionary: joint name : coords x num frame
    # hard part: visulization
    #
    print('starting visualization!')
    jntspt = joint_spect()

    vis = visualization('test_video')
    # final processing of the joints
    joints_3d = posesmoothing.joint_constrain( processed_pose_3d )

    fig = plt.figure()
    ax_fig = fig.add_subplot(121)
    # dynamic_canvas = FigureCanvas(fig)

    ax = fig.add_subplot(122, projection='3d' )
    # ax = fig.gca(projection='3d')

    for i in range( joints_3d.shape[0] ):
        if i < gs_start:
            vis.vis_2d(images[i], pose_2d[i]) # show the previous frame in the interpolatioins
            ax_fig.imshow(images[i])
            # plt.axis('off')
            # Show 3D poses
            # post processing the joints
            # try:
            #     plt.close(fig)
                # plt.close(fig=i-1)
            # except:
            #     pass
            # ax.clear()
            vis.vis_3d(joints_3d[i], ax, visible = [-1], cfrm = i)
            # ax.figure.canvas.draw()
            plt.show()

        else:
            if i % 5 ==0:
                # plt.figure(i)
                vis.vis_2d(images[int(i/5)], pose_2d[int(i/5)])
                ax_fig.imshow(images[int(i/5)])
            # try:
            #     plt.close(fig)
            #     plt.close(fig=i-1)
            # except:
            #     pass

            # ax.clear()
            vis.vis_3d(joints_3d[i], ax, visible = [-1], cfrm = i) # self, pose_3d, ax, visible = None, cfrm = 0
            # ax.figure.canvas.draw()
            plt.show()


    print('plotting finished')


    # plotting the joint history 1-1


    tx = np.linspace(0, joints_3d.shape[0], joints_3d.shape[0] + 1)
    for i in range(joints_3d.shape[2]):
        plt.Figure(5)
        plt.plot(tx[:-1], joints_3d[:,0,i]) # for plotting x
        plt.Figure(6)
        plt.plot(tx[:-1], joints_3d[:,1,i]) # for plotting y
        plt.Figure(7)
        plt.plot(tx[:-1], joints_3d[:,2,i]) # for plotting z



    # next step is to merge this with qt



    # ----------- Testing codea --------------
    # time_length = len(pose_3d)
    # t_axis = np.linspace(0, time_length, time_length + 1)
    #
    # # decision factors (z position of hands)
    # z_axis_lh = [pose_3d[i][2, 13] for i in range(time_length) ]
    # z_axis_rh = [pose_3d[i][2, 13] for i in range(time_length) ]
    #
    # # plotting before fitting
    # plt.figure(0)
    # plt.plot(t_axis[1:], z_axis_lh, 'b o')
    #
    #
    # # detecting the beginning and end of the shot
    # dz_lh = [a - b for a, b in zip(z_axis_lh[2:], z_axis_lh[0:-2])]
    # dz_rh = [a - b for a, b in zip(z_axis_rh[2:], z_axis_rh[0:-2])]
    # # rescaling
    # init_lh = np.where( np.array(dz_lh) > 100 )[0][0]
    # init_rh = np.where( np.array(dz_rh) > 100 )[0][0]
    # end_lh = np.argmin(dz_lh)
    # end_rh = np.argmin(dz_rh)
    # #
    # gs_start = int((init_lh + init_rh)/2) - 1
    # gs_end = int((end_lh + end_rh)/2) + 3
    #
    # verifying the results
    # plt.figure(2)
    # plt.plot(t_axis[2:-1], dz_lh, 'r--')
    # plt.plot(t_axis[2:-1], dz_rh, 's')
    # plt.title('plotting decision factor - all')
    # #
    # plt.figure(3)
    # plt.plot(t_axis[gs_start:gs_end], z_axis_lh[gs_start:gs_end], 'r--')
    # plt.plot(t_axis[gs_start:gs_end], z_axis_rh[gs_start:gs_end], ' s')
    # plt.title('plotting the desired interval')
    # #
    #  approxixmating for smoothing the motion
    #
    # deg = 4
    # #Number of samples of our data d
    # nsamples = gs_end - gs_start
    # #The independent variable sampled at regular intervals
    # xs=np.linspace(-1, 1, nsamples)
    # #The Legendre Vandermonde matrix
    # V=leg.legvander(xs, deg)
    # #Generate some data to fit
    # ys=xs*np.pi
    # # f=np.cos(ys) + np.sin(ys)*np.sin(ys)*np.sin(ys)+np.cos(ys)*np.cos(ys)*np.cos(ys*ys)
    # #Do the fit
    # coeffs=np.linalg.lstsq(V, z_axis_lh[gs_start:gs_end], rcond=None)[0]
    # #Evaluate the fit for plotting purposes
    # g=leg.legval(xs, coeffs)
    #
    #
    # # plotting before fitting
    # plt.figure(4)
    # plt.plot(t_axis[gs_start:gs_end], g, 'b o')
    #
    # # upsampling the detections
    # resampled_g = signal.resample(g, len(g)*5, t_axis[gs_start:gs_end])
    # #
    # plt.figure(5)
    # plt.plot(resampled_g[1], resampled_g[0], 'r >')
    # verified


    # adding the smothed form to the visualization









