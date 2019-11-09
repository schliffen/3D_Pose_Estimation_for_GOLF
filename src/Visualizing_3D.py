#
# Collecting the player Statistics
#
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import cv2
#import mayavi


def joint_name_3d( joint):
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
def joint_name_2d( joint):
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

class stats():
    def __init__(self, video_name, finit, fend):
        self.video_name = video_name
        self.save_dir = '/home/ali/CLionProjects/PoseEstimation/Golf_3D_Pose_my_impl_/results/kyle/'
        self.finit = finit
        self.series = '07' # this is for naming
        self.fend = fend
        self.pad = (243 - 1) // 2 # Padding on each side
        self.causal_shift = 0
        self.num_kpoints = 17
        self.keypoints_symmetry = [[1, 3, 5, 7, 9, 11, 13, 15],[2, 4, 6, 8, 10, 12, 14, 16]]
        self.rot = np.array([ 0.14070565, -0.15007018, -0.7552408 ,  0.62232804], dtype=np.float32)
        self.skeleton_parents =  np.array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15])
        self.pairs = [(1,2), (5,4),(6,5),(8,7),(8,9),(10,1),(11,10),(12,11),(13,1),(14,13),(15,14),(16,2),(16,3),(16,4),(16,7)]
        self.kps_left, kps_right = list(self.keypoints_symmetry[0]), list(self.keypoints_symmetry[1])
        self.joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])
        self.joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
                       [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                       [5, 11], [6, 12], [11, 12],
                       [11, 13], [12, 14], [13, 15], [14, 16]]

        # ----------------< listing pose points >---------------
        self.best_pract = [] # for loading info
        self.best_parct_pose = {} # for saving computation results
    #
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
    def update_state(self, pnt3d):
        # head is point number
        # self.head_hand_dist.append( [ np.linalg.norm((pnt3d[13] + pnt3d[16])/2 - pnt3d[9]), frm_num] ) # this part can be done instantly
        self.best_pract.append(pnt3d)
        # self.hand_l.append(pnt3d[13])
        # self.hand_r.append(pnt3d[16])
        # self.knee_l.append( [pnt3d[2], frm_num] )
    def get_data(self):
        return np.array(self.best_pract)

    def save_stats(self):
        with open(self.save_dir + self.video_name + '.pickle', 'wb') as f:
            pickle.dump(self.best_pract, f)
        # with open(self.save_dir + self.video_name + '_hand_l_pos.pickle', 'wb') as f:
        #     pickle.dump(self.hand_l, f)
        # with open(self.save_dir + self.video_name + '_hand_r_pos.pickle', 'wb') as f:
        #     pickle.dump(self.hand_r, f)
        # with open(self.save_dir + 'knee_l.pickle', 'w') as f:
        #     pickle (self.knee_l, f)

    def load_opt_stat(self, optim_data):
        # optim_data = 'hand_l_pos.pickle'
        # todo: create best practice file
        # todo: checking wether best practice file exists
        # reference for confidence interval
        z = {'70': .5, '80':1.282, '85':1.440, '90':1.645, '95':1.960, '99':2.576, '995':2.807, '999':3.291}
        # loading optimal poses from list of statistics
        # loading best practice cases ---

        self.best_pract = []
        for i, item in enumerate(optim_data):
            with open(self.save_dir + item + '.pickle', 'rb') as f:
                self.best_pract.append(pickle.load(f)[0])

        # collecting the data for each pose
        self.best_pract = np.array(self.best_pract) # .transpose(1,2,0)
        dim = self.best_pract.shape[0]
        bp_m, bp_s = stdd(self.best_pract)
        # computing confidence interval
        bp_u = bp_m + z['70'] * bp_s/np.sqrt(dim)
        bp_l = bp_m - z['70'] * bp_s/np.sqrt(dim)
        self.bp_m = bp_m.transpose(2,0,1)
        self.bp_l = bp_l.transpose(2,0,1)
        self.bp_u = bp_u.transpose(2,0,1)
        #
        bpdict = {'bp_m': self.bp_m, 'bp_l':self.bp_l, 'bp_u':self.bp_u}
        # return bp_m.transpose(2,0,1), bp_l.transpose(2,0,1), bp_u.transpose(2,0,1)
        # dumping down statistics
        with open(self.save_dir + 'best_practice_stats.pickle', 'wb') as f:
            pickle.dump(bpdict, f)

    def comparing_poses(self, pose_c):
        """
        the goal of this function is to compare poses illustratively
        :return:
        """
        # computing Euclidian norm
        eu_nrm = [ ]
        def compute_norm(pose_c, bp_m):
            for i in range(pose_c.shape[1]):
                eu_nrm.append( np.sum(np.sqrt((pose_c[:, i, 0] - bp_m[:self.fend-self.finit, i, 0])**2 + \
                            (pose_c[:, i, 1] - bp_m[:self.fend-self.finit, i, 1])**2 + \
                            (pose_c[:, i, 2] - bp_m[:self.fend-self.finit, i, 2])**2)) )
            return eu_nrm
        self.cmprslt = compute_norm(pose_c, self.bp_m)

    # plotting 2D
    def draw_2D_stats(self, pos, fig):
        import matplotlib.pyplot as plt
        clst = self.cmprslt
        self.errp = []
        # looking for only 3 largest difference
        for i in range(3):
            self.errp.append( np.where(clst== max(clst))[0][0] )
            clst.pop(self.errp[i])

        # fig = plt.figure( figsize=(12,6) )
        ax = fig.add_subplot(221)
        # plotting 2D comparison for the joints
        #
        xindx = np.linspace(self.finit, self.fend, self.fend - self.finit + 1 )
        ax.scatter(xindx[:-1], pos[:, self.errp[0], 2], lw = 1, color = '#539caf', alpha = 1, label = 'Fit')
        ax.scatter(xindx[:-1], self.bp_m[:self.fend - self.finit , self.errp[0], 2], lw = 1, color = '#007539', alpha = 1, label = 'Fit') # plotting only the z difference
        ax.fill_between(xindx[:-1], self.bp_l[:self.fend - self.finit, self.errp[0], 2], self.bp_u[:self.fend - self.finit, self.errp[0], 2], color = '#f00e0e', alpha = 0.4, label = '95% CI')
        #
        ax.set_xlabel('time frame')
        ax.set_ylabel('height of:' + joint_name_3d(self.errp[0]))
        hnds, _ = ax.get_legend_handles_labels()
        lbls = ['tranee', 'best performance', '50% confidence']
        ax.legend(hnds[::-1], lbls[::-1])


    def draw_error_3D(self, pos, fig):
        #
        # plotting the problematic area
        #
        # implementing required transformations on the pose coordinates
        ax = fig.add_subplot(223, projection='3d')
        radius = 1.7
        ax.view_init(elev=10., azim=45.)
        # ax.set_xlim3d([-radius/2, radius/2])
        # ax.set_zlim3d([0, radius])
        # ax.set_ylim3d([-radius/2, radius/2])
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel(joint_name_3d(self.errp[0]))
        #
        ax.set_xticklabels([-1., -.8 , -.6, -.4, -.2, 0.0, .2, .4, .6, .8, 1.])
        ax.set_yticklabels([-1., -.8 , -.6, -.4, -.2, 0.0, .2, .4, .6, .8, 1.])
        ax.set_zticklabels([0.0, .2, .4, .6, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.0])
        ax.dist = 7.5
        parents = self.skeleton_parents

        for j, j_parent in enumerate(parents):
            if not j_parent in self.errp:
                continue
            # managing colors
            if j_parent == self.errp[0]:
                markersize=4
                marker = 'D'
                col_1 = '#f00e0e'
                col_2 = '#4016f7'

            elif j_parent == self.errp[1]:
                markersize=3
                marker = '<'
                col_1 = '#f77b16'
                col_2 = '#2478ff'

            elif j_parent == self.errp[2]:
                markersize=2
                marker = '>'
                col_1 = '#fabf0c'
                col_2 = '#b202f7'

            # using set position instead
            for i in range(pos.shape[0]):
                # ax.plot([pos[i, j, 0], pos[i, j_parent, 0]],
                #         [pos[i, j, 1], pos[i, j_parent, 1]],
                #         [pos[i, j, 2], pos[i, j_parent, 2]], zdir='z', c=col_1 )
                # ax.plot([self.bp_m[i, j, 0], self.bp_m[i, j_parent, 0]],
                #         [self.bp_m[i, j, 1], self.bp_m[i, j_parent, 1]],
                #         [self.bp_m[i, j, 2], self.bp_m[i, j_parent, 2]], zdir='z', c=col_2 )

                ax.scatter(pos[i, j, 0], pos[i, j, 1], pos[i, j, 2], c=col_1, marker=marker, edgecolor=col_1, s=markersize)
                ax.scatter(self.bp_m[i, j, 0], self.bp_m[i, j, 1], self.bp_m[i, j, 2], c=col_2, marker='>', edgecolor=col_2, s=markersize)

    def draw_geometric_features(self, pos):
        #
        """
        the goal is to plot human pose features frame by frame.
        In order to show the differences, the colors of the poses would be RED
        with a bit larger marker and also RED limbs.
        There are arrow showing the mean best performance position.

        :param pos: computed poses for current video
            :param indx: the frame number
        :return: void
        """
        from mpl_toolkits.mplot3d import Axes3D # projection 3D
        from matplotlib import animation
        from mpl_toolkits.mplot3d import axes3d as p3
        #
        global initialize, image
        initialize = False
        fps = 30 # take it 30 for the moment
        radius = 2.0 # change this to max of z axis ..
        line_3d = []
        #
        fig = plt.figure( 2, figsize=(14, 8) )
        # plotting on image ----------
        # self.plot_on_image(pos2d_frms, img_frms, ax_fig)
        # plotting different aspects in different windows
        ax = fig.gca(projection='3d')
        ax.view_init(elev=10., azim=40.)
        ax.set_xlabel('length')
        ax.set_xlim3d([-radius/2, radius/2])
        ax.set_zlabel('height')
        ax.set_zlim3d([0, radius])
        ax.set_ylabel('width')
        ax.set_ylim3d([-radius/2, radius/2])
        ax.set_aspect('equal')
        #
        ax.set_xticklabels([-1., -.8 , -.6, -.4, -.2, 0.0, .2, .4, .6, .8, 1.])
        ax.set_yticklabels([-1., -.8 , -.6, -.4, -.2, 0.0, .2, .4, .6, .8, 1.])
        ax.set_zticklabels([0.0, .2, .4, .6, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.0])
        ax.dist = 7.5
        line_3d.append([])
        joint_pairs = self.joint_pairs
        #
        parents = self.skeleton_parents
        # ----------< another way of representaing >-------
        def update_frame( i ):
            #
            # updating for animation
            #
            global initialize
            if not initialize:
                for j, j_parent in enumerate(parents):
                    if j_parent == -1:
                        continue
                    # managing colors and markers ----
                    if j == 11:
                        markersize=5
                        marker = 'o'
                        col = '#f00e0e'
                    elif j == 13:
                        markersize= 5
                        marker = 'o'
                        col = '#f00e0e'
                    elif j == 16:
                        markersize= 5
                        marker = 'o'
                        col = '#f00e0e'
                    else:
                        markersize=5
                        marker = 's'
                        if j in range(1, 4):
                            col = '#%02x%02x%02x' % (10, 250, 255)
                        if j in range(4, 7):
                            col = '#%02x%02x%02x' % (0, 250, 10)
                        if j in range(9, 11):
                            col = '#%02x%02x%02x' % (5, 10, 250)
                        if j in range(11, 14):
                            col = '#%02x%02x%02x' % (0, 225, 120)
                        if j in range(14, 17):
                            col = '#%02x%02x%02x' %  (5, 10, 10)
                    # 3D
                    line_3d.append( ax.plot([pos[i, j, 0], pos[i, j_parent, 0]],
                                            [pos[i, j, 1], pos[i, j_parent, 1]],
                                            [pos[i, j, 2], pos[i, j_parent, 2]], zdir='z', c=col, marker = marker, markersize =markersize ) )
                # plotting a plane
                # computing normal
                pt3 = (pos[i, 13] + pos[i,16])/2
                normal_v = np.cross(pos[i, 14]- pos[i,11], pt3 - pos[i,11])
                normal_v /= np.linalg.norm(normal_v)
                middle_point = (pos[i, 14] + pos[i,11] + pt3)/3
                # plotting middle points
                ax.scatter(middle_point[0], middle_point[1], middle_point[2], c=col, marker='*', edgecolor=col)
                ax.quiver(middle_point[0], middle_point[1], middle_point[2], -normal_v[0], -normal_v[1], -normal_v[2])
                #
                # d = -pt3.dot(normal_v)
                # xx,yy = np.meshgrid(np.arange(middle_point[0]-.05,  middle_point[0]+.05, .01), \
                #                     np.arange(middle_point[1]-.05, middle_point[1]+.05, .01))
                # zz = (-normal_v[0] * xx - normal_v[1] * yy - d) * 1. /normal_v[2]
                # ax.plot_surface(xx,yy,zz, alpha = .2)
                # line_3d.append( ax )
                # plotting other planes

                initialize = True
            else:
                #
                for j, j_parent in enumerate(parents):
                    if j_parent == -1:
                        continue
                    # col = 'red' if j in joints_right else 'black' skeleton_parents
                    # managing colors and markers ----
                    if j == self.errp[0]:
                        markersize=40
                        marker = 'o'
                        col = '#f00e0e'
                    elif j == self.errp[1]:
                        markersize=30
                        marker = 'o'
                        col = '#f00e0e'
                    elif j == self.errp[2]:
                        markersize=20
                        marker = 'o'
                        col = '#f00e0e'
                    else:
                        markersize=50
                        marker = 'o'
                        if j in range(1, 4):
                            col = '#%02x%02x%02x' % (10, 250, 255)
                        if j in range(4, 7):
                            col = '#%02x%02x%02x' % (20, 20, 100)
                        if j in range(9, 11):
                            col = '#%02x%02x%02x' % (5, 10, 250)
                        if j in range(11, 14):
                            col = '#%02x%02x%02x' % (15, 250, 50)
                        if j in range(14, 17):
                            col = '#%02x%02x%02x' %  (5, 100, 50)
                    # 3D ----
                    line_3d[j ][0].set_xdata([pos[i, j, 0], pos[i, j_parent, 0]])
                    line_3d[j ][0].set_ydata([pos[i, j, 1], pos[i, j_parent, 1]])
                    line_3d[j ][0].set_3d_properties( [pos[i, j, 2], pos[i, j_parent, 2]], zdir='z')
                # plotting a plane
                # computing normal
                pt3 = (pos[i, 13] + pos[i,16])/2
                normal_v = np.cross(pos[i, 14]- pos[i,11], pt3 - pos[i,11])
                normal_v /= np.linalg.norm(normal_v)
                middle_point = (pos[i, 14] + pos[i,11] + pt3)/3
                # plotting middle points
                line_3d[j][0].add_callback(ax.scatter(middle_point[0], middle_point[1], middle_point[2], c=col, marker='*', edgecolor=col))
                line_3d[j][0].add_callback(ax.quiver(middle_point[0], middle_point[1], middle_point[2], -normal_v[0], -normal_v[1], -normal_v[2]))
                #line_3d[j][0].add_callback(ax.
                # d = -pt3.dot(normal_v)
                # xx,yy = np.meshgrid(np.arange(middle_point[0]-.05,  middle_point[0]+.05, .01), \
                #                     np.arange(middle_point[1]-.05, middle_point[1]+.05, .01))
                # zz = (-normal_v[0] * xx - normal_v[1] * yy - d) * 1. /normal_v[2]
                # line_3d[j][0].add_callback(ax.plot_surface(xx,yy,zz, alpha = .2))

                # plotting other planes
                print('{}     '.format(i), end='\r')
        #
        update_frame(0)
        # update_frame(1)
        # update_frame(2)
        fig.tight_layout()
        anim = animation.FuncAnimation(fig=fig,
                                       func=update_frame,
                                       frames=range(0, self.fend - self.finit),
                                       interval=3000/fps,
                                       repeat=True
                                       )
        # saving the animation
        anim.save(filename= self.save_dir + 'animated_pose_features_' + self.series + '.gif', dpi = 80, writer='imagemagick')
        # anim.to_html5_video(self.save_dir + 'animated_psoe.gif', 'imagemagick')

    def plot_on_image(self, pos2d, img, plt):
        joint_pairs = self.joint_pairs
        # for item in pos2d:
        #     x, y = int(item[0]), int(item[1])
        #     cv2.circle(img, (x, y), 1, (255, 5, 0), 5)
        #     cv2.putText(img, str)
        for pair in joint_pairs:
            j, j_parent = pair
            # jj = self.joint_name_2d(self.joint_name_3d(j))
            pt1 = (int(pos2d[j][0]), int(pos2d[j][1]))
            pt2 = (int(pos2d[j_parent][0]), int(pos2d[j_parent][1]))
            cv2.circle(img, pt1, 1, (255, 5, 0), 5)
            cv2.putText(img, str(j), pt1, cv2.FONT_HERSHEY_COMPLEX, 2, (255,250,250), 2, cv2.LINE_AA)
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
        plt.imshow(img)

    # smooth plotting 3D------------------
    def draw_3D_smooth(self, pos, pos2d_frms, img_frms ):
        """
        the goal is to plot human pose frame by frame.
        In order to show the differences, the colors of the poses would be RED
        with a bit larger marker and also RED limbs.
        There are arrow showing the mean best performance position.

        :param pos: computed poses for current video
        :param indx: the frame number
        :return: nothing
        """
        from mpl_toolkits.mplot3d import Axes3D # projection 3D
        from matplotlib import animation
        from mpl_toolkits.mplot3d import axes3d as p3
        #
        global initialize, image
        fps = 30 # take it 30 for the moment
        radius = 2.0 # change this to max of z axis ..
        line_3d = []
        #
        fig = plt.figure( 1, figsize=(14, 8) )
        fig.add_subplot(221)
        self.draw_2D_stats( pos, fig)
        self.draw_error_3D( pos, fig)
        # plotting extra geometric features
        self.draw_geometric_features(pos)
        # plotting on image ----------
        initialize = False
        fig_1 = plt.figure( 3, figsize=(14, 8) )
        ax_fig = fig_1.add_subplot(111)
        # self.plot_on_image(pos2d_frms, img_frms, ax_fig)
        # plotting different aspects in different windows
        ax = fig.add_subplot(122, projection='3d')
        ax.view_init(elev=10., azim=40.)
        ax.set_xlabel('length')
        ax.set_xlim3d([-radius/2, radius/2])
        ax.set_zlabel('height')
        ax.set_zlim3d([0, radius])
        ax.set_ylabel('width')
        ax.set_ylim3d([-radius/2, radius/2])
        ax.set_aspect('equal')
        #
        ax.set_xticklabels([-1., -.8 , -.6, -.4, -.2, 0.0, .2, .4, .6, .8, 1.])
        ax.set_yticklabels([-1., -.8 , -.6, -.4, -.2, 0.0, .2, .4, .6, .8, 1.])
        ax.set_zticklabels([0.0, .2, .4, .6, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.0])
        ax.dist = 7.5
        line_3d.append([])
        joint_pairs = self.joint_pairs
        #
        parents = self.skeleton_parents
        errp = self.errp
        series = self.series
        # ----------< another way of representaing >-------
        def update_frame( i ):
            #
            # updating for animation
            #
            global initialize, image
            img = img_frms[i].copy()
            if not initialize:
                # plotting 2d poses
                # for item in pos2d_frms[i]:
                #     x, y = int(item[0]), int(item[1])
                #     cv2.circle(img, (x, y), 1, (255, 5, 0), 5)
                for pair in joint_pairs:
                    j, j_parent = pair
                    pt1 = (int(pos2d_frms[i][j][0]), int(pos2d_frms[i][j][1]))
                    pt2 = (int(pos2d_frms[i][j_parent][0]), int(pos2d_frms[i][j_parent][1]))
                    if j_parent == joint_name_2d( joint_name_3d(errp[0])):
                        cv2.line(img, pt1, pt2, (255, 5, 10), 2)
                        # cv2.circle(img, pt1, 1, (255, 5, 0), 5)
                        # cv2.putText(img, str(j_parent), pt2, cv2.FONT_HERSHEY_COMPLEX, 1, (0,250,250), 1, cv2.LINE_AA)
                        cv2.circle(img, pt2, 2, (255, 5, 0), 15)
                    else:
                        cv2.line(img, pt1, pt2, (0, 255, 0), 2)
                        cv2.circle(img, pt1, 1, (2, 5, 250), 5)
                        # cv2.putText(img, str(j_parent), pt1, cv2.FONT_HERSHEY_COMPLEX, 1, (255,250,0), 1, cv2.LINE_AA)
                # -------------------plotting line part ---------------
                j1 = joint_name_2d( joint_name_3d(14))
                pt1 = (int(pos2d_frms[i][j1][0]), int(pos2d_frms[i][j1][1]))
                j2 = joint_name_2d( joint_name_3d(16))
                pt2 = [int(pos2d_frms[i][j2][0]), int(pos2d_frms[i][j2][1])]
                # line equations
                a = (pt1[1] - pt2[1])/(pt1[0] - pt2[0] + 0.00001)
                b = pt2[1] - a*pt2[0]
                # computing the line limit on the ground ...
                # this line is drawn alongside the right shoulder-hand
                pt2[1] = int(round(max(pos2d_frms[i][:,1])))
                pt2[0] = int(round((pt2[1] - b)/(a+.00001)))
                cv2.line(img,  pt1, \
                         tuple(pt2),\
                         (255,2,2), 3, cv2.LINE_AA)
                # this line is drawn prependicular to players head
                pt1 = (int(pos2d_frms[i][0][0]), int(pos2d_frms[i][0][1]))
                pt2[0] = pt1[0]
                cv2.line(img,  pt1, \
                         tuple(pt2), \
                         (5,255,2), 3, cv2.LINE_AA)

                image = ax_fig.imshow(img, aspect='equal')

                for j, j_parent in enumerate(parents):
                    if j_parent == -1:s
                        continue
                    # managing colors and markers ----
                    if j == errp[0]:
                        markersize=10
                        marker = 'o'
                        col = '#f00e0e'
                    elif j == errp[1]:
                        markersize= 5
                        marker = 'o'
                        col = '#f00e0e'
                    elif j == errp[2]:
                        markersize= 5
                        marker = 'o'
                        col = '#f00e0e'
                    else:
                        markersize=5
                        marker = 's'
                        if j in range(1, 4):
                            col = '#%02x%02x%02x' % (10, 250, 255)
                        if j in range(4, 7):
                            col = '#%02x%02x%02x' % (0, 250, 10)
                        if j in range(9, 11):
                            col = '#%02x%02x%02x' % (5, 10, 250)
                        if j in range(11, 14):
                            col = '#%02x%02x%02x' % (0, 225, 120)
                        if j in range(14, 17):
                            col = '#%02x%02x%02x' %  (5, 10, 10)
                    # 3D
                    line_3d.append( ax.plot([pos[i, j, 0], pos[i, j_parent, 0]],
                                            [pos[i, j, 1], pos[i, j_parent, 1]],
                                            [pos[i, j, 2], pos[i, j_parent, 2]], zdir='z', c=col, marker = marker, markersize =markersize ) )
                    # ax.scatter(pos[j, 0, i], pos[j, 1, i], pos[j, 2, i], c=col, marker='*', edgecolor=col)
                    # plotting a plane

                initialize = True
            else:
                for item in pos2d_frms[i]:
                    x, y = int(item[0]), int(item[1])
                    cv2.circle(img, (x, y), 1, (255, 5, 0), 5)
                for pair in joint_pairs:
                    j, j_parent = pair
                    pt1 = (int(pos2d_frms[i][j][0]), int(pos2d_frms[i][j][1]))
                    pt2 = (int(pos2d_frms[i][j_parent][0]), int(pos2d_frms[i][j_parent][1]))
                    if j_parent ==  joint_name_2d( joint_name_3d(errp[0])): #
                        cv2.line(img, pt1, pt2, (255, 5, 10), 2)
                        # cv2.circle(img, pt1, 1, (255, 5, 0), 5)
                        # cv2.putText(img, str(j), pt1, cv2.FONT_HERSHEY_COMPLEX, 1, (0,250,250), 1, cv2.LINE_AA)
                        cv2.circle(img, pt1, 2, (255, 5, 0), 15)
                    else:
                        cv2.line(img, pt1, pt2, (0, 255, 0), 2)
                        cv2.circle(img, pt1, 1, (2, 5, 250), 5)
                        # cv2.putText(img, str(j), pt1, cv2.FONT_HERSHEY_COMPLEX, 1, (255,250,0), 1, cv2.LINE_AA)
                # -------------------plotting line part ---------------
                j1 = joint_name_2d( joint_name_3d(14))
                pt1 = (int(pos2d_frms[i][j1][0]), int(pos2d_frms[i][j1][1]))
                j2 = joint_name_2d( joint_name_3d(16))
                pt2 = [int(pos2d_frms[i][j2][0]), int(pos2d_frms[i][j2][1])]
                # line equations
                a = (pt1[1] - pt2[1])/(pt1[0] - pt2[0] + .00001)
                b = pt2[1] - a*pt2[0]
                # computing the line limit on the ground ...
                # this line is drawn alongside the right shoulder-hand
                pt2[1] = int(round(max(pos2d_frms[i][:,1])))
                pt2[0] = int(round((pt2[1] - b)/(a+ .00001)))
                cv2.line(img,  pt1, \
                         tuple(pt2), \
                         (255,2,2), 3, cv2.LINE_AA)
                # this line is drawn prependicular to players head
                pt1 = (int(pos2d_frms[i][0][0]), int(pos2d_frms[i][0][1]))
                pt2[0] = pt1[0]
                cv2.line(img,  pt1, \
                         tuple(pt2), \
                         (5,255,2), 3, cv2.LINE_AA)

                image.set_data(img)
                #
                for j, j_parent in enumerate(parents):
                    if j_parent == -1:
                        continue
                    # col = 'red' if j in joints_right else 'black' skeleton_parents
                    # managing colors and markers ----
                    if j == self.errp[0]:
                        markersize=40
                        marker = 'o'
                        col = '#f00e0e'
                    elif j == self.errp[1]:
                        markersize=30
                        marker = 'o'
                        col = '#f00e0e'
                    elif j == self.errp[2]:
                        markersize=20
                        marker = 'o'
                        col = '#f00e0e'
                    else:
                        markersize=50
                        marker = 'o'
                        if j in range(1, 4):
                            col = '#%02x%02x%02x' % (10, 250, 255)
                        if j in range(4, 7):
                            col = '#%02x%02x%02x' % (20, 20, 100)
                        if j in range(9, 11):
                            col = '#%02x%02x%02x' % (5, 10, 250)
                        if j in range(11, 14):
                            col = '#%02x%02x%02x' % (15, 250, 50)
                        if j in range(14, 17):
                            col = '#%02x%02x%02x' %  (5, 100, 50)
                    # 3D ----
                    line_3d[j ][0].set_xdata([pos[i, j, 0], pos[i, j_parent, 0]])
                    line_3d[j ][0].set_ydata([pos[i, j, 1], pos[i, j_parent, 1]])
                    line_3d[j ][0].set_3d_properties( [pos[i, j, 2], pos[i, j_parent, 2]], zdir='z')


                print('{}     '.format(i), end='\r')
            image.write_png('/home/ali/CLionProjects/PoseEstimation/Golf_3D_Pose_my_impl_/results/kyle/figures_2d_pose/' + '2d_pose_figs_'+ str(i) + '_' + series + '.png')
        #
        # for i1 in range(len(pos2d_frms)):
        update_frame(0)

        fig.tight_layout()
        anim = animation.FuncAnimation(fig=fig,
                                       func=update_frame,
                                       frames=range(0, self.fend - self.finit),
                                       interval=3000/fps,
                                       repeat=True
                                       )
        # saving the animation
        anim.save(filename= self.save_dir + 'animated_pose_' + self.series + '.gif', dpi = 80, writer='imagemagick')
        # anim.to_html5_video(self.save_dir + 'animated_psoe.gif', 'imagemagick')
