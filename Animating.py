#
# creating animation from the computed poses
#
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import pickle

#
from src import PoseEstimator

#
fig, ax = plt.subplots()
fig.set_tight_layout(True)
#
# Query the figure's on-screen size and DPI. Note that when saving the figure to

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


# parameters
# threshold
VISIBLE_PART = 1e-3
MIN_NUM_JOINTS = 5
CENTER_TR = 0.4

# net attributes
SIGMA = 7
STRIDE = 8
SIGMA_CENTER = 21
INPUT_SIZE = 368
OUTPUT_SIZE = 46
NUM_JOINTS = 14
NUM_OUTPUT = NUM_JOINTS + 1
H36M_NUM_JOINTS = 17

# draw options
JOINT_DRAW_SIZE = 3
LIMB_DRAW_SIZE = 2
NORMALISATION_COEFFICIENT = 1280*720 # this normalization coefficient can be manipulated by us

# test options
BATCH_SIZE = 4
_CONNECTION = [
    [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
    [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15],
    [15, 16]]

def draw_limbs(image, pose_2d, visible):
    """Draw the 2D pose without the occluded/not visible joints."""

    _COLORS = [
        [251, 10, 0], [250, 100, 5], [180, 200, 10], [0, 10, 200],
        [1, 200, 180], [2, 180, 250], [5, 10, 255], [180, 10, 250],
        [255, 10, 180]
    ]
    _LIMBS = np.array([0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9,
                       9, 10, 11, 12, 12, 13]).reshape((-1, 2))

    _NORMALISATION_FACTOR = int( math.floor( math.sqrt( image.shape[0] * image.shape[1] / NORMALISATION_COEFFICIENT)))

    for oid in range(pose_2d.shape[0]):
        for lid, (p0, p1) in enumerate(_LIMBS):
            if not (visible[oid][p0] and visible[oid][p1]):
                continue
            y0, x0 = pose_2d[oid][p0]
            y1, x1 = pose_2d[oid][p1]
            cv2.circle(image, (x0, y0), JOINT_DRAW_SIZE *_NORMALISATION_FACTOR , _COLORS[lid], -1)
            cv2.circle(image, (x1, y1), JOINT_DRAW_SIZE*_NORMALISATION_FACTOR , _COLORS[lid], -1)
            cv2.line(image, (x0, y0), (x1, y1),
                     _COLORS[lid], LIMB_DRAW_SIZE*_NORMALISATION_FACTOR , 16)

def joint_color(j):
    colors = [
        (10, 250, 255), (250, 10, 0), (0, 250, 10),
        (5, 10, 10), (255, 5, 250), (5, 10, 250),
    ]
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

def plot_pose(pose):
    """Plot the 3D pose showing the joint connections."""
    import mpl_toolkits.mplot3d.axes3d as p3
    _CONNECTION = [
        [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
        [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15],
        [15, 16]]

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    for c in _CONNECTION:
        col = '#%02x%02x%02x' % joint_color(c[0])
        ax.plot([pose[0, c[0]], pose[0, c[1]]],
                [pose[1, c[0]], pose[1, c[1]]],
                [pose[2, c[0]], pose[2, c[1]]], c=col)


    for j in range(pose.shape[1]):
        col = '#%02x%02x%02x' % joint_color(j)
        ax.scatter(pose[0, j], pose[1, j], pose[2, j],
                   c=col, marker='*', edgecolor=col)
    smallest = pose.min()
    largest = pose.max()
    ax.set_xlim3d(smallest, largest)
    ax.set_ylim3d(smallest, largest)
    ax.set_zlim3d(smallest, largest)

    return fig

if __name__ == '__main__':
    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    # anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=200)
    out_name = 'mlm_03_animated.gif'
    start = 0
    end = 300
    # reading 3D  data for representation
    with open('data/demo_data/pose_2d_mlm_04_mlm_03.pickle', 'rb') as f:
        pose_2ds = pickle.load(f)[start:end]
    # pose_2ds = np.array()[:][ 0, :, :].transpose(0,2,1)
    print('Size of the video: ', len(pose_2ds))
    pose_2d = []
    for j in range(len(pose_2ds)):
        pose_2d.append(pose_2ds[j][ 0, :, :].transpose(1,0))
    pose_2ds = np.array(pose_2d)
    with open('data/demo_data/pose_3d_mlm_04_mlm_03.pickle', 'rb') as f:
        pose_3ds = pickle.load(f)[start:end]
    pose_3d = []
    for j in range(len(pose_3ds)):
        pose_3d.append(pose_3ds[j][ 0, :, :])
    pose_3ds = np.array(pose_3d)
    # pose_3ds = np.array(pose_3ds)[:, 0, :, :]
    with open('data/demo_data/visibility_mlm_04_mlm_03.pickle', 'rb') as f:
        visibilities = pickle.load(f)[start:end]

    with open('data/demo_data/frame_mlm_04_mlm_03.pickle', 'rb') as f:
        frames = pickle.load(f)[start:end]

    size = 10
    fps = 10

    # get number of detected people

    plt.ioff()
    fig = plt.figure(figsize=(size*(1 + 1), size))
    ax_in = fig.add_subplot(1, 1 + 1, 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7

    # for index, data in enumerate(pose_3ds):
    ax = fig.add_subplot(1, 2 , 2, projection='3d')
    # ax.view_init(elev=15., azim=azim) #
    ax.set_xlim3d([-radius/2, radius/2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius/2, radius/2])
    ax.set_aspect('equal')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.dist = 7.5
    # ax.set_title(title) #, pad=35
    smallest = np.array(pose_3ds).min()
    largest = np.array(pose_3ds).max()

    # Setting the axes properties
    ax.set_xlim3d(smallest, largest)
    ax.set_xlabel('X')

    ax.set_ylim3d(smallest, largest)
    ax.set_ylabel('Y')

    ax.set_zlim3d(smallest, largest)
    ax.set_zlabel('Z')

    ax.set_title('3D Test')

    ax_3d.append(ax)
    lines_3d.append([])
    #
    all_frames = frames
    global initialized, image, limit
    initialized = False
    limit = pose_3ds.shape[0]
    image = None
    lines = []
    points = None

    def update_video(i):
        global initialized , image, limit #lines, points
        # Update 2D poses
        # we have 2d poses

        if not initialized:
            # put limbs on the image
            img = all_frames[i]
            draw_limbs(img, np.expand_dims(pose_2ds[i].T, 0), visibilities[i])
            image = ax_in.imshow(img, aspect='equal')

            for c in _CONNECTION:
                col = '#%02x%02x%02x' % joint_color(c[0])
                # ax.plot([pose_2ds[i, 0, c[0]], pose_2ds[i, 0, c[1]]],
                #         [pose_2ds[i, 1, c[0]], pose_2ds[i, 1, c[1]]],
                        # [pose[2, c[0]], pose[2, c[1]]],
                        # c=col)
                lines_3d.append(ax.plot([pose_3ds[i, 0, c[0]], pose_3ds[i, 0, c[1]]],
                                        [pose_3ds[i, 1, c[0]], pose_3ds[i, 1, c[1]]],
                                        [pose_3ds[i, 2, c[0]], pose_3ds[i, 2, c[1]]],
                                           zdir='z', c=col))

            # points = ax_in.scatter(*pose_2ds[i].T, 5, color='red', edgecolors='white', zorder=10)
            initialized = True
        else:
            img = all_frames[i]
            draw_limbs(img, np.expand_dims(pose_2ds[i].T, 0), visibilities[i])
            image.set_data( img )
            for idx, c in enumerate(_CONNECTION):
                # col = '#%02x%02x%02x' % joint_color(c[0])
                lines_3d[idx+1][0].set_xdata([pose_3ds[i, 0, c[0]], pose_3ds[i, 0, c[1]]])
                lines_3d[idx+1][0].set_ydata([pose_3ds[i, 1, c[0]], pose_3ds[i, 1, c[1]]])
                lines_3d[idx+1][0].set_3d_properties( [pose_3ds[i, 2, c[0]], pose_3ds[i, 2, c[1]]], zdir='z')

        print('{}/{}      '.format(i, limit), end='\r')

    update_video(0)
    fig.tight_layout()
    anim = animation.FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=3000/fps, repeat=True)
    anim.save('simulations/' + out_name , dpi=80, writer='imagemagick')

    # if output.endswith('.mp4'):
    #     Writer = writers['ffmpeg']
    #     writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
    #     anim.save(output, writer=writer)
    # elif output.endswith('.gif'):
    #     anim.save(output, dpi=80, writer='imagemagick')
    # else:
    #     raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()
