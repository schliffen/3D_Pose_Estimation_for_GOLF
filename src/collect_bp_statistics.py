#
# collecting best practice statistics for analysis
#
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
#
sl_dir = '/home/ali/CLionProjects/PoseEstimation/Golf_3D_Pose_my_impl_/results/'
js_dir = '/home/ali/CLionProjects/PoseEstimation/Golf_3D_Pose_my_impl_/input_video/'
#
# with open(sl_dir + '20190813_085858883_-0500.mp4.pickle', 'rb') as f:
#     vid_data = pickle.load(f)
#
# # selectingbest practices
#
# print('looking at data')

# plotting histogram of data
with open(js_dir + 'PGATOUR_drewlisec23.json', 'r') as f:
    vid_data = json.load(f)

scr_data = list( vid_data.values() )

plt.figure(1)
n, bins, patches = plt.hist(x=scr_data, bins=60, color='#0504aa', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('break angle Frequency')
plt.title(' Histogram')
plt.text(53, 45, r'u='+ str(int(np.mean(scr_data))) + ',' + ' b=' + str(int(np.mean(scr_data))))
maxfreq = n.max()
# Set a clean upper y-axis limit.
# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)