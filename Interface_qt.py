#
# trying to implemetn plotting interface in qt
#

"""
pyqt5
"""
import sys, os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import mpl_toolkits.mplot3d.axes3d as p3
from PIL import Image
from os.path import dirname, realpath
# importing related codes
from src import smoothing_prediction
from numpy.polynomial import legendre as leg
from scipy import signal
#
from PyQt5.QtWidgets import QFileDialog

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QCheckBox, QMainWindow, QSystemTrayIcon, QAction, QMenu, QStyle, qApp, QDialog, QWidget, QPushButton


from PyQt5.QtCore import pyqtSignal, QThread, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import time

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.uic import loadUiType

from matplotlib.figure import Figure
# from matplotlib.backends.backend_qt5agg import (
#     FigureCanvasQTAgg as FigureCanvas,
#     NavigationToolbar2QT as NavigationToolbar)
from Pose3DGUI.window import Ui_MainWindow
#
# loading required libraries from pose processing part ---
#import Interface_pose
from Interface_pose import estimate_start_end, estimate_smooth_pose, visualization

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



# from myMainWindow import Ui_Dialog

root = os.getcwd()
# Ui_MainWindow, QMainWindow = loadUiType(root + '/full_flow/Pose3DGUI/window.ui')
#
# os.system('pyuic5 ' + root + '/full_flow/Pose3DGUI/window.ui' + ' -o ' + root + '/full_flow/Pose3DGUI/window.py')

class Main(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)
        self.fig_dict = {}
        # self.cmplot = Ui_Dialog()
        # self.cmplot.setupUi(self)
        # self.mplfigs.itemClicked.connect(self.changefig)
        # fig = Figure()
        # self.addmpl(fig)
        # ----------< preparing Mian time-running plots >---------
        # self._main = QtWidgets.QWidget()
        # self.setCentralWidget(self._main)
        # static canvas --
        # static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        # -----------------
        #
        self.readVideo.clicked.connect(self.openFileDialog)
        #
        # final processing of the joints

        # working on different pose motions
        self.keep_tr = False
        self.signal = []
        self.state = {}
        # added by me
        fig = Figure()
        self.addmpl(fig)
        #
        self.img_cnt = 0
        # self.timer_1 = QTimer(self)
        # self.timer_1.timeout.connect(self._update_canvas_1)
        # self._timer = self.dynamic_canvas.new_timer(self.gs_end, [(self._update_canvas_2, (), {})])

        # self._timer_1 = self.dynamic_canvas.new_timer(100, [(self._update_canvas_2, (), {})])
        self.pose_cnt = 0
        # self.timer_2 = QTimer(self)
        # self.timer_2.timeout.connect(self._update_canvas_2)

        # self.timer_1.start()
        # self.timer_2.start()
        #
        self.state = {'0':0, '1':0, '2':0, '4':0, '5':0, '10':0, '11':0, '12':0, '13':0, '14':0, '15':0, '16':0}
        # # --< Checkboxes >--
        # Whether to keep traces
        self.kt = QCheckBox( "Keep Traces", parent=self.mplfigs)
        self.kt.stateChanged.connect(self.keep_trace)
        self.kt.setChecked(False)
        self.kt.move(40,30)
        self.kt.resize(320,40)

        #
        self.b_0 = QCheckBox( "Stomach", parent=self.mplfigs)
        self.b_0.stateChanged.connect(self.clickBox_0)
        # self.b_0.setChecked(True)
        self.b_0.move(40,250)
        self.b_0.resize(320,40)
        #
        self.b_1 = QCheckBox( "Left Pelvis", parent=self.mplfigs)
        self.b_1.stateChanged.connect(self.clickBox_1)
        # self.b_1.setChecked(True)
        self.b_1.move(40,275)
        self.b_1.resize(320,40)
        #
        self.b_2 = QCheckBox( "Left Knee", parent=self.mplfigs)
        self.b_2.stateChanged.connect(self.clickBox_2)
        # self.b_2.setChecked(True)
        self.b_2.move(40,325)
        self.b_2.resize(320,40)
        #
        self.b_4 = QCheckBox( "Right Pelvis", parent=self.mplfigs)
        self.b_4.stateChanged.connect(self.clickBox_4)
        # self.b_4.setChecked(True)
        self.b_4.move(40,300)
        self.b_4.resize(320,40)
        #
        self.b_5 = QCheckBox( "Right Knee", parent=self.mplfigs)
        self.b_5.stateChanged.connect(self.clickBox_5)
        # self.b_5.setChecked(True)
        self.b_5.move(40,350)
        self.b_5.resize(320,40)
        #
        # self.b_9 = QCheckBox( "Chin", parent=self.mplfigs)
        # self.b_9.stateChanged.connect(self.clickBox_9)
        # self.b_9.setChecked(True)
        # self.b_9.move(40,80)
        # self.b_9.resize(320,40)
        #
        self.b_10 = QCheckBox( "Head", parent=self.mplfigs)
        self.b_10.stateChanged.connect(self.clickBox_10)
        # self.b_10.setChecked(True)
        self.b_10.move(40,60)
        self.b_10.resize(320,40)
        #
        self.b_11= QCheckBox( "Left Shoulder", parent=self.mplfigs)
        self.b_11.stateChanged.connect(self.clickBox_11)
        # self.b_11.setChecked(True)
        self.b_11.move(40,85)
        self.b_11.resize(320,40)
        #
        self.b_12 = QCheckBox( "Left Elbow", parent=self.mplfigs)
        self.b_12.stateChanged.connect(self.clickBox_12)
        # self.b_12.setChecked(True)
        self.b_12.move(40,135)
        self.b_12.resize(320,40)

        self.b_13 = QCheckBox( "Left Wrist", parent=self.mplfigs)
        self.b_13.stateChanged.connect(self.clickBox_13)
        # self.b_13.setChecked(True)
        self.b_13.move(40,185)
        self.b_13.resize(320,40)

        self.b_14 = QCheckBox( "Right Shoulder", parent=self.mplfigs)
        self.b_14.stateChanged.connect(self.clickBox_14)
        # self.b_14.setChecked(True)
        self.b_14.move(40,110)
        self.b_14.resize(320,40)
        #
        self.b_15 = QCheckBox( "Right Elbow", parent=self.mplfigs)
        self.b_15.stateChanged.connect(self.clickBox_15)
        # self.b_15.setChecked(True)
        self.b_15.move(40,160)
        self.b_15.resize(320,40)
        #
        self.b_16 = QCheckBox( "Right Wrist", parent=self.mplfigs)
        self.b_16.stateChanged.connect(self.clickBox_16)
        # self.b_16.setChecked(True)
        self.b_16.move(40,210)
        self.b_16.resize(320,40)
        #
        # ---------------------------------------------------- DEMO -------------------------
        # self.b_1 = QCheckBox( "left hand", parent=self.mplfigs)
        # self.b_1.stateChanged.connect(self.clickBox_1)
        # self.b_1.setChecked(True)
        # self.b_1.move(40,80)
        # self.b_1.resize(320,40)
        #
        # self.b_2 = QCheckBox( "right hand", parent=self.mplfigs)
        # self.b_2.stateChanged.connect(self.clickBox_2)
        # self.b_2.setChecked(True)
        # self.b_2.move(40,100)
        # self.b_2.resize(320,40)
        # ------------------------------------------------------
        # setting the state
        self.mplfigs.itemClicked.connect(self.changefig)
        # self.mplfigs.itemClicked.connect(self.refresh)

    def openFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*)", options=options)
        if fileName:
            print(fileName)
            # self.img = Image.open(fileName)
            # self.timer.start(FPS)
            # global running
            # running = True
            # capture_thread = threading.Thread(target=grab, args=(fileName, q))
            # capture_thread.start()
            # setting options to select the video
            data_address = {'images': '20190716_151154_-0500_image_s', 'pose_2ds':'20190716_151154_-0500_pose2Ds', 'pose_3ds': '20190716_151154_-0500_pose3Ds'}
            #
            with open( ap.root + ap.data + data_address['pose_3ds'] + '.pickle', 'rb' ) as f:
                pose_3d = pickle.load(f) #
            with open( ap.root + ap.data + data_address['pose_2ds'] + '.pickle', 'rb' ) as f:
                self.pose_2d = pickle.load(f) # frm number x joints x coordinates
            with open( ap.root + ap.data + data_address['images'] + '.pickle', 'rb' ) as f:
                self.images = pickle.load(f) # frm number x image coord
            #
            pose_3d.pop(37)
            self.pose_2d.pop(37)
            self.images.pop(37)
            self.gs_start, self.gs_end = estimate_start_end(pose_3d)
            posesmoothing = estimate_smooth_pose(self.gs_start, self.gs_end)
            #
            processed_pose_3d, focus_time = posesmoothing.pose_postprocess( pose_3d )
            # computing sampling ratio
            self.focus_time = np.sort(focus_time)
            self.upsampling_rate = int((len(processed_pose_3d['joint_0'][0])-self.gs_start)/(self.gs_end  - self.gs_start))

            # processed_pose_3d # dictionary: joint name : coords x num frame
            # hard part: visulization
            #
            self.joints_3d = posesmoothing.joint_constrain( processed_pose_3d )
            # print('starting visualization!')
            self.vis = visualization('test_video')
        else:
            print('No video file is selected!')
            sys.exit()


    # ------------------------------------ < Required Functions > --------------------------
    def keep_trace(self, state):
        if state == QtCore.Qt.Checked:
            self.keep_tr = True
        else:
            self.keep_tr = False

    def clickBox_0(self, state):
        if state == QtCore.Qt.Checked:
            self.state.update({'0': 1})
            # print('stomach checked')
        else:
            self.state.update({'0': 0})
            # print('stomach unchecked')

    def clickBox_1(self, state):
        if state == QtCore.Qt.Checked:
            self.state.update({'1': 1})
            # print('right pelvis checked')
        else:
            self.state.update({'1': 0})
            # print('right pelvis unchecked')

    def clickBox_2(self, state):
        if state == QtCore.Qt.Checked:
            self.state.update({'2':1})
            # print('right knee checked', )
        else:
            self.state.update({'2':0})
            # print('right knee unchecked')

    def clickBox_4(self, state):
        if state == QtCore.Qt.Checked:
            self.state.update({'4':1})
            # print('left pelvis checked', )
        else:
            self.state.update({'4':0})
            # print('left pelvis unchecked')

    def clickBox_5(self, state):
        if state == QtCore.Qt.Checked:
            self.state.update({'5':1})
            # print('left knee checked', )
        else:
            self.state.update({'5':0})
            # print('left knee unchecked')

    def clickBox_10(self, state):
        if state == QtCore.Qt.Checked:
            self.state.update({'10': 1})
            # print('head checked', )
        else:
            self.state.update({'10':0})
            # print('head unchecked')

    def clickBox_11(self, state):
        if state == QtCore.Qt.Checked:
            self.state.update({'11': 1})
            # print('left shoulder checked', )
        else:
            self.state.update({'11': 0})
            # print('left shoulder unchecked')

    def clickBox_12(self, state):
        if state == QtCore.Qt.Checked:
            self.state.update({'12': 1})
            # print('left elbow checked', )
        else:
            self.state.update({'12': 0})
            # print('left elbow unchecked')

    def clickBox_13(self, state):
        if state == QtCore.Qt.Checked:
            self.state.update({'13': 1})
            # print('left wrist checked', )
        else:
            self.state.update({'13': 0})
            # print('left wrist unchecked')

    def clickBox_14(self, state):
        if state == QtCore.Qt.Checked:
            self.state.update({'14': 1})
            # print('right shoulder checked', )
        else:
            self.state.update({'14': 0})
            # print('right shoulder unchecked')

    def clickBox_15(self, state):
        if state == QtCore.Qt.Checked:
            self.state.update({'15': 1})
            # print('right elbow checked', )
        else:
            self.state.update({'15': 0})
            # print('right elbow unchecked')

    def clickBox_16(self, state):
        if state == QtCore.Qt.Checked:
            self.state.update({'16':1})
            # print('right wrist checked', )
        else:
            self.state.update({'16': 0})
            # print('right wrist unchecked')
    #

    def _update_canvas_1(self):
        # try:
        #     print(' convas 1... ', self.text_)
        # except:
        #     print('text is set -canvas 1 - ' , )
        self._dynamic_ax_1.clear()
        # Shift the sinusoid as a function of time.
        pos2d_img = self.vis.vis_2d(self.images[self.img_cnt], self.pose_2d[self.img_cnt])

        self._dynamic_ax_1.imshow(pos2d_img)  #plot(t, np.sin(t + time.time()))
        # managing the frame representation
        self.img_cnt = int( self.focus_time[self.pose_cnt] )
        # if self.pose_cnt > self.gs_start:
        #     self.img_cnt = int(self.pose_cnt/self.upsampling_rate)
        # elif self.img_cnt > self.gs_end:
        #     self.img_cnt = (self.img_cnt + 1) % self.gs_end
        # else:
        #     self.img_cnt = self.pose_cnt
        self._dynamic_ax_1.figure.canvas.draw()
        # pose
        self._dynamic_ax_2.clear()
        # sending signal to denote which joints to show ---
        # print('signal --- ', self.signal[0])
        self.vis.vis_3d(self.joints_3d[ self.pose_cnt ], self._dynamic_ax_2, self.signal, self.pose_cnt, self.keep_tr)
        #
        self.pose_cnt = (self.pose_cnt + 1) % ((self.gs_end - self.gs_start)*self.upsampling_rate + self.gs_start)
        self._dynamic_ax_2.figure.canvas.draw()


    def _update_canvas_2(self):
        # try:
        #     print('canvas 2 ... ', self.text_)
        # except:
        #     print('text is set -canvas 2 - ' , )
        # image
        self._dynamic_ax_1.clear()
        # Shift the sinusoid as a function of time.
        pos2d_img = self.vis.vis_2d(self.images[self.img_cnt], self.pose_2d[self.img_cnt])

        self._dynamic_ax_1.imshow(pos2d_img)  #plot(t, np.sin(t + time.time()))
        # managing the frame representation
        self.img_cnt = int( self.focus_time[self.pose_cnt] )
        # if self.pose_cnt > self.gs_start:
        #     self.img_cnt = int(self.pose_cnt/self.upsampling_rate)
        # elif self.img_cnt > self.gs_end:
        #     self.img_cnt = (self.img_cnt + 1) % self.gs_end
        # else:
        #     self.img_cnt = self.pose_cnt
        self._dynamic_ax_1.figure.canvas.draw()
        # pose
        self._dynamic_ax_2.clear()
        # sending signal to denote which joints to show ---
        # print('signal --- ', self.signal[0])
        self.vis.vis_3d(self.joints_3d[ self.pose_cnt ], self._dynamic_ax_2, self.signal, self.pose_cnt, self.keep_tr)
        #
        self.pose_cnt = (self.pose_cnt + 1) % ((self.gs_end - self.gs_start)*self.upsampling_rate + self.gs_start)
        self._dynamic_ax_2.figure.canvas.draw()

    def changefig(self, item):

        self.text_ = item.text()

        self.rmmpl()
        self._dynamic_ax_2.remove()
        self._dynamic_ax_1.remove()

        self.addmpl(self.fig_dict[self.text_])



        self.img_cnt = 0
        self.pose_cnt = 0
        self.signal = []

        # print('text is set -changefig- '  )
        if self.text_ == '     Start':
            # print(' state of the order in start ', self.state['2'])
            self.signal = [-1]
            self.timer_1 = QTimer(self)
            self.timer_1.timeout.connect(self._update_canvas_1)
            self.timer_1.start()

        elif self.text_ == '     tryJoints':
            for item in list(self.state.keys()):
                if self.state[item] == 1:
                    self.signal.append(int(item)) # = [11, 12, 13, 14, 15, 16]
            self.timer_2 = QTimer(self)
            self.timer_2.timeout.connect(self._update_canvas_2)
            self.timer_2.start()

    def addfig(self, name, fig):
        # making here to read different formats
        # if not kwargs is None:
        #     for key, value in kwargs:
        self.fig_dict[name] = fig
        self.mplfigs.addItem(name)
        # print('text is set -addfig- ' )



    def addmpl(self, fig):
        rect=plt.Rectangle((0,0),1,1, transform=fig.transFigure,
                           clip_on=False, zorder=100, alpha=0.05, color="blue")
        fig.patches.extend([rect])

        # for evt, callback in fig.canvas.callbacks.callbacks.items():
        #     for cid, _ in callback.items():
        #         fig.canvas.mpl_disconnect(cid)
        self.dynamic_canvas = FigureCanvas(fig)
        self._dynamic_ax_1 = self.dynamic_canvas.figure.add_subplot(121) # two windows are important
        # self._dynamic_ax_1.view_init(elev=20., azim=-60.)
        self._dynamic_ax_2 = self.dynamic_canvas.figure.add_subplot(122, projection='3d' )
        self._dynamic_ax_2.view_init(elev=20., azim=-60.)
        self.mplvl.addWidget(self.dynamic_canvas)
        self.dynamic_canvas.draw()
        self.toolbar = NavigationToolbar(self.dynamic_canvas, self.mplwindow, coordinates=True)
        self.mplvl.addWidget(self.toolbar)
        # print('text is set -addmpl- ' , )

        # ---------------------------------------------------------------------------
        # This is the alternate toolbar placement. Susbstitute the three lines above
        # for these lines to see the different look.
        # self.toolbar = NavigationToolbar(self.canvas,
        #        self, coordinates=True)
        # self.addToolBar(self.toolbar)
        # ------------------------------------------------------------------------------
    # def refresh(self):
    #     self.mplvl.removeWidget(self.dynamic_canvas)
    #     self.dynamic_canvas.close()
    #     self.mplvl.removeWidget(self.toolbar)
    #     self.toolbar.close()
    #     try:
    #         self.timer_1.stop()
    #         self.timer_2.stop
    #     except:
    #         pass

    def rmmpl(self,):
        self.mplvl.removeWidget(self.dynamic_canvas)
        self.dynamic_canvas.close()
        self.mplvl.removeWidget(self.toolbar)
        self.toolbar.close()
        #
        try:
            self.timer_1.stop()
            self.timer_2.stop
        except:
            pass
        # print('text is set -rmmpl- ' , )

# Functionizing all of plots





if __name__ == '__main__':
    #
    # step 1: loading pose results

    # qt functions
    app = QApplication(sys.argv)
    main = Main()

    #



    # fig0 = Figure()
    # ax1f0 = fig0.add_subplot(111)
    # ax1f0.plot()

    fig1 = Figure()
    # ax1f1 = fig1.add_subplot(111)
    # ax1f1.plot(np.random.rand(5))

    fig2 = Figure()
    # ax1f2 = fig2.add_subplot(121)
    # ax1f2.plot(np.random.rand(5))
    # ax2f2 = fig2.add_subplot(122)
    # ax2f2.plot(np.random.rand(10))

    # fig3 = Figure()
    # ax1f3 = fig3.add_subplot(111)
    # ax1f3.pcolormesh(np.random.rand(20,20))

    # _timer = dynamic_canvas.new_timer(100, [(self._update_canvas, (), {})])
    # _timer.start()

    # main.addfig('z plot',       fig1)

    # main.addmpl2( fig1 )
    main.addfig('     Start', fig1)
    main.addfig('     tryJoints', fig2)
    # main.addfig('Pcolormesh',   fig3)
    main.show()
    sys.exit(app.exec_())
