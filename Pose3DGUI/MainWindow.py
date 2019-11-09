# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '3dposeestimationgui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1442, 418)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(MainWindow)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame = QtWidgets.QFrame(MainWindow)
        self.frame.setMinimumSize(QtCore.QSize(1450, 400))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lbl_videoStream = QtWidgets.QLabel(self.frame)
        self.lbl_videoStream.setMinimumSize(QtCore.QSize(640, 360))
        self.lbl_videoStream.setMaximumSize(QtCore.QSize(640, 360))
        self.lbl_videoStream.setFrameShape(QtWidgets.QFrame.Panel)
        self.lbl_videoStream.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_videoStream.setObjectName("lbl_videoStream")
        self.horizontalLayout.addWidget(self.lbl_videoStream)
        self.lbl_customView = QtWidgets.QLabel(self.frame)
        self.lbl_customView.setMinimumSize(QtCore.QSize(640, 360))
        self.lbl_customView.setMaximumSize(QtCore.QSize(640, 360))
        self.lbl_customView.setFrameShape(QtWidgets.QFrame.Box)
        self.lbl_customView.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_customView.setObjectName("lbl_customView")
        self.horizontalLayout.addWidget(self.lbl_customView)
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setMinimumSize(QtCore.QSize(100, 0))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_3 = QtWidgets.QFrame(self.frame_2)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.groupBox = QtWidgets.QGroupBox(self.frame_3)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_4.addWidget(self.groupBox)
        self.verticalLayout.addWidget(self.frame_3)
        self.pbLoadVideo = QtWidgets.QPushButton(self.frame_2)
        self.pbLoadVideo.setObjectName("pbLoadVideo")
        self.verticalLayout.addWidget(self.pbLoadVideo)
        self.horizontalLayout.addWidget(self.frame_2)
        self.verticalLayout_2.addWidget(self.frame)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "3DPoseEstimationGUI"))
        self.lbl_videoStream.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:36pt; font-weight:600; font-style:italic;\">VideoStream</span></p></body></html>"))
        self.lbl_customView.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:36pt; font-weight:600; font-style:italic;\">CustomView</span></p></body></html>"))
        self.groupBox.setTitle(_translate("MainWindow", "Joints"))
        self.pbLoadVideo.setText(_translate("MainWindow", " Open "))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QWidget()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

