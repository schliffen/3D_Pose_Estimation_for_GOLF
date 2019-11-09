import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtCore
from MainWindow import Ui_MainWindow

from queue import Queue
import threading
import cv2
import time

running = False
capture_thread = None
q = Queue()
fileName = ""
JNTS_NUM = 10
FPS = 30.0

def grab(fileName, queue):
    global running
    capture = cv2.VideoCapture(fileName)

    while(running):
        frame = {}
        retval, img = capture.read()

        if (retval):

            frame["img"] = img
            if queue.qsize() < 10:
                queue.put(frame)
            else:
                print(queue.qsize())
        else:
            running = False
        time.sleep(1.0/FPS)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    cbList = []
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.window_width = self.lbl_videoStream.width()
        self.window_height = self.lbl_videoStream.height()

        self.pbLoadVideo.clicked.connect(self.openFileDialog)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        vbox = QtWidgets.QVBoxLayout()
        self.groupBox.setLayout(vbox)
        for i in range(JNTS_NUM):
            cb = QtWidgets.QCheckBox("Jnt#"+str(i), self)
            cb.setChecked(True)
            vbox.addWidget(cb)
            cb.clicked.connect(self.checkBoxChecked)
            self.cbList.append(cb);

    def checkBoxChecked(self):
        print("CBox checked")
        print( self.cbList[0].isChecked())

    def openFileDialog(self):
        global running
        running = False
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*)", options=options)
        if fileName:
            print(fileName)
            self.timer.start(FPS)

            #global running
            running = True
            capture_thread = threading.Thread(target=grab, args=(fileName, q))
            capture_thread.start()

    def update_frame(self):
        if not q.empty():
            #self.startButton.setText('Camera is live')
            frame = q.get()
            img = frame["img"]

            img_height, img_width, img_colors = img.shape
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])

            if scale == 0:
                scale = 1

            #fill VideoStream
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, bpc = img.shape
            bpl = bpc * width
            image = QImage(img.data, width, height, bpl, QImage.Format_RGB888)
            pixmap = QPixmap(image)
            self.lbl_videoStream.setPixmap(pixmap)

            #fill customView
            grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            imageGray = QImage(grayImg.data, width, height, QImage.Format_Grayscale8)
            pixmapGray = QPixmap(imageGray)
            self.lbl_customView.setPixmap(pixmapGray)


    def closeEvent(self, event):
        global running
        running = False

app = QtWidgets.QApplication(sys.argv)

window = MainWindow()
window.show()
app.exec()