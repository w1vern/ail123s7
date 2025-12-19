
import os
import sys
import cv2
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QWidget
)


class Thread(QThread):
    updateFrame = Signal(QImage)

    def __init__(self, parent=None) -> None:
        QThread.__init__(self, parent)
        self.face_cascade_file = os.path.join(
            cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        self.eye_cascade_file = os.path.join(
            cv2.data.haarcascades, 'haarcascade_eye.xml')
        self.status = True
        self.cap = None

    def run(self) -> None:
        self.cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(self.face_cascade_file)
        eye_cascade = cv2.CascadeClassifier(self.eye_cascade_file)

        while self.status:
            ret, frame = self.cap.read()
            if not ret:
                continue
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                roi_gray = gray_frame[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                eyes = eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey),
                                  (ex + ew, ey + eh), (255, 0, 0), 2)

            color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w, ch = color_frame.shape
            img = QImage(color_frame.data, w, h, ch *
                         w, QImage.Format.Format_RGB888)
            scaled_img = img.scaled(
                640, 480, Qt.AspectRatioMode.KeepAspectRatio)

            self.updateFrame.emit(scaled_img)

    def stop(self):
        self.status = False
        if self.cap:
            self.cap.release()
        self.quit()
        self.wait()


class Window(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Face and Eye Detection")
        self.setGeometry(0, 0, 800, 500)

        self.menu = self.menuBar()
        self.menu_file = self.menu.addMenu("File")
        exit_action = QAction("Exit", self, triggered=QApplication.quit)
        self.menu_file.addAction(exit_action)

        self.menu_about = self.menu.addMenu("&About")
        about = QAction("About Qt", self,
                        shortcut=QKeySequence(
                            QKeySequence.StandardKey.HelpContents),
                        triggered=QApplication.aboutQt)
        self.menu_about.addAction(about)

        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)

        self.th = Thread(self)
        self.th.finished.connect(self.close)
        self.th.updateFrame.connect(self.setImage)

        buttons_layout = QHBoxLayout()
        self.button1 = QPushButton("Start")
        self.button2 = QPushButton("Stop/Close")
        self.button1.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.button2.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        buttons_layout.addWidget(self.button2)
        buttons_layout.addWidget(self.button1)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(buttons_layout)

        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.button1.clicked.connect(self.start)
        self.button2.clicked.connect(self.kill_thread)
        self.button2.setEnabled(False)

    @Slot()
    def kill_thread(self) -> None:
        print("Finishing...")
        self.button2.setEnabled(False)
        self.button1.setEnabled(True)
        self.th.stop()
        cv2.destroyAllWindows()

    @Slot()
    def start(self) -> None:
        print("Starting...")
        self.button2.setEnabled(True)
        self.button1.setEnabled(False)
        self.th.start()

    @Slot(QImage)
    def setImage(self, image) -> None:
        self.label.setPixmap(QPixmap.fromImage(image))


if __name__ == "__main__":
    app = QApplication()
    w = Window()
    w.show()
    sys.exit(app.exec())
