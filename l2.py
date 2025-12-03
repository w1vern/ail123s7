import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                               QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt

class FeatureMatchingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SIFT Object Detector (OpenCV + PySide6)")
        self.resize(1000, 700)

        self.sift = cv2.SIFT_create()
        self.reference_image = None
        self.ref_kp = None  
        self.ref_des = None
        
        self.cap = None 
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        
        self.btn_load_ref = QPushButton("1. Загрузить эталон (Картинка)")
        self.btn_load_ref.clicked.connect(self.load_reference_image)
        
        self.btn_video_file = QPushButton("2. Открыть видео файл")
        self.btn_video_file.clicked.connect(self.open_video_file)

        self.btn_camera = QPushButton("2. Включить камеру")
        self.btn_camera.clicked.connect(self.start_camera)

        self.btn_stop = QPushButton("Стоп / Сброс")
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_stop.setStyleSheet("background-color: #ffcccc;")

        btn_layout.addWidget(self.btn_load_ref)
        btn_layout.addWidget(self.btn_video_file)
        btn_layout.addWidget(self.btn_camera)
        btn_layout.addWidget(self.btn_stop)

        self.image_label = QLabel("Загрузите эталонное изображение, затем выберите источник видео.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed gray; background-color: #f0f0f0;")
        self.image_label.setScaledContents(True)

        layout.addLayout(btn_layout)
        layout.addWidget(self.image_label)
        main_widget.setLayout(layout)

    def load_reference_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите изображение объекта", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                QMessageBox.critical(self, "Ошибка", "Не удалось загрузить изображение.")
                return

            h, w, c = img.shape
            if w > 640:
                scale = 640 / w
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

            self.reference_image = img
            self.ref_kp, self.ref_des = self.sift.detectAndCompute(self.reference_image, None)
            
            self.image_label.setText(f"Эталон загружен.\nНайдено точек: {len(self.ref_kp)}\nТеперь выберите видео.")
            
    def start_camera(self):
        if self.reference_image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение-эталон!")
            return
        self.stop_video()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Ошибка", "Не удалось открыть камеру.")
            return
        self.timer.start(30)

    def open_video_file(self):
        if self.reference_image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение-эталон!")
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if file_path:
            self.stop_video()
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Ошибка", "Не удалось открыть видеофайл.")
                return
            self.timer.start(30)

    def stop_video(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.image_label.setText("Видео остановлено.")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            return

        max_dim = 800
        h, w = frame.shape[:2]
        if w > max_dim:
            scale = max_dim / w
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp_frame, des_frame = self.sift.detectAndCompute(frame_gray, None)

        if des_frame is None or len(kp_frame) < 2:
            self.display_image(frame)
            return

        matches = self.matcher.knnMatch(self.ref_des, des_frame, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        MIN_MATCH_COUNT = 10
        final_img = None
        
        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                h_ref, w_ref, c = self.reference_image.shape
                pts = np.float32([[0, 0], [0, h_ref - 1], [w_ref - 1, h_ref - 1], [w_ref - 1, 0]]).reshape(-1, 1, 2)
                
                dst = cv2.perspectiveTransform(pts, M)

                frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

        final_img = cv2.drawMatches(
            self.reference_image, self.ref_kp,
            frame, kp_frame,
            good_matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        self.display_image(final_img)

    def display_image(self, img):
        """Конвертация OpenCV BGR -> Qt Pixmap и вывод на экран"""
        if len(img.shape) == 3:
            h, w, ch = img.shape
            bytes_per_line = ch * w
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            h, w = img.shape
            qt_image = QImage(img.data, w, h, w, QImage.Format_Grayscale8)

        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FeatureMatchingApp()
    window.show()
    sys.exit(app.exec())