
import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QFileDialog,
    QMessageBox
)
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtCore import Qt
import pytesseract
from PIL import Image


class OCRApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.image_path = None
        self.init_ui()

    def init_ui(self) -> None:
        self.setWindowTitle('Распознавание текста с изображений')
        self.setGeometry(100, 100, 1000, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        title = QLabel('OCR - Распознавание текста')
        title.setFont(QFont('Arial', 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        self.load_btn = QPushButton('Загрузить изображение')
        self.load_btn.setFont(QFont('Arial', 11))
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.load_btn.clicked.connect(self.load_image)
        main_layout.addWidget(self.load_btn)

        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        image_layout = QVBoxLayout()
        image_label = QLabel('Изображение:')
        image_label.setFont(QFont('Arial', 12, QFont.Bold))
        image_layout.addWidget(image_label)

        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setText('Изображение не загружено')
        image_layout.addWidget(self.image_display)

        content_layout.addLayout(image_layout)

        text_layout = QVBoxLayout()
        text_label = QLabel('Распознанный текст:')
        text_label.setFont(QFont('Arial', 12, QFont.Bold))
        text_layout.addWidget(text_label)

        self.text_display = QTextEdit()
        self.text_display.setFont(QFont('Arial', 10))
        self.text_display.setReadOnly(True)
        self.text_display.setPlaceholderText(
            'Текст появится здесь после распознавания...')
        text_layout.addWidget(self.text_display)

        content_layout.addLayout(text_layout)

        self.recognize_btn = QPushButton('Распознать текст')
        self.recognize_btn.setFont(QFont('Arial', 11))
        self.recognize_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.recognize_btn.setEnabled(False)
        self.recognize_btn.clicked.connect(self.recognize_text)
        main_layout.addWidget(self.recognize_btn)

        self.statusBar().showMessage('Готов к работе')

    def load_image(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            'Выберите изображение',
            '',
            'Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)'
        )

        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)

            scaled_pixmap = pixmap.scaled(
                400, 400,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.image_display.setPixmap(scaled_pixmap)
            self.recognize_btn.setEnabled(True)
            self.text_display.clear()
            self.statusBar().showMessage(f'Загружено: {file_name}')

    def recognize_text(self) -> None:
        if not self.image_path:
            QMessageBox.warning(
                self, 'Ошибка', 'Сначала загрузите изображение!')
            return

        try:
            self.statusBar().showMessage('Распознавание текста...')
            self.recognize_btn.setEnabled(False)
            QApplication.processEvents()

            image = Image.open(self.image_path)

            text = pytesseract.image_to_string(image, lang='rus+eng')

            if text.strip():
                self.text_display.setText(text)
                self.statusBar().showMessage('Текст успешно распознан!')
            else:
                self.text_display.setText('Текст не обнаружен на изображении.')
                self.statusBar().showMessage('Текст не найден')

        except Exception as e:
            QMessageBox.critical(
                self,
                'Ошибка',
                f'Произошла ошибка при распознавании:\n{str(e)}\n\n'
                'Убедитесь, что Tesseract-OCR установлен правильно.'
            )
            self.statusBar().showMessage('Ошибка распознавания')
        finally:
            self.recognize_btn.setEnabled(True)


def main() -> None:
    app = QApplication(sys.argv)
    window = OCRApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
