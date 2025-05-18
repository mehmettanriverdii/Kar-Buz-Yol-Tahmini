import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QFont, QPalette, QBrush,QIcon
from PyQt5.QtCore import Qt
import tensorflow as tf
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.keras")
CLASS_NAME = ["temiz", "hafif karlı", "karlı", "temizlenmiş"]
IMG_SIZE = (180, 180)
BG_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "sivas4.png")

model = tf.keras.models.load_model(MODEL_PATH)

class Window(QWidget):
    
    def __init__(self):
        super().__init__()
        
        self.setGeometry(100, 100, 500, 500)
        self.setWindowIcon(QIcon("icons.png"))
        self.setWindowTitle("Kar & Buz Tespit Masaüstü Uygulaması")
        
        self.ui()
        
        self.show()
        
        
    def ui(self):
        
        if os.path.exists(BG_IMAGE_PATH):
            palette = QPalette()
            pixmap = QPixmap(BG_IMAGE_PATH).scaled(self.size(), Qt.IgnoreAspectRatio)
            palette.setBrush(QPalette.Window, QBrush(pixmap))
            self.setPalette(palette)
        
        self.image_label = QLabel("Lütfen Resim Yükleyiniz.", self)
        self.image_label.setGeometry(160, 10, IMG_SIZE[0], IMG_SIZE[1])
        self.image_label.setStyleSheet("border: 2px solid black; color: yellow;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFont(QFont("Arial", 10, QFont.Bold))
        
        self.result_label = QLabel("", self) # sonuclar
        self.result_label.setGeometry(10, 300, 480, 50)
        self.result_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.result_label.setStyleSheet("color: white;")

        
        load_btn = QPushButton("Resim Yükle", self)
        load_btn.setGeometry(210,200,80, 30)
        load_btn.clicked.connect(self.load_image) # resim yukleme butonu

        
    def load_image(self):
        
        file_path, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Images (*.png *.jpg *.jpeg)")

        
        if not file_path:
            return 
        
        pix = QPixmap(file_path).scaled(*IMG_SIZE) # seçilen gorseli label uzerinde gosteriyoruz.
        self.image_label.setPixmap(pix)
        
        
        input_image = tf.keras.utils.load_img(file_path, target_size=(180,180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array,0)
    
        
        # tahmin 
        preds = model.predict(input_image_exp_dim)
        
        result = tf.nn.softmax(preds[0])
        
        prediction = CLASS_NAME[np.argmax(result)]
        
        confidence = np.max(result) * 100
        
        if prediction == "temiz":
            self.result_label.setText(f"Kameramıza yansıyan görüntü temizdir.\nYolda herhangi bir sorun gözükmüyor.\n(Olasılık: %{confidence:.2f})")
            
        elif prediction == "hafif karlı" or prediction == "temizlenmiş":
            self.result_label.setText(f"Yol hafif karlı gözüküyor,\nbu bir tehdit oluşturmasa da tuzlama ve temizleme işlemi yapılabilir.\n(Olasılık: %{confidence:.2f})")
            
        elif prediction == "karlı":
            self.result_label.setText(f"Yolda kameralarımız tehdit oluşturabilecek buzlanma farketmiştir,\n lütfen tuzlama ve temizleme işlemlerine başlayınız!\n(Olasılık: %{confidence:.2f})")
        
        
        

        
window = Window()
