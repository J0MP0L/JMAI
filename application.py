from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# สร้าง Flask app
app = Flask(__name__)

# โหลดโมเดล MNIST
model = load_model('mnist_model.h5')

# ฟังก์ชันช่วยในการเตรียมภาพเพื่อส่งเข้าโมเดล
def preprocess_image(image, target_size=(28, 28)):
    image = image.convert("L")  # แปลงเป็น grayscale
    image = image.resize(target_size)  # ปรับขนาดภาพ
    image = np.array(image) / 255.0  # ปรับขนาดค่า pixel ให้อยู่ในช่วง 0-1
    image = np.expand_dims(image, axis=0)  # เพิ่ม dimension ให้กับข้อมูล
    return image

# Route สำหรับหน้าแรก
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return "No file uploaded", 400

        # อ่านรูปภาพและเตรียมข้อมูล
        img = Image.open(io.BytesIO(file.read()))
        img = preprocess_image(img)

        # ทำนายผลด้วยโมเดล
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)

        return jsonify({'predicted_class': int(predicted_class)})

    return render_template('index.html')

# เริ่ม Flask server
if __name__ == "__main__":
    app.run(host='0.0.0.0')
