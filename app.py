from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model/model_cnn.h5')

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan folder upload ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class_names = ['Kucing', 'Anjing']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded!', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.0

        prediction = model.predict(img_tensor)
        
        if prediction.shape[1] > 1:
            class_index = np.argmax(prediction[0])
        else:
            class_index = int(prediction[0] > 0.5)
            
        result = class_names[class_index]
        image_url = os.path.join('uploads', file.filename).replace(os.path.sep, '/')
        
        return render_template('index.html', prediction=result, image_path=image_url)

    except Exception as e:
        # Jika ada error, tampilkan di halaman web untuk debugging
        error_message = f"Terjadi error: {e}"
        return render_template('index.html', prediction=error_message)

if __name__ == '__main__':
    app.run(debug=True)
