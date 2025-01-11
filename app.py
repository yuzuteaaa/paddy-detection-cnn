from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename, redirect
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from controllers import chatController

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model_path = "model/detection/five-class.keras"
model = load_model(model_path)

Categories = ['layu','non-leaf', 'normal', 'tungro', 'wereng']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_to_category_folder(file_path, predicted_label):
    category_folder = os.path.join(app.config['UPLOAD_FOLDER'], predicted_label)
    if not os.path.exists(category_folder):
        os.makedirs(category_folder)
    new_file_path = os.path.join(category_folder, os.path.basename(file_path))
    os.rename(file_path, new_file_path)
    return new_file_path


def process_image(img_path):

    img = load_img(img_path, target_size=(128, 128))  # Pastikan ukuran sesuai model
    img_array = img_to_array(img) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch


    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]) * 100)

    predicted_label = Categories[predicted_class]

    categorized_file_path = save_to_category_folder(img_path, predicted_label)

    return predicted_label, confidence, categorized_file_path


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            predicted_class, confidence, categorized_file_path = process_image(file_path)
            return render_template(
                'index.html',
                uploaded_image=categorized_file_path,
                prediction=predicted_class,
                confidence=confidence
            )
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        predicted_class, confidence = process_image(file_path)
        return jsonify({
            'prediction': predicted_class,
            'confidence': float(confidence)
        })
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    if request.method == "POST":
        return chatController.chat()

if __name__ == '__main__':
    app.run(debug=True)
