from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_data/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and preprocessors
model = load_model('model/cnn_model.h5')
scaler_mean = np.load('model/scaler.npy')
scaler_std = np.load('model/scaler_std.npy')
label_classes = np.load('model/label_classes.npy', allow_pickle=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    if uploaded_file and uploaded_file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(filepath)

        # Load and process CSV
        df = pd.read_csv(filepath)
        X_input = df.drop('Machining_Process', axis=1).values
        X_input = (X_input - scaler_mean) / scaler_std
        X_input = X_input.reshape(X_input.shape[0], X_input.shape[1], 1)

        # Predict
        predictions = model.predict(X_input)
        predicted_labels = label_classes[np.argmax(predictions, axis=1)]

        return render_template('index.html', predictions=predicted_labels.tolist())

    return "Invalid or missing file."

if __name__ == '__main__':
    app.run(debug=True)
