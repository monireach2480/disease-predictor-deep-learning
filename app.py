import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --------------------------
# Load Models
# --------------------------

def load_heart_model():
    try:
        with open('models/heart.pkl', 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Error loading heart model: {e}")
        return None

def load_pneumonia_model():
    try:
        return load_model('models/resnet50v2_pneumonia.keras')
    except Exception as e:
        print(f"Error loading pneumonia model: {e}")
        return None

heart_model = load_heart_model()
pneumonia_model = load_pneumonia_model()

# FIXED FEATURE ORDER
HEART_FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal'
]

# --------------------------
# Page Routes
# --------------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/heart')
def heart_disease():
    return render_template('heart.html')

@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

# --------------------------
# HEART DISEASE PREDICTION
# --------------------------

@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    if heart_model is None:
        return jsonify({'error': 'Heart disease model not available'})

    try:
        # Read form data
        data = {
            'age': float(request.form['age']),
            'sex': int(request.form['sex']),
            'cp': int(request.form['cp']),
            'trestbps': float(request.form['trestbps']),
            'chol': float(request.form['chol']),
            'fbs': int(request.form['fbs']),
            'restecg': int(request.form['restecg']),
            'thalach': float(request.form['thalach']),
            'exang': int(request.form['exang']),
            'oldpeak': float(request.form['oldpeak']),
            'slope': int(request.form['slope']),
            'ca': int(request.form['ca']),
            'thal': int(request.form['thal'])
        }

        # FIX: ensure correct order for model
        df = pd.DataFrame([data], columns=HEART_FEATURES)

        # Predict class (0 = no disease, 1 = disease)
        prediction = int(heart_model.predict(df)[0])

        # Predict probability (if supported)
        try:
            proba = heart_model.predict_proba(df)[0]
            confidence = float(proba[1] if prediction == 1 else proba[0])
        except:
            confidence = 1.0  # fallback

        result = {
            'prediction': prediction,
            'probability': confidence,
            'message': 'High risk of heart disease' if prediction == 1 else 'Low risk of heart disease'
        }

        return render_template(
            'results.html',
            result=result,
            model_type='Heart Disease',
            input_data=data
        )

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})

# --------------------------
# PNEUMONIA
# --------------------------

@app.route('/predict/pneumonia', methods=['POST'])
def predict_pneumonia():
    if pneumonia_model is None:
        return jsonify({'error': 'Pneumonia model not available'})
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'})
        
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        prediction = pneumonia_model.predict(img_array)[0][0]
        predicted_class = 1 if prediction > 0.5 else 0
        confidence = prediction if predicted_class == 1 else 1 - prediction
        
        result = {
            'prediction': predicted_class,
            'probability': float(confidence),
            'message': 'Pneumonia detected' if predicted_class == 1 else 'Normal (No Pneumonia)'
        }

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return render_template('results.html', 
                               result=result, 
                               model_type='Pneumonia',
                               image_data=img_str)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})

# --------------------------
# DIABETES (placeholder)
# --------------------------

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    return render_template('results.html', 
                           result={'message': 'Diabetes model is not yet implemented. Coming soon!'},
                           model_type='Diabetes')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
