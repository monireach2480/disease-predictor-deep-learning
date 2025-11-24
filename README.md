# Disease Predictor (Deep Learning & ML)

![Disease Predictor Demo](./images/image.png)

A Flask-powered medical diagnostics suite that runs three pre-trained models:
- Random Forest for heart disease risk
- ResNet50V2 CNN for pneumonia X-ray detection
- Random Forest for diabetes risk

All tools share a modern glassmorphic UI with AI-focused copy to keep the clinical experience cohesive.

---

## 🚀 Features

- Heart disease form with 13 classic UCI-style vitals and ECG inputs.
- Pneumonia X-ray upload (JPEG/PNG) with on-the-fly resizing and ResNet50V2 inference.
- Diabetes questionnaire capturing Pima indicators (BMI, insulin, pedigree function, etc.).
- Unified result screen showing predictions, confidence, and submitted inputs/images.

---

## 🛠️ Tech Stack

- Python 3.10+
- Flask 2.x
- scikit-learn (RandomForest models serialized via pickle)
- TensorFlow / Keras (ResNet50V2 `.keras` model)
- Bootstrap 5.3, Inter font, Bootstrap Icons

---

## 📁 Project Structure

```
app.py
models/
  heart.pkl
  diabetes.pkl
  resnet50v2_pneumonia.keras
static/
  css/style.css
templates/
  base.html (layout)
  index, heart, pneumonia, diabetes, results
data/
  heart.csv, dataset.txt (sample data)
uploads/ (runtime image uploads)
```

---

## 🔧 Setup & Installation

1. **Clone & enter the repo**
   ```bash
   git clone <repo-url>
   cd disease-predictor-deep-learning
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model files exist in `models/`** (heart.pkl, diabetes.pkl, resnet50v2_pneumonia.keras).

---

## ▶️ Running the App

```bash
python app.py
```

Server runs on `http://127.0.0.1:5000/`. Navigate there to access the dashboard.

---

## 📊 Usage Notes

- **Heart Disease**: All numeric fields must be within the specified ranges (e.g., `ca` 0–4). The RandomForest expects the feature order defined in `HEART_FEATURES`.
- **Pneumonia**: Upload a clear chest X-ray. Files are resized to 224×224 and normalized before ResNet inference.
- **Diabetes**: Currently scales inputs via a newly-fitted `StandardScaler`. For production, serialize your training scaler to ensure consistency.

---

## 🧪 Testing

Manual functional testing is recommended:
1. Run `python app.py`.
2. Submit representative forms for each condition.
3. Confirm result cards show prediction + confidence and that uploaded images render on the results screen.

---

## 📌 Roadmap Ideas

- Persist uploaded files and predictions for audit trails.
- Add authentication for clinical deployments.
- Export predictions as PDFs or FHIR resources.
- Integrate monitoring (Prometheus/Grafana) for model drift.

---

## 📄 License

See `LICENSE` for details.