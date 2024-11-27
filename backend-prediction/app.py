from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import os

# Nonaktifkan GPU dan abaikan log TensorFlow selain error fatal
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Inisialisasi Flask app
app = Flask(__name__)

# Konfigurasi lokasi model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../model/model_karir.h5")

# Fungsi untuk memuat model
def load_prediction_model(model_path):
    """
    Memuat model dari jalur yang ditentukan.
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File model tidak ditemukan di {model_path}.")
        model = load_model(model_path)
        print(f"Model berhasil dimuat dari: {model_path}")
        return model
    except Exception as e:
        print(f"Error saat memuat model: {e}")
        return None

# Muat model
model = load_prediction_model(MODEL_PATH)

# Daftar karir
daftar_karir = [
    'Accountant', 'Graphic Designer', 'Salesperson', 'Research Scientist', 'Teacher',
    'Architect', 'Nurse', 'Software Developer', 'Psychologist', 'Chef', 'Marketing Manager',
    'Physician', 'Artist', 'Human Resources Manager', 'Construction Engineer', 'Journalist',
    'Astronomer', 'Financial Analyst', 'Biologist', 'Event Planner', 'Real Estate Agent',
    'Environmental Scientist', 'Lawyer', 'IT Support Specialist', 'Fashion Designer',
    'Marketing Coordinator', 'Biomedical Engineer', 'Event Photographer', 'Data Analyst',
    'Pharmacist', 'Social Worker', 'Financial Planner', 'Biotechnologist', 'HR Recruiter',
    'Software Quality Assurance Tester', 'Elementary School Teacher', 'Industrial Engineer',
    'Market Research Analyst', 'Financial Auditor', 'Musician', 'Police Detective',
    'Marketing Copywriter', 'Zoologist', 'Speech Therapist', 'Mechanical Engineer',
    'Forensic Scientist', 'Social Media Manager', 'Geologist', 'Web Developer',
    'Wildlife Biologist', 'Air Traffic Controller', 'Game Developer', 'Urban Planner',
    'Financial Advisor', 'Airline Pilot', 'Environmental Engineer', 'Interior Designer',
    'Physical Therapist', 'Mechanical Designer', 'Dental Hygienist', 'Marketing Analyst',
    'Aerospace Engineer', 'Pediatric Nurse', 'Advertising Executive', 'Wildlife Conservationist',
    'IT Project Manager', 'Forestry Technician', 'Video Game Tester', 'Marriage Counselor',
    'Biomedical Researcher', 'Database Administrator', 'Public Relations Specialist',
    'Genetic Counselor', 'Market Researcher', 'Occupational Therapist', 'Electrical Engineer',
    'Investment Banker', 'Marine Biologist', 'Human Rights Lawyer', 'Database Analyst',
    'Pediatrician', 'Technical Writer', 'Forensic Psychologist', 'Product Manager',
    'Fashion Stylist', 'Speech Pathologist', 'Public Health Analyst', 'Sports Coach',
    'Insurance Underwriter', 'Chiropractor', 'Radiologic Technologist', 'Tax Accountant',
    'Quality Control Inspector', 'Rehabilitation Counselor', 'Film Director', 'Diplomat',
    'Police Officer', 'Administrative Officer', 'Tax Collector', 'Foreign Service Officer',
    'Customs and Border Protection Officer', 'Civil Engineer', 'Robotics Engineer',
    'Electronics Design Engineer'
]

# Fungsi prediksi
def prediksi_karir(input_fitur):
    """
    Prediksi berdasarkan fitur input.
    input_fitur: array numpy dengan 10 nilai.
    """
    if model is None:
        raise RuntimeError("Model belum dimuat. Periksa log kesalahan.")
    if len(input_fitur) != 10:
        raise ValueError("Input harus berupa array dengan 10 fitur.")

    # Prediksi
    prediksi_prob = model.predict(np.array([input_fitur]))[0]
    indeks_teratas = prediksi_prob.argsort()[-3:][::-1]  # Top 3 predictions
    hasil = [{"karir": daftar_karir[i], "probabilitas": float(prediksi_prob[i])} for i in indeks_teratas]
    return hasil

# Endpoint prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Data harus menyertakan 'features'."}), 400

        input_fitur = data["features"]
        if not isinstance(input_fitur, list) or len(input_fitur) != 10:
            return jsonify({"error": "Fitur harus berupa array dengan 10 nilai."}), 400

        hasil = prediksi_karir(input_fitur)
        return jsonify({"status": "success", "predictions": hasil})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Endpoint home
@app.route('/')
def home():
    return "API untuk Prediksi Karir"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
