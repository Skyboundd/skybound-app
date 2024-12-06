from flask import Flask, request, jsonify
import tensorflow as tf
from transformers import AutoTokenizer, TFT5ForConditionalGeneration
import os
from google.cloud import firestore

# Inisialisasi Flask
app = Flask(__name__)

# Konfigurasi Firestore
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "backend-443906-8a21a207f878.json"
db = firestore.Client()

# Konfigurasi Path Lokal untuk Model
LOCAL_MODEL_PATH = "model/pretrained-v4"  # Forward slash

# Pastikan file model tersedia
if not os.path.exists(LOCAL_MODEL_PATH):
    raise FileNotFoundError(f"Model file tidak ditemukan di path: {LOCAL_MODEL_PATH}")

# Load model dan tokenizer
print("Loading model...")
model = TFT5ForConditionalGeneration.from_pretrained(LOCAL_MODEL_PATH, from_pt=False)
tokenizer = AutoTokenizer.from_pretrained("t5-small")
print("Model loaded successfully.")

# Fungsi untuk menghasilkan pertanyaan
def generate_question(text, max_length=512):
    input_text = f"Generate multiple choice question: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="tf", max_length=512, truncation=True)

    # Menghasilkan teks
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=5,
        top_k=50,
        top_p=0.95,
        temperature=1.2,
        do_sample=True,
        early_stopping=True
    )

    # Decode output
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_text

@app.route('/generate-question', methods=['POST'])
def api_generate_question():
    try:
        # Ambil input dari request
        data = request.json
        text = data.get('text', '')

        # Validasi input
        if not text:
            return jsonify({'error': 'Text tidak boleh kosong'}), 400

        # Generate question
        result = generate_question(text)

        # Simpan data ke Firestore
        doc_ref = db.collection('questions').document()  # Buat dokumen baru
        doc_ref.set({
            'input_text': text,
            'generated_question': result
        })

        # Return hasil
        return jsonify({'generated_question': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
