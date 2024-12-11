from flask import Flask, request, jsonify
import tensorflow as tf
from transformers import AutoTokenizer, TFT5ForConditionalGeneration
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import os
from google.cloud import firestore
import re
import spacy
from nltk.corpus import wordnet as wn
import random
import nltk
nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

# Konfigurasi Firestore
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\projects\question-generator\source-code\serviceAccountKey.json"
db = firestore.Client()

# Konfigurasi Path Lokal untuk Model
LOCAL_QG_MODEL_PATH = "blaxx14/t5-question-generation"  
LOCAL_TRANS_INDO_ENG_PATH = "blaxx14/indo-eng-translator"
LOCAL_TRANS_ENG_INDO_PATH = "blaxx14/eng-indo-translator"

# Pastikan file model tersedia
# if not os.path.exists(LOCAL_QG_MODEL_PATH):
#     raise FileNotFoundError(f"Model file tidak ditemukan di path: {LOCAL_QG_MODEL_PATH}")

# if not os.path.exists(LOCAL_TRANS_INDO_ENG_PATH):
#     raise FileNotFoundError(f"Model file tidak ditemukan di path: {LOCAL_TRANS_INDO_ENG_PATH}")

# if not os.path.exists(LOCAL_TRANS_ENG_INDO_PATH):
#     raise FileNotFoundError(f"Model file tidak ditemukan di path: {LOCAL_TRANS_ENG_INDO_PATH}")


"""Mengubah string menjadi dictionary"""
def parse_to_dict(input_string):
    try:
        question_part, answer_part = input_string.split('Answer: ')
        question = question_part.replace('Question: ', '').strip()  
        answer = answer_part.strip()  
        
        result_dict = {
            "Question": question,
            "Answer": answer
        }
        
        return result_dict
    
    except ValueError:
        print("Format input string tidak sesuai")
        return None


"""Mencari sinonim"""
def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


"""Membuat distractor"""
def generate_distractors(question, correct_answer):
    doc = nlp(question)
    
    keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    
    distractors = []
    
    for keyword in keywords:
        synonyms = get_synonyms(keyword)
        synonyms = [word for word in synonyms if word.lower() != correct_answer.lower()]
        distractors.extend(synonyms)
        
    distractors = random.sample(distractors, min(3, len(distractors)))
    
    return distractors

"""Load question generator model dan tokenizer"""
print("Loading model...")
model = TFT5ForConditionalGeneration.from_pretrained(LOCAL_QG_MODEL_PATH, from_pt=False)
tokenizer = AutoTokenizer.from_pretrained("t5-small")
print("Model loaded successfully.")

"""Fungsi untuk menghasilkan pertanyaan"""
def generate_question(text, max_length=4096):
    input_text = f"Generate question answer: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="tf", max_length=512, truncation=True)

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=10,
        top_k=0,
        top_p=0.8,
        temperature=1.5,
        do_sample=True,
        early_stopping=True
    )

    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_text


"""Load translator indo eng model dan tokenizer"""
print("Loading model...")
translation_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer_model = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
print("Model loaded successfully.")

"""Fungsi untuk menerjemahkan"""
def translator_indo_eng(text):
    tokenizer_model.src_lang = "id_ID"
    encoded_hi = tokenizer_model(text, return_tensors="pt")
    generated_tokens = translation_model.generate(
        **encoded_hi,
        forced_bos_token_id=tokenizer_model.lang_code_to_id["en_XX"]
    )
    output_text = tokenizer_model.batch_decode(generated_tokens, skip_special_tokens=True)
    output_text = ' '.join(output_text)
    return output_text

"""Fungsi untuk menerjemahkan"""
def translator_eng_indo(text):
    tokenizer_model.src_lang = "en_XX"
    encoded_hi = tokenizer_model(text, return_tensors="pt")
    generated_tokens = translation_model.generate(
        **encoded_hi,
        forced_bos_token_id=tokenizer_model.lang_code_to_id["id_ID"]
    )
    output_text = tokenizer_model.batch_decode(generated_tokens, skip_special_tokens=True)
    output_text = ' '.join(output_text)
    return output_text


"""Cleaning input"""
def clean_text(text):
    cleaned_text = text.replace("translit.", "")
    cleaned_text = re.sub(r'\[.*?\]', '', cleaned_text)
    return cleaned_text

def split_text_into_sentences(paragraph):
    text = clean_text(paragraph)
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return sentences

def split_into_parts(sentences, num_parts=5):
    if len(sentences) <= num_parts:
        return sentences
    else:
        part_size = len(sentences) // num_parts
        parts = [sentences[i:i + part_size] for i in range(0, len(sentences), part_size)]

        if len(parts) > num_parts:
            parts[-2].extend(parts[-1])
            parts = parts[:-1]

        return parts


@app.route('/generate-question', methods=['POST'])
def api_generate_question():
    try:
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'Text tidak boleh kosong'}), 400



        """Run cleaning input"""
        formatted_sentences = split_text_into_sentences(text)
        parts = split_into_parts(formatted_sentences)


        """Just for checking"""
        #print(parts) 


        """Run translator indo eng"""
        translated_input = []
        for i, sentence in enumerate(parts):
            combined_input = ' '.join(sentence)
            translated_input.append(translator_indo_eng(combined_input))
            print(f"Result: {translated_input[i]}")
        
        """Generate question"""
        question_list = []
        
        """versi memakai translator"""
        for i in translated_input:
            result = generate_question(i)
            print(f"Result generate question: {generate_question(i)}")
            result_dict = parse_to_dict(result)
            distractors = generate_distractors(result_dict["Question"], result_dict["Answer"])
            result_dict["Distractor"] = distractors

            result_dict['Question'] = translator_eng_indo(result_dict['Question'])
            result_dict['Answer'] = translator_eng_indo(result_dict['Answer'])
            result_dict['Distractor'] = [translator_eng_indo(d) for d in result_dict['Distractor']]
    
            # doc_ref = db.collection('questions').document()  
            # doc_ref.set({
            #     'input_text': text,
            #     'generated_question': result_dict["Question"],
            #     'answer': result_dict["Answer"],
            #     'distractor' : result_dict["Distractor"]
            # })
            question_list.append(result_dict)
        
        try:
            doc_ref = db.collection('questions').document()  # Buat dokumen utama
            doc_ref.set({
                'input_text': text  # Data input pengguna
            })
            # Simpan setiap elemen dalam subcollection
            for question in question_list:
                doc_ref.collection('generated_questions').add(question)
        except Exception as e:
             print({'error': str(e)})


        """versi tidak memakai translator"""
        # for sentence in parts:
        #     combined_input = ' '.join(sentence)
        #     result = generate_question(combined_input)
        #     # result = summarize_eng_indo(result) tunggu model dari caca
        #     result_dict = parse_to_dict(result)
        #     print(result_dict)
        #     distractors = generate_distractors(result_dict["Question"], result_dict["Answer"])
        #     result_dict["distractor"] = distractors
        #     question_list.append(result_dict)

        # print(question_list)
        return jsonify({'generated_question': question_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-questions', methods=['GET'])
def api_get_questions():
    try:
        # Ambil semua dokumen dari koleksi 'questions'
        questions = db.collection('questions').stream()

        # Konversi dokumen ke dalam format list
        results = []
        for question in questions:
            results.append(question.to_dict())

        # Return hasil sebagai JSON
        return jsonify({'questions': results}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)