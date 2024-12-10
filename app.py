from flask import Flask, request, jsonify
import tensorflow as tf
from transformers import AutoTokenizer, TFT5ForConditionalGeneration, TFAutoModelForSeq2SeqLM
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
LOCAL_QG_MODEL_PATH = "C:/projects/question-generator/pretrained-v4"  
LOCAL_TRANS_INDO_ENG_PATH = "C:/projects/question-generator/translator-indo-eng"
LOCAL_TRANS_ENG_INDO_PATH = "C:/projects/question-generator/translator-eng-indo"

# Pastikan file model tersedia
if not os.path.exists(LOCAL_QG_MODEL_PATH):
    raise FileNotFoundError(f"Model file tidak ditemukan di path: {LOCAL_QG_MODEL_PATH}")

if not os.path.exists(LOCAL_TRANS_INDO_ENG_PATH):
    raise FileNotFoundError(f"Model file tidak ditemukan di path: {LOCAL_TRANS_INDO_ENG_PATH}")

if not os.path.exists(LOCAL_TRANS_ENG_INDO_PATH):
    raise FileNotFoundError(f"Model file tidak ditemukan di path: {LOCAL_TRANS_ENG_INDO_PATH}")


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
    
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)


"""Membuat distractor"""
def generate_distractors_from_text(text, correct_answer):
    doc = nlp(text)
    
    distractors = []
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB']:  # Pilih kata benda, proper noun, adjektiva, dan verba
            synonyms = get_synonyms(token.text)
            if synonyms:
                distractors.extend(synonyms)
    
    distractors = list(set(distractors))  
    if correct_answer in distractors:
        distractors.remove(correct_answer)
    
    random.shuffle(distractors)
    return distractors[:3]


"""Load question generator model dan tokenizer"""
print("Loading model...")
model = TFT5ForConditionalGeneration.from_pretrained(LOCAL_QG_MODEL_PATH, from_pt=False)
tokenizer = AutoTokenizer.from_pretrained("t5-small")
print("Model loaded successfully.")

"""Fungsi untuk menghasilkan pertanyaan"""
def generate_question(text, max_length=512):
    input_text = f"Generate multiple choice question: {text}"
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
translation_indo_eng = TFAutoModelForSeq2SeqLM.from_pretrained(LOCAL_TRANS_INDO_ENG_PATH, from_pt=False)
tokenizer_indo_eng = AutoTokenizer.from_pretrained("t5-small")
print("Model loaded successfully.")


"""Fungsi untuk menerjemahkan"""
def translator_indo_eng(text, max_length=512):
    input_text = f"translate Indonesia to English: {text}"
    input_ids = tokenizer_indo_eng.encode(input_text, return_tensors="tf", max_length=max_length, truncation=True)

    output = translation_indo_eng.generate(
        input_ids,
        max_length=max_length,
        num_beams=10,
        top_k=30,
        top_p=0.95,
        temperature=1.5,
        do_sample=True,
        early_stopping=True
    )

    output_text = tokenizer_indo_eng.decode(output[0], skip_special_tokens=True)
    return output_text


"""Load translator eng indo model dan tokenizer"""
print("Loading model...")
translation_eng_indo = TFAutoModelForSeq2SeqLM.from_pretrained(LOCAL_TRANS_ENG_INDO_PATH, from_pt=False)
tokenizer_eng_indo = AutoTokenizer.from_pretrained("t5-small")
print("Model loaded successfully.")


"""Fungsi untuk menerjemahkan"""
def translator_eng_indo(text, max_length=512):
    input_text = f"translate Indonesia to English: {text}"
    input_ids = tokenizer_eng_indo.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    output = translation_eng_indo.generate(
        input_ids,
        max_length=max_length,
        num_beams=10,
        top_k=0,
        top_p=0.8,
        temperature=1.5,
        do_sample=True,
        early_stopping=True
    )

    output_text = tokenizer_eng_indo.decode(output[0], skip_special_tokens=True)
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
        # Ambil input dari request
        data = request.json
        text = data.get('text', '')

        # Validasi input
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
        for i in translated_input:
            # print(f"Result: {generate_question(i)}")
            result = generate_question(i)
            # result = summarize_eng_indo(result) tunggu model dari caca
            result_dict = parse_to_dict(result)
            distractors = generate_distractors_from_text(i, result_dict["Answer"])
            result_dict["distractor"] = distractors
            question_list.append(result_dict)

        # print(question_list)
        return jsonify({'generated_question': question_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)