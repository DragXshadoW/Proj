from flask import Flask, request, jsonify, send_file
from transformers import pipeline
import sqlite3
import speech_recognition as sr
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import tempfile
import os
from fpdf import FPDF

app = Flask(__name__)

# Load AI text generation model (DistilGPT2)
description_generator = pipeline("text-generation", model="distilgpt2")

# SQLite database setup
DB_FILE = "catalog.db"

def init_db():
    """Initialize the database with a proper schema."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Drop existing table for development simplicity; use migrations in production
    c.execute('DROP TABLE IF EXISTS catalog')
    c.execute('''
        CREATE TABLE catalog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product TEXT NOT NULL,
            quantity REAL CHECK (quantity >= 0),
            price REAL CHECK (price >= 0),
            unit TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Load TFLite MobileNet model for image classification
interpreter = tf.lite.Interpreter(model_path="mobilenet_v2_1.0_224.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load MobileNet labels
with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def preprocess_image(image_bytes):
    """Preprocess image for MobileNet prediction."""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))
    input_data = np.expand_dims(image, axis=0)
    input_data = np.array(input_data, dtype=np.float32)
    input_data = (input_data / 127.5) - 1.0  # Normalize to [-1, 1]
    return input_data

def generate_description(product):
    """Generate a product description using DistilGPT2."""
    prompt = f"Describe the product: {product}."
    response = description_generator(prompt, max_length=50, num_return_sequences=1)
    desc = response[0]['generated_text'].replace(prompt, "").strip()
    return desc

def parse_input(raw_text):
    """Parse raw text input into product, quantity, price, and unit."""
    words = raw_text.lower().split()
    product = []
    quantity = None
    price = None
    unit = None

    units = ["kg", "g", "litre", "l", "pcs", "piece", "pieces"]

    i = 0
    while i < len(words):
        if words[i].isdigit():
            if i + 1 < len(words) and words[i + 1] in units:
                quantity = words[i]
                unit = words[i + 1]
                i += 2
            else:
                if price is None:
                    price = words[i]
                i += 1
        else:
            product.append(words[i])
            i += 1

    product_name = " ".join(product) if product else "Unknown"
    quantity = quantity if quantity else "Unknown"
    price = price if price else "Unknown"
    unit = unit if unit else "unit"

    return product_name, quantity, price, unit

@app.route("/add", methods=["POST"])
def add_product():
    """Add a product to the catalog from text input."""
    data = request.json
    raw_input = data.get("input")

    if not raw_input:
        return jsonify({"error": "No input provided."}), 400

    try:
        product, quantity, price, unit = parse_input(raw_input)
        description = generate_description(product)
    except Exception as e:
        return jsonify({"error": f"Failed to process input: {str(e)}"}), 500

    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO catalog (product, quantity, price, unit, description) VALUES (?, ?, ?, ?, ?)",
                  (product, float(quantity) if quantity != "Unknown" else None,
                   float(price) if price != "Unknown" else None, unit, description))
        conn.commit()
        conn.close()
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500

    return jsonify({
        "product": product,
        "quantity": quantity,
        "price": price,
        "unit": unit,
        "description": description
    })

@app.route("/catalog", methods=["GET"])
def get_catalog():
    """Retrieve the entire product catalog."""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT id, product, quantity, price, unit, description, created_at FROM catalog")
        rows = c.fetchall()
        conn.close()

        catalog = [
            {
                "id": row[0],
                "product": row[1],
                "quantity": row[2],
                "price": row[3],
                "unit": row[4],
                "description": row[5],
                "created_at": row[6]
            } for row in rows
        ]
        return jsonify(catalog)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch catalog: {str(e)}"}), 500

@app.route("/clear", methods=["POST"])
def clear_catalog():
    """Clear all entries from the catalog."""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM catalog")
        conn.commit()
        conn.close()
        return jsonify({"status": "Catalog cleared."})
    except Exception as e:
        return jsonify({"error": f"Failed to clear catalog: {str(e)}"}), 500

@app.route("/export/pdf", methods=["GET"])
def export_pdf():
    """Export the catalog as a PDF file."""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT product, quantity, price, unit, description FROM catalog")
        rows = c.fetchall()
        conn.close()

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Product Catalog", ln=1, align="C")

        for row in rows:
            pdf.multi_cell(0, 10, txt=f"Product: {row[0]}\nQty: {row[1]}\nPrice: {row[2]}\nUnit: {row[3]}\nDesc: {row[4]}\n----------------------")

        pdf_output = pdf.output(dest='S').encode('latin1')
        return send_file(io.BytesIO(pdf_output), as_attachment=True, download_name="catalog_export.pdf", mimetype="application/pdf")
    except Exception as e:
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500

@app.route("/voice", methods=["POST"])
def voice_to_text():
    """Convert uploaded audio to text using speech recognition."""
    recognizer = sr.Recognizer()
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio uploaded."}), 400

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    try:
        with sr.AudioFile(temp_audio_path) as source:
'Interpreting audio file as WAV'
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return jsonify({"transcription": text})
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio."}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Speech recognition failed: {e}"}), 500
    finally:
        os.remove(temp_audio_path)

@app.route('/predict-image', methods=['POST'])
def predict_image():
    """Predict product name from an uploaded image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']
    image_bytes = file.read()

    try:
        input_data = preprocess_image(image_bytes)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = np.argmax(output_data)
        confidence = float(np.max(output_data))
        predicted_label = labels[predicted_index]
        return jsonify({
            'product': predicted_label,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': f'Image prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)