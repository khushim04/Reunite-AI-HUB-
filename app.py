import os
import io
import torch
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from pymongo import MongoClient
from bson.objectid import ObjectId
import faiss
from werkzeug.utils import secure_filename
import time
from ai_model import (
    get_multimodal_embedding,
    get_text_embedding,
    get_image_embedding,
    FoundItemDatabase
)

# --- 1. AI Model and Database Initialization ---
# Note: The AI model classes and functions are now imported directly from 'ai_model.py'
# You should ensure your ai_model.py file contains the necessary code.

app = Flask(__name__)
CORS(app)  # Enable CORS for front-end access

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MongoDB client setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "lost_and_found")

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    found_items_collection = db.found_items
    print("Successfully connected to MongoDB.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    client = None

# Initialize and populate the FAISS index on startup
found_db = FoundItemDatabase()
if client:
    all_found_items = list(found_items_collection.find({}))
    if all_found_items:
        print(f"Populating FAISS index with {len(all_found_items)} items...")
        for item in all_found_items:
            if 'embedding' in item and isinstance(item['embedding'], list):
                found_db.add_item(item['embedding'], item['_id'])
            else:
                print(f"Warning: Item {item.get('_id')} has no valid embedding. Skipping.")
        print("FAISS index populated.")
    else:
        print("No items found in the database. Starting with an empty index.")

def get_full_item_details(item_ids_with_similarity):
    """Retrieves full item details from MongoDB based on a list of IDs."""
    match_details = []
    for item_id, similarity in item_ids_with_similarity:
        if isinstance(item_id, ObjectId):
            item = found_items_collection.find_one({"_id": item_id})
            if item:
                del item['embedding']
                item['_id'] = str(item['_id'])
                # Convert numpy.float32 to a standard Python float
                item['similarity'] = float(round(similarity, 2)) 
                match_details.append(item)
    return match_details

# --- 2. API Endpoints ---

@app.route('/found_item', methods=['POST'])
def add_found_item():
    """API endpoint to add a new found item to the system."""
    if client is None:
        return jsonify({"error": "Database connection failed"}), 500

    text = request.form.get('description', '')
    item_name = request.form.get('item-name', '')
    location = request.form.get('location', '')
    contact = request.form.get('contact', '')
    image_file = request.files.get('file-upload')
    
    image_path = None
    if image_file and image_file.filename:
        filename = secure_filename(f"{int(time.time())}_{image_file.filename}")
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_path)

    image_bytes = None
    if image_path:
        with open(image_path, 'rb') as img:
            image_bytes = img.read()
            
    embedding = get_multimodal_embedding(f"{item_name} {text}", image_bytes)
    
    if embedding is None or not np.any(embedding):
        return jsonify({"error": "Failed to generate embedding for the item."}), 400

    item_doc = {
        "name": item_name,
        "description": text,
        "location": location,
        "contact": contact,
        "embedding": embedding,
        "image_url": f"{UPLOAD_FOLDER}/{filename}" if image_path else None,
        "type": "found",
        "timestamp": time.time()
    }

    result = found_items_collection.insert_one(item_doc)
    
    found_db.add_item(embedding, result.inserted_id)
    
    return jsonify({"message": "Found item added successfully!", "item_id": str(result.inserted_id)}), 201

@app.route('/lost_item/search', methods=['POST'])
def search_lost_item():
    """API endpoint to search for a lost item and find matches."""
    if client is None:
        return jsonify({"error": "Database connection failed"}), 500

    text = request.form.get('description', '')
    item_name = request.form.get('item-name', '')
    image_file = request.files.get('file-upload')
    
    image_bytes = None
    if image_file and image_file.filename:
        image_bytes = image_file.read()
    
    lost_embedding = get_multimodal_embedding(f"{item_name} {text}", image_bytes)
    
    if lost_embedding is None or not np.any(lost_embedding):
        return jsonify({"error": "Failed to generate a query embedding."}), 400

    matches = found_db.search_similar(lost_embedding)
    
    match_details = get_full_item_details(matches)

    return jsonify({"matches": match_details}), 200

@app.route('/api/found_items', methods=['GET'])
def get_recent_items():
    """API endpoint to get a list of recently found items."""
    if client is None:
        return jsonify({"error": "Database connection failed"}), 500
    
    items = list(found_items_collection.find().sort("timestamp", -1).limit(9))
    
    for item in items:
        item['_id'] = str(item['_id'])
        del item['embedding']
        
    return jsonify({"items": items}), 200

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """API endpoint to serve uploaded image files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- NEW: Serve the index.html file directly from the Flask app ---
@app.route('/')
def serve_frontend():
    """Renders the main frontend page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)