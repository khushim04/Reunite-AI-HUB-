import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import io
import numpy as np
import faiss

# Check for GPU and set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
# Note: These are loaded once when the module is imported to save memory and time
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

# --- Vector Database Class ---
class FoundItemDatabase:
    """
    Manages storage and retrieval of found item embeddings using a FAISS index
    for efficient similarity search.
    """
    def __init__(self):
        # The embedding dimension from the text model (768 for all-MiniLM-L6-v2)
        embedding_dim = text_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.item_ids = []

    def add_item(self, item_embedding, item_id):
        """Adds a single item embedding to the database."""
        if item_embedding is None or not np.any(item_embedding):
            print(f"Warning: Skipping item {item_id} due to invalid embedding.")
            return False
        
        # Ensure embedding is a numpy array of type float32 for FAISS
        embedding_array = np.array([item_embedding]).astype('float32')
        self.index.add(embedding_array)
        self.item_ids.append(item_id)
        return True

    def search_similar(self, query_embedding, k=10):
        """
        Searches the database for the k most similar items to the query.
        Returns a list of (item_id, similarity_percentage).
        """
        if self.index.ntotal == 0:
            return []
            
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Faiss returns -1 for empty slots
                item_id = self.item_ids[idx]
                distance = distances[0][i]
                # Convert L2 distance to a similarity score
                similarity = 1 / (1 + distance) 
                similarity_percentage = max(0, min(100, similarity * 100))
                results.append((item_id, similarity_percentage))
        return results

# --- Embedding Functions ---
def get_text_embedding(text: str):
    """
    Generates a vector embedding for a given text. Handles empty or invalid input.
    """
    if not text or not isinstance(text, str):
        print("Warning: Empty or invalid text input. Returning zero vector.")
        return np.zeros(text_model.get_sentence_embedding_dimension()).tolist()
    embedding = text_model.encode(text, convert_to_tensor=True)
    return embedding.cpu().tolist()

def get_image_embedding(image_bytes: bytes):
    """Generates a vector embedding for a given image using CLIP."""
    if image_bytes is None:
        return None
        
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        return image_features.cpu().squeeze().tolist()
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def get_multimodal_embedding(text: str, image_bytes: bytes = None):
    """
    Generates a combined embedding. Prioritizes a combined vector if both
    text and a valid image are available.
    """
    text_embedding = get_text_embedding(text)
    image_embedding = get_image_embedding(image_bytes)
    
    # Check if both embeddings are valid before combining
    if image_embedding is not None and np.any(image_embedding):
        # We need to ensure the dimensions are the same for combination.
        # This is a common issue with different models.
        if len(text_embedding) == len(image_embedding):
            # Combine embeddings using a weighted average
            combined_embedding = np.average(
                [text_embedding, image_embedding], 
                axis=0, 
                weights=[0.4, 0.6]
            )
            return combined_embedding.tolist()
        else:
            print("Warning: Text and image embedding dimensions do not match. Falling back to text-only.")
    
    # Fallback to text-only embedding
    return text_embedding