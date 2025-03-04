from fastapi import FastAPI, HTTPException, Depends
import joblib
import shap
import numpy as np
import uvicorn
from pydantic import BaseModel
from typing import List
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Initialize FastAPI app
app = FastAPI(title="ML Model API with RAG & SHAP", version="1.0")

# Dependency Injection for Model and Index Loading
def get_model():
    return joblib.load("model.pkl")

def get_explainer(model=Depends(get_model)):
    return shap.Explainer(model)

def get_faiss_index():
    index = faiss.read_index("vector_db.index")
    with open("vector_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def get_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# API Request Model
class ModelInput(BaseModel):
    features: List[str]  # Accepts text input for embedding

@app.get("/health")
async def health_check():
    return {"status": "API is running"}

@app.post("/predict")
async def predict(
    input_data: ModelInput,
    model=Depends(get_model),
    explainer=Depends(get_explainer),
    index_metadata=Depends(get_faiss_index),
    embedding_model=Depends(get_embedding_model)
):
    try:
        index, metadata = index_metadata
        # Convert input text to embedding
        query_vector = embedding_model.encode(input_data.features, convert_to_numpy=True).reshape(1, -1).astype("float32")
        
        # Retrieve relevant documents from FAISS
        distances, indices = index.search(query_vector, k=3)
        retrieved_docs = [metadata[idx] for idx in indices[0] if idx < len(metadata)]
        
        # Convert input to NumPy array for prediction
        features_array = np.array(query_vector).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        # Compute SHAP values
        shap_values = explainer(features_array)
        explanation = shap_values.values.tolist()
        
        return {
            "prediction": prediction,
            "shap_values": explanation,
            "retrieved_documents": retrieved_docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
