# movie_recommender_api.py
# A free API for movie recommendations using FastAPI and DistilBERT with JSON input

import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Set up NLTK (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Create the FastAPI app
app = FastAPI(title="Movie Recommender API")

# Define JSON input structures
class UserEntry(BaseModel):
    user_id: int
    movie_id: int
    rating: Optional[float] = None  # Rating can be missing or null

class MovieEntry(BaseModel):
    movie_id: int
    title: str
    genres: str

# Reusable functions
def setup_distilbert():
    """Load DistilBERT tools."""
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    return tokenizer, model

def preprocess_text(text):
    """Clean text for DistilBERT."""
    text = text.replace('|', ' ')
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    cleaned = [word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in cleaned])

def get_embedding(text, tokenizer, model):
    """Turn text into numbers."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

def preprocess_movies(movie_data):
    """Clean all movie texts."""
    movie_data['processed'] = (movie_data['title'] + ' ' + movie_data['genres']).apply(preprocess_text)
    return movie_data

def precompute_movie_embeddings(movie_data, tokenizer, model):
    """Make number piles for all movies."""
    embeddings = np.array([get_embedding(text, tokenizer, model) for text in movie_data['processed']])
    return embeddings, movie_data['movie_id'].tolist()

def build_weighted_profile(user_id, user_data, movie_data, tokenizer, model):
    """Make user's taste pile."""
    user_history = user_data[user_data['user_id'] == user_id]
    user_movies = pd.merge(user_history, movie_data, on='movie_id')
    if user_movies.empty:
        raise ValueError(f"No watch history for user {user_id}")
    embeddings = []
    ratings = user_movies['rating'].fillna(3.0).tolist()  # Default 3.0 if no rating
    for text in user_movies['processed']:
        emb = get_embedding(text, tokenizer, model)
        embeddings.append(emb)
    embeddings = np.array(embeddings)
    weights = np.array(ratings) / sum(ratings)
    profile_emb = np.average(embeddings, axis=0, weights=weights)
    return profile_emb, user_movies['movie_id'].tolist()

def recommend_movies(user_id, user_data, movie_data, movie_embeddings, movie_ids, tokenizer, model, top_n=5):
    """Find new movies."""
    profile_emb, watched_ids = build_weighted_profile(user_id, user_data, movie_data, tokenizer, model)
    unseen_mask = ~np.isin(movie_ids, watched_ids)
    unseen_embs = movie_embeddings[unseen_mask]
    unseen_ids = np.array(movie_ids)[unseen_mask]
    if len(unseen_ids) < top_n:
        raise ValueError("Not enough unseen movies to recommend")
    similarities = cosine_similarity([profile_emb], unseen_embs)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_movie_ids = unseen_ids[top_indices]
    rec_df = movie_data[movie_data['movie_id'].isin(top_movie_ids)][['title', 'genres']]
    rec_df = rec_df.reset_index(drop=True)
    rec_df.insert(0, 'rank', range(1, len(rec_df) + 1))
    return rec_df.to_dict(orient='records')

# API endpoint for JSON input
@app.post("/recommend", response_model=List[dict])
async def get_recommendations(user_id: int, user_data: List[UserEntry], movie_data: List[MovieEntry]):
    """API to recommend movies from JSON data."""
    try:
        # Convert JSON to pandas DataFrames
        user_df = pd.DataFrame([entry.dict() for entry in user_data])
        movie_df = pd.DataFrame([entry.dict() for entry in movie_data])

        # Load DistilBERT
        tokenizer, model = setup_distilbert()

        # Preprocess and compute embeddings
        movie_df = preprocess_movies(movie_df)
        movie_embeddings, movie_ids = precompute_movie_embeddings(movie_df, tokenizer, model)

        # Get recommendations
        recs = recommend_movies(user_id, user_df, movie_df, movie_embeddings, movie_ids, tokenizer, model)
        return recs

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Run locally (for testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)