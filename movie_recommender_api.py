# movie_recommender_api.py
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
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Download all required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')  # Added for tokenization
nltk.download('stopwords')
nltk.download('wordnet')

app = FastAPI(title="Movie Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://movie-recommender-tester.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserEntry(BaseModel):
    user_id: int
    movie_id: int
    rating: Optional[float] = None

class MovieEntry(BaseModel):
    movie_id: int
    title: str
    genres: str

class RecommendRequest(BaseModel):
    user_id: int
    user_data: List[UserEntry]
    movie_data: List[MovieEntry]

def setup_distilbert():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    return tokenizer, model

def preprocess_text(text):
    text = text.replace('|', ' ')
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    cleaned = [word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in cleaned])

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

def preprocess_movies(movie_data):
    movie_data['processed'] = (movie_data['title'] + ' ' + movie_data['genres']).apply(preprocess_text)
    return movie_data

def precompute_movie_embeddings(movie_data, tokenizer, model):
    embeddings = np.array([get_embedding(text, tokenizer, model) for text in movie_data['processed']])
    return embeddings, movie_data['movie_id'].tolist()

def build_weighted_profile(user_id, user_data, movie_data, tokenizer, model):
    user_history = user_data[user_data['user_id'] == user_id]
    user_movies = pd.merge(user_history, movie_data, on='movie_id')
    if user_movies.empty:
        raise ValueError(f"No watch history for user {user_id}")
    embeddings = []
    ratings = user_movies['rating'].fillna(3.0).tolist()
    for text in user_movies['processed']:
        emb = get_embedding(text, tokenizer, model)
        embeddings.append(emb)
    embeddings = np.array(embeddings)
    weights = np.array(ratings) / sum(ratings)
    profile_emb = np.average(embeddings, axis=0, weights=weights)
    return profile_emb, user_movies['movie_id'].tolist()

def recommend_movies(user_id, user_data, movie_data, movie_embeddings, movie_ids, tokenizer, model, top_n=5):
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

@app.post("/recommend", response_model=List[dict])
async def get_recommendations(request: RecommendRequest):
    try:
        user_df = pd.DataFrame([entry.dict() for entry in request.user_data])
        movie_df = pd.DataFrame([entry.dict() for entry in request.movie_data])
        tokenizer, model = setup_distilbert()
        movie_df = preprocess_movies(movie_df)
        movie_embeddings, movie_ids = precompute_movie_embeddings(movie_df, tokenizer, model)
        recs = recommend_movies(request.user_id, user_df, movie_df, movie_embeddings, movie_ids, tokenizer, model)
        return recs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)