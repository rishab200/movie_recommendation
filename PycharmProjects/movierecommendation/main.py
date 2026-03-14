import pickle
import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# load movies
movies = pickle.load(open('movies.pkl', 'rb'))

# generate similarity only if not exists
if not os.path.exists('similarity.pkl'):
    print("Generating similarity matrix...")
    vectorizer = TfidfVectorizer(max_features=5000)
    vectors = vectorizer.fit_transform(movies['tag']).toarray()
    similarity = cosine_similarity(vectors)
    pickle.dump(similarity, open('similarity.pkl', 'wb'))
    print("Done!")
else:
    print("Loading similarity matrix...")
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    print("Done!")


def recommendation(movie):
    try:
        movie_index = movies[movies['title'].str.lower() == movie.lower()].index[0]
        distance = similarity[movie_index]
        movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
        recommendations = []
        for i in movie_list:
            recommendations.append(movies.iloc[i[0]].title)
        return recommendations
    except:
        return []


@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    movie_list = movies['title'].tolist()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "movies": movie_list
    })


@app.get('/recommend')
def get_recommendation(movie: str):
    results = recommendation(movie)
    return {"movie": movie, "results": results}
