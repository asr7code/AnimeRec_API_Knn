from fastapi import FastAPI
import gzip
import pickle

app = FastAPI()

# Load model
with gzip.open("pivot.pkl.gz", "rb") as f:
    pivot = pickle.load(f)

with gzip.open("knn_model.pkl.gz", "rb") as f:
    model = pickle.load(f)


def recommend(anime_name):
    anime_name = anime_name.lower().strip()

    matches = [anime for anime in pivot.index if anime_name in anime.lower()]

    if not matches:
        return ["Anime not found"]

    anime_name = matches[0]

    index = pivot.index.get_loc(anime_name)

    distances, indices = model.kneighbors(
        pivot.iloc[index, :].values.reshape(1, -1),
        n_neighbors=6
    )

    return [pivot.index[i] for i in indices[0][1:]]


@app.get("/")
def home():
    return {"message": "Anime API running"}


@app.get("/recommend")
def get_recommendations(anime: str):
    return {"recommendations": recommend(anime)}
