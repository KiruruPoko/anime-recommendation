import streamlit as st
import pandas as pd
import torch
import html
from sentence_transformers import SentenceTransformer, util

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    return SentenceTransformer("KiruruP/anime-recommendation-multilingual-mpnet-base-v2-peft")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def embed_corpus(_model, texts):
    return _model.encode(texts, convert_to_tensor=True)
st.set_page_config(
    page_title="🎌 Anime Recommender",
    page_icon = "🍿",
    layout="centered",
)
def render_anime_card(row):
    css_style = """
    <style>
        .anime-card {
            border-radius: 6px;
            border: 1px solid #333;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            padding: 15px;
            background-color: #222;
            margin-bottom: 20px;
            color: #f1f1f1;
            font-family: Arial, sans-serif;
        }
        .anime-title {
            font-size: 1.5em;
            margin-bottom: 8px;
        }
        .anime-title a {
            color: #81c995;
            text-decoration: none;
        }
        .anime-title a:hover {
            text-decoration: underline;
        }
        .anime-meta {
            font-size: 0.9em;
            color: #aaa;
            margin-bottom: 12px;
        }
        .anime-synopsis {
            color: #dcdcdc;
            margin-bottom: 10px;
            line-height: 1.6;
        }
        .meta-line {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            font-size: 0.9em;
            margin-top: 10px;
            border-top: 1px solid #333;
            padding-top: 8px;
        }
        .meta-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .anime-image {
            max-width: 180px;
            border-radius: 4px;
            margin: 10px 0;
        }
    </style>
    """

    title = str(row.get("title", "Unknown"))
    synopsis_raw = str(row.get("synopsis", "") or "")
    synopsis_clean = synopsis_raw.replace("\n", " ").replace("\r", " ")
    safe_synopsis = html.escape(synopsis_clean[:350]) + ("..." if len(synopsis_clean) > 350 else "")

    genres = str(row.get("genres", ""))
    anime_type = str(row.get("type", "N/A"))
    episodes = str(row.get("episodes", "N/A"))
    rating = str(row.get("rating", "N/A"))
    score = row.get("score", "N/A")
    similarity = row.get("Similarity", 0.0)
    mal_id = row.get("mal_id", None)
    mal_link = f"https://myanimelist.net/anime/{mal_id}" if mal_id else "#"
    image_url = str(row.get("image_jpg_url", ""))

    card_html = f"""
    {css_style}
    <div class="anime-card">
        <div class="anime-title">
            <a href="{mal_link}" target="_blank" rel="noopener noreferrer">{title}</a>
        </div>
        <img class="anime-image" src="{image_url}" alt="Anime Poster" onerror="this.style.display='none'">
        <div class="anime-synopsis">{safe_synopsis}</div>
        <div class="anime-meta">Genres: {genres}</div>
        <div class="meta-line">
            <div class="meta-item">🎬 Type: {anime_type}</div>
            <div class="meta-item">🎞 Episodes: {episodes}</div>
            <div class="meta-item">⭐ Score: {score}</div>
            <div class="meta-item">⭐ Rating: {rating}</div>
            <div class="meta-item">🧠 Similarity: {f"{similarity*100:.1f}%"}</div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def recommend_anime(q_encode, embeddings, df, top_k=3):
    cosine_score = util.cos_sim(q_encode, embeddings)
    top_indices = torch.topk(cosine_score.flatten(), top_k).indices.tolist()
    recs = df.iloc[top_indices].copy()
    recs["Similarity"] = [cosine_score.flatten()[i].item() for i in top_indices]
    return recs


# --- Streamlit app ---
def main():
    df = load_data("../anime recommendation/data/anime_clean.csv") 
    model = load_model()
    #embeddings = embed_corpus(model, df['text_corpus'].to_list())
    embeddings = torch.load('embedding_corpus.pt', map_location=device).to(device)

    st.title("Anime Recommendation System 🇯🇵")

    query = st.text_input("Enter your anime search query 👇", placeholder = "Example query: Anime about spy that has to create a fake family for a mission")

    if query:
        q_encode = model.encode(query, convert_to_tensor=True, device=device)
        recs = recommend_anime(q_encode, embeddings, df, top_k=3)

        st.markdown("**Here are your recommendations**")
        for _, row in recs.iterrows():
            render_anime_card(row)


if __name__ == "__main__":
    main()
