import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ----------------------------
# Custom Styling
# ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --gold: #e8b84b;
    --gold-dim: #c9973a;
    --bg: #0c0c0f;
    --surface: #14141a;
    --surface2: #1c1c26;
    --border: rgba(232, 184, 75, 0.18);
    --text: #f0ece2;
    --text-muted: #8a8590;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 960px; margin: 0 auto; }

/* ---- Hero ---- */
.hero {
    text-align: center;
    padding: 3.5rem 1rem 2.5rem;
    position: relative;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 340px; height: 340px;
    background: radial-gradient(circle, rgba(232,184,75,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.9rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    font-weight: 900;
    line-height: 1.1;
    color: var(--text);
    margin: 0 0 1.1rem;
}
.hero-title span { color: var(--gold); }
.hero-subtitle {
    font-size: 1rem;
    font-weight: 300;
    color: var(--text-muted);
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.7;
}

/* ---- Divider ---- */
.divider {
    height: 1px;
    background: linear-gradient(to right, transparent, var(--border), transparent);
    margin: 2rem 0;
}

/* ---- Select box label ---- */
.select-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.5rem;
}

/* ---- Streamlit selectbox override ---- */
div[data-baseweb="select"] > div {
    background-color: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    transition: border-color 0.2s;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(232,184,75,0.15) !important;
}

/* ---- Button ---- */
.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3.2em;
    font-size: 0.95rem;
    font-weight: 500;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.06em;
    background: linear-gradient(135deg, var(--gold) 0%, var(--gold-dim) 100%);
    color: #0c0c0f;
    border: none;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s;
    margin-top: 1rem;
}
.stButton > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }

/* ---- Results section ---- */
.results-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
    margin: 2rem 0 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.movie-card {
    display: flex;
    align-items: center;
    gap: 1.1rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s, background 0.2s;
    animation: slideIn 0.35s ease both;
}
.movie-card:hover {
    border-color: var(--gold);
    background: var(--surface2);
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.movie-rank {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 900;
    color: var(--gold);
    opacity: 0.55;
    min-width: 2rem;
    text-align: center;
    line-height: 1;
}
.movie-name {
    font-size: 1.05rem;
    font-weight: 500;
    color: var(--text);
}
.movie-badge {
    margin-left: auto;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--gold);
    background: rgba(232,184,75,0.1);
    border: 1px solid rgba(232,184,75,0.25);
    border-radius: 20px;
    padding: 0.25em 0.75em;
}

/* ---- Footer ---- */
.footer {
    text-align: center;
    margin-top: 4rem;
    font-size: 0.78rem;
    color: var(--text-muted);
    letter-spacing: 0.04em;
}
.footer span { color: var(--gold); }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Dataset
# ----------------------------
movies = pickle.load(open("movies.pkl", "rb"))

# ----------------------------
# Create Similarity Matrix
# ----------------------------
@st.cache_resource
def create_similarity():
    cv = CountVectorizer(
        max_features=5000,
        stop_words='english'
    )
    vectors = cv.fit_transform(
        movies['tags']
    )
    similarity = cosine_similarity(
        vectors
    )
    return similarity

similarity = create_similarity()

# ----------------------------
# Recommendation Function
# ----------------------------
def recommend(movie):
    movie_index = movies[
        movies['title'] == movie
    ].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]
    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(
            movies.iloc[i[0]].title
        )
    return recommended_movies

# ----------------------------
# UI — Hero
# ----------------------------
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">✦ ML-Powered Discovery</div>
    <h1 class="hero-title">Find Your Next<br><span>Favourite Film</span></h1>
    <p class="hero-subtitle">Select a movie you love and let our content-based engine surface five films you're likely to enjoy.</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ----------------------------
# UI — Selector
# ----------------------------
st.markdown('<p class="select-label">Choose a movie</p>', unsafe_allow_html=True)

selected_movie = st.selectbox(
    "Select a Movie",
    movies['title'].values,
    label_visibility="collapsed"
)

if st.button("✦  Get Recommendations"):
    recommendations = recommend(selected_movie)

    st.markdown('<div class="results-header">🎬 Top Picks For You</div>', unsafe_allow_html=True)

    for idx, movie in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="movie-card">
            <div class="movie-rank">{idx:02d}</div>
            <div class="movie-name">{movie}</div>
            <div class="movie-badge">Recommended</div>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
<div class="divider" style="margin-top:3rem;"></div>
<div class="footer">Built with <span>Streamlit</span> &amp; scikit-learn &nbsp;·&nbsp; Content-Based Filtering</div>
""", unsafe_allow_html=True)