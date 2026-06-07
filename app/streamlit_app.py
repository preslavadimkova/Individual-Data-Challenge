from pathlib import Path
import os
import json
import random
import re
import html
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import spacy
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LEXICON_PATH = PROJECT_ROOT / "data" / "emotion_lexicon" / "poetic_emotion_lexicon.json"
MODEL_DIR = PROJECT_ROOT / "models" / "sentence_transformer_poetry"
POEMS_CLEAN_PATH = PROCESSED_DIR / "poems_clean.csv"
RELATIONS_PATH = PROCESSED_DIR / "symbol_emotion_edges.csv"
GRAPH_NODES_PATH = PROCESSED_DIR / "graph_nodes.csv"
GRAPH_EDGES_PATH = PROCESSED_DIR / "graph_edges.csv"
EMBEDDINGS_PATH = PROCESSED_DIR / "poem_embeddings.npy"
EMBEDDING_METADATA_PATH = PROCESSED_DIR / "poem_embedding_metadata.csv"
BASE_SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
MAX_DISTANCE = 10

st.set_page_config(page_title="Find the Poet Inside You", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --poetry-purple: #8b5cf6;
        --poetry-purple-dark: #5b21b6;
        --poetry-purple-soft: #f6f0ff;
        --poetry-lavender: #efe4ff;
        --poetry-rose: #fce7f3;
        --poetry-ink: #241b35;
        --poetry-muted: #6f6680;
    }
    .stApp {
        background: #ffffff;
        color: var(--poetry-ink);
    }
    .block-container {
        padding-top: 1.5rem;
        max-width: 1380px;
    }
    h1, h2, h3 {
        color: var(--poetry-ink);
        letter-spacing: 0;
    }
    div[data-testid="stSidebar"] {
        display: none;
    }
    div.stButton > button {
        border-radius: 18px;
        min-height: 3.1rem;
        font-weight: 600;
        border: 1px solid #dac8ff;
        background: linear-gradient(180deg, #ffffff 0%, #f8f3ff 100%);
        color: var(--poetry-purple-dark);
        box-shadow: 0 1px 2px rgba(39, 24, 74, 0.08);
        transition: all 140ms ease;
    }
    div.stButton > button:hover {
        border-color: var(--poetry-purple);
        color: #ffffff;
        background: var(--poetry-purple);
        box-shadow: 0 8px 20px rgba(124, 58, 237, 0.22);
    }
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        border-radius: 10px !important;
        border: 1px solid #eadfff !important;
        background: #ffffff !important;
        color: var(--poetry-ink) !important;
        box-shadow: none !important;
        outline: none !important;
        font-weight: 400 !important;
    }
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextArea"] textarea:focus,
    input:focus,
    textarea:focus {
        border: 1px solid #eadfff !important;
        box-shadow: none !important;
        outline: none !important;
    }
    div[data-baseweb="input"],
    div[data-baseweb="textarea"] {
        border-radius: 10px !important;
        border-color: #eadfff !important;
        border-width: 1px !important;
        box-shadow: none !important;
        outline: none !important;
        background: #ffffff !important;
    }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="textarea"]:focus-within {
        border-color: #eadfff !important;
        border-width: 1px !important;
        box-shadow: none !important;
        outline: none !important;
    }
    div[data-baseweb="input"]::before,
    div[data-baseweb="input"]::after,
    div[data-baseweb="textarea"]::before,
    div[data-baseweb="textarea"]::after {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    div[data-testid="stTextInput"] input::placeholder,
    div[data-testid="stTextArea"] textarea::placeholder {
        color: #9a8cad !important;
    }
    div[data-baseweb="select"] > div {
        border-radius: 10px;
        border-color: #eadfff !important;
        border-width: 1px !important;
        background: #ffffff !important;
        color: var(--poetry-ink) !important;
        box-shadow: none !important;
        outline: none !important;
        font-weight: 400 !important;
    }
    div[data-baseweb="select"] > div:focus,
    div[data-baseweb="select"] > div:focus-within {
        border-color: #eadfff !important;
        border-width: 1px !important;
        box-shadow: none !important;
        outline: none !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #eadfff !important;
        box-shadow: none !important;
        outline: none !important;
    }
    div[data-baseweb="select"] > div::before,
    div[data-baseweb="select"] > div::after,
    div[data-baseweb="select"] span::before,
    div[data-baseweb="select"] span::after {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] input:focus,
    div[data-baseweb="select"] [contenteditable="true"],
    div[data-baseweb="select"] [contenteditable="true"]:focus {
        color: var(--poetry-ink) !important;
        background: transparent !important;
        box-shadow: none !important;
        outline: none !important;
        border: none !important;
    }
    div[data-baseweb="select"] input::placeholder {
        color: #b7aec4 !important;
        opacity: 1 !important;
    }
    div[data-baseweb="select"] span {
        color: #b7aec4 !important;
    }
    div[data-baseweb="select"] div[aria-selected="true"],
    div[data-baseweb="select"] div[aria-selected="false"],
    div[data-baseweb="select"] div:focus {
        box-shadow: none !important;
        outline: none !important;
    }
    div[data-baseweb="select"] * {
        background-color: #ffffff !important;
        color: var(--poetry-ink) !important;
        font-weight: 400 !important;
    }
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] div,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] span,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] input {
        color: #9f93b4 !important;
        font-weight: 400 !important;
    }
    div[data-testid="stMultiSelect"] div[data-baseweb="tag"] span {
        color: var(--poetry-purple-dark) !important;
    }
    div[data-baseweb="popover"] div {
        background-color: #ffffff !important;
        color: var(--poetry-ink) !important;
    }
    div[data-baseweb="popover"] li:hover {
        background-color: var(--poetry-purple-soft) !important;
    }
    div[data-testid="stSlider"] [role="slider"] {
        background-color: var(--poetry-purple) !important;
        border-color: var(--poetry-purple) !important;
    }
    div[data-testid="stSlider"] div {
        color: var(--poetry-purple-dark) !important;
    }
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
        background: #eee8fb !important;
    }
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div > div {
        background: var(--poetry-purple) !important;
    }
    label, .st-emotion-cache-ue6h4q, .st-emotion-cache-1y4p8pa {
        color: var(--poetry-ink) !important;
    }
    .poetry-hero {
        text-align: center;
        padding: 5.5rem 1rem 2.2rem;
        max-width: 1280px;
        margin: 0 auto;
    }
    .poetry-hero h1 {
        font-size: clamp(3.2rem, 5vw, 5.5rem);
        line-height: 1;
        margin: 0 0 1rem;
        color: var(--poetry-purple-dark);
        white-space: nowrap;
    }
    @media (max-width: 900px) {
        .poetry-hero h1 {
            font-size: clamp(2.2rem, 9vw, 4rem);
            white-space: normal;
        }
    }
    .poetry-hero p {
        max-width: 720px;
        margin-left: auto;
        margin-right: auto;
        font-size: 1.2rem;
        color: var(--poetry-muted);
        text-align: center;
        display: block;
    }
    .hero-subtitle {
        width: 100%;
        text-align: center;
        display: flex;
        justify-content: center;
    }
    .home-actions {
        width: min(1180px, 94vw);
        margin: 1.2rem auto 0;
        text-align: center;
    }
    .home-card {
        display: block;
        min-height: 9rem;
        border: 1px solid #dbcaff;
        border-radius: 30px;
        padding: 1.4rem 1.1rem;
        text-align: center;
        text-decoration: none !important;
        background: linear-gradient(145deg, #ffffff 0%, #f6efff 62%, #fff6fb 100%);
        color: var(--poetry-purple-dark) !important;
        box-shadow: 0 12px 34px rgba(76, 42, 126, 0.11);
        transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
    }
    .home-card:hover {
        transform: translateY(-2px);
        border-color: var(--poetry-purple);
        box-shadow: 0 18px 42px rgba(124, 58, 237, 0.18);
    }
    .home-card .home-title {
        display: block;
        font-size: 1.18rem;
        font-weight: 700;
        margin-bottom: 0.55rem;
    }
    .home-card .home-desc {
        display: block;
        font-size: 0.9rem;
        line-height: 1.35;
        color: var(--poetry-muted);
    }
    .top-nav {
        padding: 0.8rem 1rem;
        margin: 0 auto 1.4rem;
        border: 1px solid #eadfff;
        border-radius: 22px;
        background: linear-gradient(180deg, #ffffff 0%, #fbf8ff 100%);
        box-shadow: 0 8px 24px rgba(79, 45, 130, 0.07);
        max-width: 820px;
        text-align: center;
    }
    .section-panel {
        background: #ffffff;
        border: 1px solid #eee4ff;
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 10px 30px rgba(79, 45, 130, 0.07);
    }
    .graph-panel {
        border: 1px solid #efe4ff;
        border-radius: 20px;
        background: #ffffff;
        padding: 0.6rem;
        box-shadow: 0 12px 34px rgba(82, 48, 130, 0.08);
    }
    .example-line {
        border-left: 3px solid var(--poetry-purple);
        padding: 0.35rem 0 0.35rem 0.8rem;
        margin: 0.5rem 0;
        color: var(--poetry-muted);
    }
    .graph-legend {
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 0.4rem 0 1rem;
        color: var(--poetry-muted);
        font-size: 0.95rem;
    }
    .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }
    .legend-dot {
        width: 0.8rem;
        height: 0.8rem;
        border-radius: 50%;
        display: inline-block;
    }
    .poem-card {
        border: 1px solid #e8ddff;
        border-radius: 18px;
        padding: 1rem;
        background: #ffffff;
        box-shadow: 0 8px 22px rgba(49, 28, 85, 0.08);
        min-height: 360px;
    }
    .poem-card h4 {
        margin: 0 0 0.25rem;
        color: var(--poetry-purple-dark);
    }
    .poem-card .byline {
        color: var(--poetry-muted);
        font-size: 0.92rem;
        margin-bottom: 0.8rem;
    }
    .poem-card .poem-text {
        white-space: pre-wrap;
        color: #302642;
        line-height: 1.5;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def read_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource
def cached_nlp():
    return spacy.load("en_core_web_sm")


@st.cache_resource
def cached_sentence_model():
    path = MODEL_DIR if MODEL_DIR.exists() else BASE_SENTENCE_MODEL
    model = SentenceTransformer(str(path))
    model.max_seq_length = 128
    return model


def line_number_from_char(text, char_index):
    return text[:char_index].count("\n") + 1


def normalize_symbol_text(value):
    text = re.sub(r"\s+", " ", value.lower()).strip()
    return re.sub(r"^[^a-z]+|[^a-z]+$", "", text)


def load_emotion_lexicon():
    if not LEXICON_PATH.exists():
        return {}
    with open(LEXICON_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def emotion_lookup_from_lexicon(lexicon):
    lookup = {}
    for category, words in lexicon.items():
        for word in words:
            lookup.setdefault(word.lower(), set()).add(category)
    return lookup


def emotion_word_set():
    words = set()
    for values in load_emotion_lexicon().values():
        words.update(str(value).lower() for value in values)
    words.update(load_emotion_lexicon().keys())
    return words


def is_emotion_word(value):
    return str(value).lower().strip() in emotion_word_set()


def extract_symbols_from_text(text, nlp):
    generic = {"thing", "time", "way", "day", "man", "woman", "people", "life", "world", "one", "body", "place", "something", "nothing", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "novemeber", "december"}
    symbol_allowed_emotion_words = {"heart"}
    emotion_words = emotion_word_set() - symbol_allowed_emotion_words
    rows = []
    seen = set()
    doc = nlp(text)
    for chunk in doc.noun_chunks:
        token = chunk.root
        lemma = normalize_symbol_text(token.lemma_)
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop and lemma and len(lemma) >= 3 and lemma not in generic and lemma not in emotion_words:
            key = lemma
            if key not in seen:
                seen.add(key)
                rows.append({"symbol": lemma, "symbol_text": normalize_symbol_text(token.text), "lemma": lemma, "start_token": token.i, "end_token": token.i + 1, "line_number": line_number_from_char(text, token.idx), "source_method": "noun_chunk_head"})
    for token in doc:
        lemma = normalize_symbol_text(token.lemma_)
        key = lemma
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop and lemma and len(lemma) >= 3 and lemma not in generic and lemma not in emotion_words and key not in seen:
            seen.add(key)
            rows.append({"symbol": lemma, "symbol_text": normalize_symbol_text(token.text), "lemma": lemma, "start_token": token.i, "end_token": token.i + 1, "line_number": line_number_from_char(text, token.idx), "source_method": "noun"})
    return rows


def extract_emotions_from_text(text, nlp, lexicon):
    lookup = emotion_lookup_from_lexicon(lexicon)
    rows = []
    for token in nlp(text):
        categories = set()
        categories.update(lookup.get(token.text.lower(), set()))
        categories.update(lookup.get(token.lemma_.lower(), set()))
        for category in sorted(categories):
            rows.append({"emotion_category": category, "matched_word": token.text, "lemma": token.lemma_.lower(), "start_token": token.i, "line_number": line_number_from_char(text, token.idx)})
    return rows


def token_context(text, start_token, end_token, window=10):
    tokens = text.split()
    start = max(0, min(start_token, end_token) - window)
    end = min(len(tokens), max(start_token, end_token) + window + 1)
    return " ".join(tokens[start:end])


def extract_user_relations(text, symbols, emotions):
    rows = []
    symbols_df = pd.DataFrame(symbols)
    if symbols_df.empty:
        return rows
    for emotion in emotions:
        candidates = symbols_df.copy()
        candidates["token_distance"] = (candidates["start_token"] - emotion["start_token"]).abs()
        candidates["line_distance"] = (candidates["line_number"] - emotion["line_number"]).abs()
        candidates = candidates[candidates["token_distance"] <= MAX_DISTANCE]
        if candidates.empty:
            continue
        candidates = candidates[candidates["token_distance"] == candidates["token_distance"].min()]
        candidates = candidates[candidates["line_distance"] == candidates["line_distance"].min()]
        for symbol in candidates.to_dict("records"):
            rows.append({"source_symbol": symbol["symbol"], "target_emotion": emotion["emotion_category"], "token_distance": int(abs(symbol["start_token"] - emotion["start_token"])), "line_distance": int(abs(symbol["line_number"] - emotion["line_number"])), "context_snippet": token_context(text, int(symbol["start_token"]), int(emotion["start_token"]))})
    return rows


def build_user_poem_graph(symbols, emotions, user_relations):
    node_rows = []
    edge_rows = []
    for symbol in sorted({item["symbol"] for item in symbols}):
        node_rows.append({"id": f"user_symbol:{symbol}", "label": symbol, "type": "symbol", "frequency": 1, "author": ""})
    for emotion in sorted({item["emotion_category"] for item in emotions}):
        node_rows.append({"id": f"user_emotion:{emotion}", "label": emotion, "type": "emotion", "frequency": 1, "author": ""})
    for relation in user_relations:
        edge_rows.append(
            {
                "source": f"user_symbol:{relation['source_symbol']}",
                "target": f"user_emotion:{relation['target_emotion']}",
                "type": "POSSIBLE_NEAR_EMOTION",
                "weight": 1,
                "poem_id": "user_poem",
                "evidence": relation["context_snippet"],
            }
        )
    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def random_walk_on_visible_graph(nodes_df, edges_df, steps=5):
    if nodes_df.empty or edges_df.empty:
        return []
    current = random.choice(nodes_df["id"].tolist())
    previous = None
    path = [current]
    for value in range(steps):
        adjacent = edges_df[(edges_df["source"] == current) | (edges_df["target"] == current)]
        neighbors = []
        for edge in adjacent.itertuples(index=False):
            neighbor = edge.target if edge.source == current else edge.source
            if neighbor != previous:
                neighbors.append(neighbor)
        if not neighbors:
            break
        previous = current
        current = random.choice(neighbors)
        path.append(current)
    return path


def labels_for_path(path, nodes_df):
    labels = nodes_df.set_index("id")["label"].to_dict() if not nodes_df.empty else {}
    return [labels.get(node_id, node_id.split(":", 1)[-1]) for node_id in path]


def format_unique_values(values, limit=20):
    cleaned = []
    for value in values:
        if value and value not in cleaned:
            cleaned.append(value)
        if len(cleaned) == limit:
            break
    return cleaned


def show_similar_poems(similar_poems):
    if not similar_poems:
        st.info("Similar poems will appear after notebook 04 creates embeddings.")
        return
    visible_count = st.session_state.get("similar_poems_visible_count", 3)
    visible = similar_poems[:visible_count]
    for index in range(0, len(visible), 3):
        columns = st.columns(3)
        for column, poem in zip(columns, visible[index:index + 3]):
            with column:
                title = poem.get("title", "Untitled")
                author = poem.get("author", "Unknown")
                poem_text = str(poem.get("poem_text", "")).strip()
                st.markdown(
                    f"""
                    <div class="poem-card">
                        <h4>{html.escape(str(title))}</h4>
                        <div class="byline">{html.escape(str(author))}</div>
                        <div class="poem-text">{html.escape(poem_text)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    if visible_count < len(similar_poems):
        if st.button("Show more similar poems"):
            st.session_state.similar_poems_visible_count = min(len(similar_poems), visible_count + 3)
            st.rerun()


def parse_terms(value):
    return [term.strip().lower() for term in re.split(r"[,;]", value or "") if term.strip()]


def current_fragment(value):
    parts = re.split(r"[,;]", value or "")
    return parts[-1].strip().lower() if parts else ""


def term_suggestions(value, options, limit=6):
    fragment = current_fragment(value)
    if len(fragment) < 1:
        return []
    matches = [option for option in options if fragment in str(option).lower()]
    return matches[:limit]


def add_term_to_text(existing, term):
    terms = parse_terms(existing)
    if term.lower() not in terms:
        terms.append(term.lower())
    return ", ".join(terms)


def add_selected_term(state_key, term):
    values = st.session_state.get(state_key, [])
    if term.lower() not in values:
        values.append(term.lower())
    st.session_state[state_key] = values


def build_random_walk_prompt(original_poem, walk_labels, relations):
    relation_text = "; ".join(f"{row['source_symbol']} may suggest {row['target_emotion']}" for row in relations[:8])
    walk_text = " -> ".join(walk_labels)
    required_words = ", ".join(walk_labels)
    return f"""You are helping a beginner poet imagine an alternate version of their poem.
Write in the style, tone, rhythm, level of simplicity, imagery density, and line length pattern of the original poem, but do not copy its lines or phrases.
Use this suggestive path from their poem's symbol-emotion graph:

Random walk: {walk_text}
Required words or very close forms that must visibly appear in the poem: {required_words}
Possible symbol-emotion connections: {relation_text}

Original poem:
{original_poem}

Write a short original poem showing how the poem could have also turned out.
Keep it stylistically close to the pasted poem.
Include every required word from the random walk, especially concrete symbols.
Do not replace concrete required words with only a mood or implication. If the path contains dog, the poem must visibly include dog or dogs.
Do not copy the original poem.
Use the random walk as inspiration."""


def search_symbol(symbol_query):
    nodes = read_csv(GRAPH_NODES_PATH)
    relations = read_csv(RELATIONS_PATH)
    if nodes.empty:
        return {"nodes": [], "relations": []}
    query = symbol_query.lower().strip()
    matches = nodes[(nodes["type"] == "symbol") & (nodes["label"].str.lower().str.contains(query, na=False))]
    related = relations[relations["source_symbol"].str.lower().str.contains(query, na=False)] if not relations.empty else pd.DataFrame()
    return {"nodes": matches.to_dict("records"), "relations": related.head(20).to_dict("records")}


def search_emotion(emotion_query):
    nodes = read_csv(GRAPH_NODES_PATH)
    relations = read_csv(RELATIONS_PATH)
    if nodes.empty:
        return {"nodes": [], "relations": []}
    query = emotion_query.lower().strip()
    matches = nodes[(nodes["type"] == "emotion") & (nodes["label"].str.lower().str.contains(query, na=False))]
    related = relations[relations["target_emotion"].str.lower().str.contains(query, na=False)] if not relations.empty else pd.DataFrame()
    return {"nodes": matches.to_dict("records"), "relations": related.head(20).to_dict("records")}


def search_poem(query):
    poems = read_csv(POEMS_CLEAN_PATH)
    if poems.empty:
        return []
    return poems[poems["title"].fillna("").str.lower().str.contains(query.lower(), na=False)].head(20).to_dict("records")


def search_author(query):
    poems = read_csv(POEMS_CLEAN_PATH)
    if poems.empty:
        return []
    return poems[poems["author"].fillna("").str.lower().str.contains(query.lower(), na=False)].head(20).to_dict("records")


def expand_node(node_id, top_k=10):
    nodes = read_csv(GRAPH_NODES_PATH)
    edges = read_csv(GRAPH_EDGES_PATH)
    if nodes.empty or edges.empty:
        return {"nodes": [], "edges": []}
    connected = edges[(edges["source"] == node_id) | (edges["target"] == node_id)].head(top_k)
    neighbor_ids = set(connected["source"]).union(set(connected["target"]))
    return {"nodes": nodes[nodes["id"].isin(neighbor_ids)].to_dict("records"), "edges": connected.to_dict("records")}


def node_id_for_search(search_type, query):
    nodes = read_csv(GRAPH_NODES_PATH)
    poems = read_csv(POEMS_CLEAN_PATH)
    if nodes.empty:
        return ""
    value = query.lower().strip()
    if search_type == "symbol":
        matches = nodes[(nodes["type"] == "symbol") & (nodes["label"].fillna("").str.lower() == value)]
        if matches.empty:
            matches = nodes[(nodes["type"] == "symbol") & (nodes["label"].fillna("").str.lower().str.contains(value, na=False))]
        return matches["id"].iloc[0] if not matches.empty else ""
    if search_type == "emotion":
        matches = nodes[(nodes["type"] == "emotion") & (nodes["label"].fillna("").str.lower() == value)]
        if matches.empty:
            matches = nodes[(nodes["type"] == "emotion") & (nodes["label"].fillna("").str.lower().str.contains(value, na=False))]
        return matches["id"].iloc[0] if not matches.empty else ""
    if search_type == "poem":
        if poems.empty:
            return ""
        matches = poems[poems["title"].fillna("").str.lower() == value]
        if matches.empty:
            matches = poems[poems["title"].fillna("").str.lower().str.contains(value, na=False)]
        return f"poem:{matches['poem_id'].iloc[0]}" if not matches.empty else ""
    matches = nodes[(nodes["type"] == "author") & (nodes["label"].fillna("").str.lower() == value)]
    if matches.empty:
        matches = nodes[(nodes["type"] == "author") & (nodes["label"].fillna("").str.lower().str.contains(value, na=False))]
    return matches["id"].iloc[0] if not matches.empty else ""


def connected_edges_for_node(node_id, top_k=18):
    edges = read_csv(GRAPH_EDGES_PATH)
    if edges.empty or not node_id:
        return pd.DataFrame()
    connected = edges[(edges["source"] == node_id) | (edges["target"] == node_id)].copy()
    if connected.empty:
        return connected
    bad_relation = (
        (connected["type"] == "NEAR_EMOTION")
        & connected["source"].fillna("").str.startswith("symbol:")
        & (
            connected["source"].fillna("").str.replace("symbol:", "", regex=False).map(is_emotion_word)
            | (
                connected["source"].fillna("").str.replace("symbol:", "", regex=False)
                == connected["target"].fillna("").str.replace("emotion:", "", regex=False)
            )
        )
    )
    connected = connected[~bad_relation]
    type_order = {"NEAR_EMOTION": 0, "HAS_SYMBOL": 1, "HAS_EMOTION_WORD": 1, "WRITTEN_BY": 2}
    connected["type_order"] = connected["type"].map(type_order).fillna(9)
    connected["weight_sort"] = pd.to_numeric(connected["weight"], errors="coerce").fillna(1)
    connected = connected.sort_values(["type_order", "weight_sort"], ascending=[True, False])
    return connected.drop(columns=["type_order", "weight_sort"]).head(top_k)


def graph_records_for_node(node_id, top_k=18):
    nodes = read_csv(GRAPH_NODES_PATH)
    if nodes.empty or not node_id:
        return pd.DataFrame(), pd.DataFrame()
    connected = connected_edges_for_node(node_id, top_k=top_k)
    ids = {node_id}
    if not connected.empty:
        ids.update(connected["source"].tolist())
        ids.update(connected["target"].tolist())
    return nodes[nodes["id"].isin(ids)].copy(), connected.copy()


def initialize_search_graph(search_type, query, top_k=18):
    center_id = node_id_for_search(search_type, query)
    if not center_id:
        st.session_state.search_graph_nodes = pd.DataFrame()
        st.session_state.search_graph_edges = pd.DataFrame()
        st.session_state.search_graph_center = ""
        return
    nodes, edges = graph_records_for_node(center_id, top_k=top_k)
    st.session_state.search_graph_nodes = nodes
    st.session_state.search_graph_edges = edges
    st.session_state.search_graph_center = center_id
    st.session_state.last_expanded_node = ""


def expand_search_graph(node_id, top_k=12):
    if not node_id:
        return
    current_nodes = st.session_state.get("search_graph_nodes", pd.DataFrame())
    current_edges = st.session_state.get("search_graph_edges", pd.DataFrame())
    new_nodes, new_edges = graph_records_for_node(node_id, top_k=top_k)
    if current_nodes.empty:
        combined_nodes = new_nodes
    else:
        combined_nodes = pd.concat([current_nodes, new_nodes], ignore_index=True).drop_duplicates("id")
    if current_edges.empty:
        combined_edges = new_edges
    else:
        combined_edges = pd.concat([current_edges, new_edges], ignore_index=True).drop_duplicates(["source", "target", "type", "poem_id"])
    st.session_state.search_graph_nodes = combined_nodes
    st.session_state.search_graph_edges = combined_edges
    st.session_state.search_graph_center = node_id


def random_walk(start_node, steps=5):
    edges = read_csv(GRAPH_EDGES_PATH)
    if edges.empty:
        return [start_node]
    path = [start_node]
    previous = None
    current = start_node
    for value in range(steps):
        adjacent = edges[(edges["source"] == current) | (edges["target"] == current)]
        neighbors = []
        for edge in adjacent.itertuples(index=False):
            neighbor = edge.target if edge.source == current else edge.source
            if neighbor != previous:
                neighbors.append(neighbor)
        if not neighbors:
            break
        previous = current
        current = random.choice(neighbors)
        path.append(current)
    return path


def find_similar_poems(query_text, top_k=5):
    if not EMBEDDINGS_PATH.exists() or not EMBEDDING_METADATA_PATH.exists():
        return []
    model = cached_sentence_model()
    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = pd.read_csv(EMBEDDING_METADATA_PATH)
    query_embedding = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
    scores = cosine_similarity(query_embedding, embeddings)[0]
    order = np.argsort(scores)[::-1][:top_k]
    rows = metadata.iloc[order].copy()
    rows["similarity"] = scores[order]
    return rows.to_dict("records")


def build_generation_prompt(symbols, emotions, relations, snippets, style=None, length=None):
    return f"""You are helping a beginner poet write an original poem.
Use the following inspiration graph:

Symbols: {', '.join(symbols) if symbols else 'none selected'}
Emotions: {', '.join(emotions) if emotions else 'none selected'}
Relations: {'; '.join(relations) if relations else 'no explicit relation selected'}
Example poetic associations from corpus:
{chr(10).join(snippets[:5]) if snippets else 'No corpus snippets were selected.'}

Write a {length or 'short'} original poem in a {style or 'clear, image-rich, contemporary lyric'} style.
Do not copy the example lines.
Use the graph only as inspiration."""


def generate_poem_with_openrouter(prompt, temperature=0.8, max_tokens=400):
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return {"poem": "", "message": "Set OPENROUTER_API_KEY to enable generation."}
    try:
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": OPENROUTER_MODEL, "messages": [{"role": "system", "content": "You write original poems inspired by graph evidence without copying source text."}, {"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens},
            timeout=60,
        )
        response.raise_for_status()
        poem_content = response.json()["choices"][0]["message"]["content"]
        if not poem_content or poem_content.strip() == "":
            return {"poem": "", "message": f"OpenRouter returned empty content. Model: {OPENROUTER_MODEL}. Try a different model or try again."}
        return {"poem": poem_content, "message": "Generated with OpenRouter."}
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.text if hasattr(e.response, 'text') else str(e)
        return {"poem": "", "message": f"API Error ({e.response.status_code}): {error_detail}"}
    except Exception as e:
        return {"poem": "", "message": f"Error: {str(e)}"}


def graph_figure(nodes_df, edges_df, height=760, center_id=""):
    if nodes_df.empty:
        return go.Figure()
    ids = nodes_df["id"].tolist()
    if center_id and center_id in ids:
        ordered_ids = [center_id] + [node_id for node_id in ids if node_id != center_id]
    else:
        ordered_ids = ids
    positions = {}
    for index, node_id in enumerate(ordered_ids):
        if index == 0 and node_id == center_id:
            positions[node_id] = (0, 0)
        else:
            angle = 2 * np.pi * (index - 1) / max(1, len(ordered_ids) - 1)
            radius = 1.25 + ((index - 1) // 18) * 0.65
            positions[node_id] = (radius * np.cos(angle), radius * np.sin(angle))
    x_values = [positions[node_id][0] for node_id in ids]
    y_values = [positions[node_id][1] for node_id in ids]
    edge_x = []
    edge_y = []
    for edge in edges_df.itertuples(index=False):
        if edge.source in positions and edge.target in positions:
            s = positions[edge.source]
            t = positions[edge.target]
            edge_x.extend([s[0], t[0], None])
            edge_y.extend([s[1], t[1], None])
    colors = {"symbol": "#7c3aed", "emotion": "#db2777", "poem": "#2563eb", "author": "#6d28d9"}
    node_colors = [colors.get(row.type, "#666666") for row in nodes_df.itertuples(index=False)]
    node_sizes = [30 if row.id == center_id else 20 for row in nodes_df.itertuples(index=False)]
    hover_text = [f"{row.id}<br>type: {row.type}<br>frequency: {row.frequency}" for row in nodes_df.itertuples(index=False)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1.2, color="#d8c7ff"), hoverinfo="none"))
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers+text",
            text=nodes_df["label"],
            textposition="top center",
            customdata=nodes_df["id"],
            marker=dict(size=node_sizes, color=node_colors, line=dict(width=1.5, color="#ffffff")),
            hovertext=hover_text,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        height=height,
        showlegend=False,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#241b35", family="Arial"),
        margin=dict(l=4, r=4, t=20, b=4),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        clickmode="event+select",
    )
    return fig


def selected_node_from_event(event):
    points = event.get("selection", {}).get("points", []) if isinstance(event, dict) else []
    if not points:
        return ""
    return points[0].get("customdata", "")


def evidence_for_visible_graph(edges_df):
    relations = read_csv(RELATIONS_PATH)
    if relations.empty or edges_df.empty:
        return pd.DataFrame()
    relation_edges = edges_df[edges_df["type"] == "NEAR_EMOTION"]
    if relation_edges.empty:
        return pd.DataFrame()
    rows = []
    for edge in relation_edges.itertuples(index=False):
        if edge.source.startswith("symbol:") and edge.target.startswith("emotion:"):
            symbol = edge.source.replace("symbol:", "", 1)
            emotion = edge.target.replace("emotion:", "", 1)
            if is_emotion_word(symbol) or symbol == emotion:
                continue
            matches = relations[(relations["source_symbol"] == symbol) & (relations["target_emotion"] == emotion)].copy()
            if not matches.empty:
                matches["context_snippet"] = matches["context_snippet"].fillna("").astype(str)
                matches = matches[~matches["context_snippet"].str.lower().isin(["", "nan", "none"])]
                matches["contains_symbol"] = matches["context_snippet"].str.lower().str.contains(rf"\b{re.escape(symbol.lower())}\b", regex=True)
                if matches["contains_symbol"].any():
                    matches = matches[matches["contains_symbol"]]
                matches["snippet_length"] = matches["context_snippet"].str.len()
                matches = matches.sort_values(["contains_symbol", "snippet_length"], ascending=[False, False]).head(3)
            rows.extend(matches.to_dict("records"))
    evidence = pd.DataFrame(rows)
    if evidence.empty:
        return evidence
    evidence["context_snippet"] = evidence["context_snippet"].fillna("").astype(str)
    evidence = evidence[~evidence["context_snippet"].str.lower().isin(["", "nan", "none"])]
    return evidence


def normalized_example_key(row):
    snippet = re.sub(r"\s+", " ", str(row.context_snippet).lower()).strip()
    words = snippet.split()
    compact_snippet = " ".join(words[-18:]) if len(words) > 18 else snippet
    return (
        str(row.source_symbol).lower(),
        str(row.target_emotion).lower(),
        str(row.title).lower(),
        str(row.author).lower(),
        compact_snippet,
    )


nodes_df = read_csv(GRAPH_NODES_PATH)
edges_df = read_csv(GRAPH_EDGES_PATH)
relations_df = read_csv(RELATIONS_PATH)

def go_to(page_name):
    st.session_state.page = page_name


def top_navigation():
    st.markdown('<div class="top-nav">', unsafe_allow_html=True)
    nav_cols = st.columns(4)
    with nav_cols[0]:
        st.button("Home", use_container_width=True, on_click=go_to, args=("Home",))
    with nav_cols[1]:
        st.button("Search Graph", use_container_width=True, on_click=go_to, args=("Search Graph",))
    with nav_cols[2]:
        st.button("Analyze My Poem", use_container_width=True, on_click=go_to, args=("Analyze My Poem",))
    with nav_cols[3]:
        st.button("Generate Poem", use_container_width=True, on_click=go_to, args=("Generate Poem",))
    st.markdown("</div>", unsafe_allow_html=True)


def graph_legend():
    st.markdown(
        """
        <div class="graph-legend">
            <span class="legend-item"><span class="legend-dot" style="background:#7c3aed;"></span>Symbol</span>
            <span class="legend-item"><span class="legend-dot" style="background:#db2777;"></span>Emotion</span>
            <span class="legend-item"><span class="legend-dot" style="background:#2563eb;"></span>Poem</span>
            <span class="legend-item"><span class="legend-dot" style="background:#6d28d9;"></span>Author</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


query_page = st.query_params.get("page")
if query_page in {"Home", "Search Graph", "Analyze My Poem", "Generate Poem"}:
    st.session_state.page = query_page
    st.query_params.clear()
elif "page" not in st.session_state:
    st.session_state.page = "Home"

page = st.session_state.page

if page == "Home":
    st.markdown('<div class="poetry-hero"><h1>Find the poet inside you</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle"><p>Explore how poets connect symbols and emotions, then use those paths as inspiration for your own writing.</p></div></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.write("")
    st.markdown('<div class="home-actions">', unsafe_allow_html=True)
    home_cols = st.columns(3)
    with home_cols[0]:
        st.markdown('<a class="home-card" href="?page=Search%20Graph" target="_self"><span class="home-title">Search the Graph</span><span class="home-desc">Look up a symbol, emotion, poem, or author and expand its connections.</span></a>', unsafe_allow_html=True)
    with home_cols[1]:
        st.markdown('<a class="home-card" href="?page=Analyze%20My%20Poem" target="_self"><span class="home-title">Analyze My Poem</span><span class="home-desc">Paste your poem and see possible symbols, emotions, and graph paths.</span></a>', unsafe_allow_html=True)
    with home_cols[2]:
        st.markdown('<a class="home-card" href="?page=Generate%20Poem" target="_self"><span class="home-title">Generate a Poem</span><span class="home-desc">Use selected graph associations as inspiration for a new poem.</span></a>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    top_navigation()
    st.title(page)

if page == "Search Graph":
    if nodes_df.empty or edges_df.empty:
        st.info("Run notebooks 01, 02, and 03 first to generate graph files.")
    else:
        controls = st.columns([1, 2, 1, 1])
        with controls[0]:
            search_type = st.selectbox("Search type", ["symbol", "emotion", "poem", "author"])
        with controls[1]:
            query = st.text_input("Search term", value=st.session_state.get("last_query", "moon"))
        with controls[2]:
            top_k = st.slider("Connections", 5, 40, 18)
        with controls[3]:
            st.write("")
            st.write("")
            search_clicked = st.button("Search", use_container_width=True)
        if search_clicked and query.strip():
            st.session_state.last_query = query
            initialize_search_graph(search_type, query, top_k=top_k)
        if "search_graph_nodes" not in st.session_state:
            initialize_search_graph("symbol", "moon", top_k=top_k)
        graph_nodes = st.session_state.get("search_graph_nodes", pd.DataFrame())
        graph_edges = st.session_state.get("search_graph_edges", pd.DataFrame())
        center_id = st.session_state.get("search_graph_center", "")
        if graph_nodes.empty:
            st.info("No matching graph node found for this search.")
        else:
            graph_legend()
            selected_event = st.plotly_chart(
                graph_figure(graph_nodes, graph_edges, center_id=center_id),
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key="search_graph_plot",
            )
            selected_node = selected_node_from_event(selected_event)
            if selected_node and selected_node != st.session_state.get("last_expanded_node", ""):
                st.session_state.last_expanded_node = selected_node
                expand_search_graph(selected_node, top_k=12)
                st.rerun()
            selected_label = graph_nodes[graph_nodes["id"] == st.session_state.get("last_expanded_node", "")]
            if not selected_label.empty:
                row = selected_label.iloc[0]
                st.caption(f"Selected: {row['label']} ({row['type']})")
            evidence = evidence_for_visible_graph(graph_edges)
            if not evidence.empty:
                st.subheader("Examples")
                seen_evidence = set()
                shown = 0
                for row in evidence.itertuples(index=False):
                    key = normalized_example_key(row)
                    if key in seen_evidence:
                        continue
                    seen_evidence.add(key)
                    st.markdown(f"**{row.source_symbol} -> {row.target_emotion}** in *{row.title}* by {row.author}")
                    snippet = str(row.context_snippet).strip()
                    if snippet.lower() not in {"", "nan", "none"}:
                        st.markdown(f'<div class="example-line">{html.escape(snippet)}</div>', unsafe_allow_html=True)
                    shown += 1
                    if shown == 6:
                        break
            reset_cols = st.columns([1, 5])
            with reset_cols[0]:
                if st.button("Reset graph"):
                    initialize_search_graph(search_type, query, top_k=top_k)
                    st.rerun()

if page == "Analyze My Poem":
    poem_text = st.text_area("Write/paste your poem", height=260, value=st.session_state.get("user_poem_text", ""))
    if st.button("Analyze") and poem_text.strip():
        nlp = cached_nlp()
        lexicon = load_emotion_lexicon()
        symbols = extract_symbols_from_text(poem_text, nlp)
        emotions = extract_emotions_from_text(poem_text, nlp, lexicon)
        user_relations = extract_user_relations(poem_text, symbols, emotions)
        user_nodes, user_edges = build_user_poem_graph(symbols, emotions, user_relations)
        similar_poems = find_similar_poems(poem_text)
        st.session_state.user_poem_text = poem_text
        st.session_state.user_symbols = symbols
        st.session_state.user_emotions = emotions
        st.session_state.user_relations = user_relations
        st.session_state.user_graph_nodes = user_nodes
        st.session_state.user_graph_edges = user_edges
        st.session_state.user_similar_poems = similar_poems
        st.session_state.random_walk_poem = ""
        st.session_state.random_walk_message = ""
        st.session_state.random_walk_prompt = ""
        st.session_state.random_walk_path = []
        st.session_state.similar_poems_visible_count = 3

    if st.session_state.get("user_poem_text", "").strip():
        symbols = st.session_state.get("user_symbols", [])
        emotions = st.session_state.get("user_emotions", [])
        user_relations = st.session_state.get("user_relations", [])
        user_nodes = st.session_state.get("user_graph_nodes", pd.DataFrame())
        user_edges = st.session_state.get("user_graph_edges", pd.DataFrame())
        similar_poems = st.session_state.get("user_similar_poems", [])

        symbol_values = format_unique_values([item["symbol"] for item in symbols], limit=25)
        emotion_values = format_unique_values([item["emotion_category"] for item in emotions], limit=25)

        st.subheader("Found Symbols")
        if symbol_values:
            st.write(", ".join(symbol_values))
        else:
            st.info("No clear symbols were found.")

        st.subheader("Found Emotions")
        if emotion_values:
            st.write(", ".join(emotion_values))
        else:
            st.info("No emotion words from the lexicon were found.")

        st.subheader("Your Poem as a Small Graph")
        if not user_nodes.empty and not user_edges.empty:
            st.plotly_chart(graph_figure(user_nodes, user_edges, center_id=user_nodes["id"].iloc[0]), use_container_width=True)
        elif not user_nodes.empty:
            st.info("Symbols or emotions were found, but no nearby connections were detected.")
            st.plotly_chart(graph_figure(user_nodes, pd.DataFrame(), center_id=user_nodes["id"].iloc[0]), use_container_width=True)
        else:
            st.info("There is not enough extracted material to draw a graph yet.")

        if st.button("Random walk on graph"):
            path = random_walk_on_visible_graph(user_nodes, user_edges, steps=5)
            walk_labels = labels_for_path(path, user_nodes)
            if len(walk_labels) < 2:
                st.session_state.random_walk_path = walk_labels
                st.session_state.random_walk_poem = ""
                st.session_state.random_walk_message = "The graph needs at least one symbol-emotion connection before a random walk can generate an alternate poem."
            else:
                st.session_state.random_walk_path = walk_labels
                prompt = build_random_walk_prompt(st.session_state.user_poem_text, walk_labels, user_relations)
                result = generate_poem_with_openrouter(prompt)
                st.session_state.random_walk_poem = result["poem"]
                st.session_state.random_walk_message = result["message"]
                st.session_state.random_walk_prompt = prompt

        if st.session_state.get("random_walk_path"):
            st.subheader("Random Walk")
            st.write(" -> ".join(st.session_state.random_walk_path))

        if st.session_state.get("random_walk_poem"):
            st.subheader("How your poem could have also turned out")
            st.write(st.session_state.random_walk_poem)
        elif st.session_state.get("random_walk_message"):
            st.info(st.session_state.random_walk_message)

        st.subheader("Similar Poems")
        show_similar_poems(similar_poems)

if page == "Generate Poem":
    symbol_options = sorted(relations_df["source_symbol"].dropna().astype(str).unique().tolist()) if not relations_df.empty else []
    emotion_options = sorted(relations_df["target_emotion"].dropna().astype(str).unique().tolist()) if not relations_df.empty else []
    st.markdown("Select symbol")
    symbols = st.multiselect("Select symbol", symbol_options, placeholder="start typing", label_visibility="collapsed")
    st.markdown("Select emotion")
    emotions = st.multiselect("Select emotion", emotion_options, placeholder="start typing", label_visibility="collapsed")
    selected = relations_df[relations_df["source_symbol"].isin(symbols) | relations_df["target_emotion"].isin(emotions)] if not relations_df.empty else pd.DataFrame()
    relation_strings = [f"{row.source_symbol} -> {row.target_emotion}" for row in selected.head(10).itertuples(index=False)]
    if not selected.empty and "context_snippet" in selected.columns:
        snippets_series = selected["context_snippet"].dropna().astype(str)
        snippets = snippets_series[~snippets_series.str.lower().isin(["", "nan", "none"])].head(5).tolist()
    else:
        snippets = []
    style = st.text_input("Style", placeholder="start typing")
    length = st.selectbox("Length", ["short", "medium", "long"])
    prompt = build_generation_prompt(symbols, emotions, relation_strings, snippets, style=style, length=length)
    if st.button("Generate"):
        result = generate_poem_with_openrouter(prompt)
        if result["poem"]:
            st.subheader("Generated Poem")
            st.write(result["poem"])
        else:
            st.info(result["message"])
