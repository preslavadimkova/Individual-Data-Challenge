from functools import lru_cache
from pathlib import Path
import json
import os
import random
import re

import numpy as np
import pandas as pd
import requests
import spacy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from app.symbol_filter import (
    basic_symbol_rejection_reason,
    filter_relations,
    filter_symbol_records,
    normalize_symbol_text,
    token_symbol_rejection_reason,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
STATIC_DIR = APP_DIR / "static"
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
MAX_USER_SYMBOLS = 12
MIN_USER_SYMBOL_SCORE = 2

app = FastAPI(title="Poetry Graph App")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def no_cache_static_assets(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path in {"/", "/index.html", "/styles.css", "/app.js"} or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


class GraphSearchRequest(BaseModel):
    search_type: str = "symbol"
    query: str = "moon"
    top_k: int = 18


class ExpandRequest(BaseModel):
    current_nodes: list[dict] = []
    current_edges: list[dict] = []
    node_id: str
    top_k: int = 12


class AnalyzeRequest(BaseModel):
    poem_text: str


class GenerateRequest(BaseModel):
    symbols: list[str] = []
    emotions: list[str] = []
    style: str = ""
    length: str = "short"


class RandomWalkRequest(BaseModel):
    poem_text: str
    nodes: list[dict] = []
    edges: list[dict] = []
    relations: list[dict] = []
    steps: int = 5


@lru_cache(maxsize=16)
def read_csv_cached(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def read_csv(path: Path) -> pd.DataFrame:
    return read_csv_cached(str(path))


@lru_cache(maxsize=1)
def cached_nlp():
    return spacy.load("en_core_web_sm")


@lru_cache(maxsize=1)
def cached_sentence_model():
    from sentence_transformers import SentenceTransformer

    path = MODEL_DIR if MODEL_DIR.exists() else BASE_SENTENCE_MODEL
    model = SentenceTransformer(str(path))
    model.max_seq_length = 128
    return model


@lru_cache(maxsize=1)
def load_emotion_lexicon():
    if not LEXICON_PATH.exists():
        return {}
    with open(LEXICON_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def emotion_word_set():
    words = set()
    for values in load_emotion_lexicon().values():
        words.update(str(value).lower() for value in values)
    words.update(load_emotion_lexicon().keys())
    return words


@lru_cache(maxsize=1)
def emotion_category_set():
    return {str(category).lower() for category in load_emotion_lexicon()}


@lru_cache(maxsize=1)
def filtered_relations_cached() -> pd.DataFrame:
    relations = read_csv(RELATIONS_PATH)
    if relations.empty:
        return relations
    return filter_relations(relations, cached_nlp(), emotion_words=emotion_category_set())


def filtered_relations() -> pd.DataFrame:
    return filtered_relations_cached().copy()


@lru_cache(maxsize=1)
def public_symbol_set() -> set[str]:
    relations = filtered_relations_cached()
    if relations.empty or "source_symbol" not in relations.columns:
        return set()
    return set(relations["source_symbol"].dropna().astype(str).map(normalize_symbol_text))


def symbol_label_from_id(node_id):
    text = str(node_id)
    return normalize_symbol_text(text.replace("symbol:", "", 1)) if text.startswith("symbol:") else ""


def is_public_symbol_label(value):
    return normalize_symbol_text(value) in public_symbol_set()


def is_public_symbol_node_id(node_id):
    label = symbol_label_from_id(node_id)
    return not label or label in public_symbol_set()


def filter_public_graph_frames(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocked_ids = set()
    if not nodes.empty and {"id", "type", "label"}.issubset(nodes.columns):
        symbol_nodes = nodes[nodes["type"] == "symbol"].copy()
        if not symbol_nodes.empty:
            public_symbol_mask = symbol_nodes["label"].fillna("").map(is_public_symbol_label).astype(bool)
            blocked_ids.update(symbol_nodes[~public_symbol_mask]["id"].astype(str).tolist())
    if not edges.empty:
        for column in ("source", "target"):
            blocked_ids.update(
                value for value in edges[column].dropna().astype(str).tolist()
                if value.startswith("symbol:") and not is_public_symbol_node_id(value)
            )
    if blocked_ids:
        if not nodes.empty and "id" in nodes.columns:
            nodes = nodes[~nodes["id"].astype(str).isin(blocked_ids)].copy()
        if not edges.empty and {"source", "target"}.issubset(edges.columns):
            edges = edges[
                ~edges["source"].astype(str).isin(blocked_ids)
                & ~edges["target"].astype(str).isin(blocked_ids)
            ].copy()
    return nodes, edges


def df_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    clean = df.replace({np.nan: None})
    return clean.to_dict("records")


def line_number_from_char(text, char_index):
    return text[:char_index].count("\n") + 1


def emotion_lookup_from_lexicon(lexicon):
    lookup = {}
    for category, words in lexicon.items():
        for word in words:
            lookup.setdefault(str(word).lower(), set()).add(category)
    return lookup


def is_emotion_word(value):
    return str(value).lower().strip() in emotion_word_set()


def is_emotion_category(value):
    return str(value).lower().strip() in emotion_category_set()


def is_valid_symbol_token(token, lemma, emotion_words):
    return not token_symbol_rejection_reason(token, lemma, emotion_words=emotion_words)


def symbol_candidate_score(item, frequency):
    score = 0
    if item["source_method"] == "noun_chunk_head":
        score += 2
    if item["pos"] == "NOUN":
        score += 1
    if frequency > 1:
        score += 1
    return score


def extract_symbols_from_text(text, nlp):
    emotion_words = emotion_category_set()
    doc = nlp(text)

    candidates = []
    seen_positions = set()

    def add_candidate(token, source_method):
        lemma = normalize_symbol_text(token.lemma_)
        key = (token.i, lemma)
        if key in seen_positions or not is_valid_symbol_token(token, lemma, emotion_words):
            return
        seen_positions.add(key)
        candidates.append({
            "symbol": lemma,
            "symbol_text": normalize_symbol_text(token.text),
            "lemma": lemma,
            "start_token": token.i,
            "end_token": token.i + 1,
            "line_number": line_number_from_char(text, token.idx),
            "source_method": source_method,
            "pos": token.pos_,
        })

    for chunk in doc.noun_chunks:
        token = chunk.root
        add_candidate(token, "noun_chunk_head")
    for token in doc:
        add_candidate(token, "noun")

    frequencies = pd.Series([item["symbol"] for item in candidates]).value_counts().to_dict() if candidates else {}
    best_by_symbol = {}
    for item in candidates:
        score = symbol_candidate_score(item, frequencies.get(item["symbol"], 1))
        if score < MIN_USER_SYMBOL_SCORE:
            continue
        ranked_item = dict(item)
        ranked_item["score"] = score
        current = best_by_symbol.get(item["symbol"])
        if current is None or (ranked_item["score"], -ranked_item["start_token"]) > (current["score"], -current["start_token"]):
            best_by_symbol[item["symbol"]] = ranked_item

    ranked = sorted(best_by_symbol.values(), key=lambda item: (-item["score"], item["start_token"], item["symbol"]))
    rows = []
    for item in ranked:
        item = dict(item)
        item["source_method"] = f"{item['source_method']}+ranked"
        item.pop("pos", None)
        item.pop("score", None)
        rows.append(item)
    return filter_symbol_records(rows, nlp, emotion_words=emotion_words, limit=MAX_USER_SYMBOLS)


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
        edge_rows.append({"source": f"user_symbol:{relation['source_symbol']}", "target": f"user_emotion:{relation['target_emotion']}", "type": "POSSIBLE_NEAR_EMOTION", "weight": 1, "poem_id": "user_poem", "evidence": relation["context_snippet"]})
    return node_rows, edge_rows


def node_id_for_search(search_type, query):
    nodes = read_csv(GRAPH_NODES_PATH)
    poems = read_csv(POEMS_CLEAN_PATH)
    if nodes.empty:
        return ""
    value = query.lower().strip()
    if search_type == "symbol":
        if basic_symbol_rejection_reason(value, emotion_words=emotion_category_set()):
            return ""
        public_symbols = public_symbol_set()
        if value not in public_symbols and not any(value in symbol for symbol in public_symbols):
            return ""
        matches = nodes[
            (nodes["type"] == "symbol")
            & (nodes["label"].fillna("").map(is_public_symbol_label))
            & (nodes["label"].fillna("").str.lower() == value)
        ]
        if matches.empty:
            matches = nodes[
                (nodes["type"] == "symbol")
                & (nodes["label"].fillna("").map(is_public_symbol_label))
                & (nodes["label"].fillna("").str.lower().str.contains(value, na=False, regex=False))
            ]
        return matches["id"].iloc[0] if not matches.empty else ""
    if search_type == "emotion":
        matches = nodes[(nodes["type"] == search_type) & (nodes["label"].fillna("").str.lower() == value)]
        if matches.empty:
            matches = nodes[(nodes["type"] == search_type) & (nodes["label"].fillna("").str.lower().str.contains(value, na=False, regex=False))]
        return matches["id"].iloc[0] if not matches.empty else ""
    if search_type == "poem":
        if poems.empty:
            return ""
        matches = poems[poems["title"].fillna("").str.lower() == value]
        if matches.empty:
            matches = poems[poems["title"].fillna("").str.lower().str.contains(value, na=False, regex=False)]
        return f"poem:{matches['poem_id'].iloc[0]}" if not matches.empty else ""
    matches = nodes[(nodes["type"] == "author") & (nodes["label"].fillna("").str.lower() == value)]
    if matches.empty:
        matches = nodes[(nodes["type"] == "author") & (nodes["label"].fillna("").str.lower().str.contains(value, na=False, regex=False))]
    return matches["id"].iloc[0] if not matches.empty else ""


def connected_edges_for_node(node_id, top_k=18):
    edges = read_csv(GRAPH_EDGES_PATH)
    if edges.empty or not node_id or not is_public_symbol_node_id(node_id):
        return pd.DataFrame()
    connected = edges[(edges["source"] == node_id) | (edges["target"] == node_id)].copy()
    if connected.empty:
        return connected
    _, connected = filter_public_graph_frames(pd.DataFrame(), connected)
    if connected.empty:
        return connected
    bad_relation = (
        (connected["type"] == "NEAR_EMOTION")
        & connected["source"].fillna("").str.startswith("symbol:")
        & (
            connected["source"].fillna("").str.replace("symbol:", "", regex=False).map(is_emotion_category)
            | (connected["source"].fillna("").str.replace("symbol:", "", regex=False) == connected["target"].fillna("").str.replace("emotion:", "", regex=False))
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
        return [], []
    connected = connected_edges_for_node(node_id, top_k=top_k)
    ids = {node_id}
    if not connected.empty:
        ids.update(connected["source"].tolist())
        ids.update(connected["target"].tolist())
    visible_nodes, visible_edges = filter_public_graph_frames(nodes[nodes["id"].isin(ids)].copy(), connected.copy())
    return df_records(visible_nodes), df_records(visible_edges)


def evidence_for_edges(edges):
    relations = filtered_relations()
    if relations.empty or not edges:
        return []
    rows = []
    for edge in edges:
        if edge.get("type") != "NEAR_EMOTION":
            continue
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if not source.startswith("symbol:") or not target.startswith("emotion:"):
            continue
        symbol = source.replace("symbol:", "", 1)
        emotion = target.replace("emotion:", "", 1)
        if not is_public_symbol_label(symbol) or is_emotion_category(symbol) or symbol == emotion:
            continue
        matches = relations[(relations["source_symbol"] == symbol) & (relations["target_emotion"] == emotion)].copy()
        if matches.empty:
            continue
        matches["context_snippet"] = matches["context_snippet"].fillna("").astype(str)
        matches = matches[~matches["context_snippet"].str.lower().isin(["", "nan", "none"])]
        matches["contains_symbol"] = matches["context_snippet"].str.lower().str.contains(rf"\b{re.escape(symbol.lower())}\b", regex=True)
        if matches["contains_symbol"].any():
            matches = matches[matches["contains_symbol"]]
        matches["snippet_length"] = matches["context_snippet"].str.len()
        rows.extend(matches.sort_values(["contains_symbol", "snippet_length"], ascending=[False, False]).head(3).to_dict("records"))
    seen = set()
    unique = []
    for row in rows:
        key = (str(row.get("source_symbol", "")).lower(), str(row.get("target_emotion", "")).lower(), str(row.get("title", "")).lower(), str(row.get("author", "")).lower(), re.sub(r"\s+", " ", str(row.get("context_snippet", "")).lower()).strip())
        if key not in seen:
            seen.add(key)
            unique.append(row)
        if len(unique) == 40:
            break
    return df_records(pd.DataFrame(unique))


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
    return df_records(rows)


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


def build_random_walk_prompt(original_poem, walk_labels, relations):
    relation_text = "; ".join(f"{row['source_symbol']} may suggest {row['target_emotion']}" for row in relations[:8])
    walk_text = " to ".join(walk_labels)
    required_words = ", ".join(walk_labels)
    return f"""You are helping a beginner poet write an original poem from a small symbol-emotion path.
Use the following inspiration graph:

Path: {walk_text}
Required images or moods: {required_words}
Possible symbol-emotion relations: {relation_text or 'no explicit relation selected'}

Original poem:
{original_poem}

Write a short alternate poem in the style, tone, rhythm, simplicity, imagery density, and line length pattern of the original poem.
Make it feel like a real poem, not an explanation, list, summary, or graph description.
Use line breaks and, if fitting, short stanzas.
Include the required images or moods naturally, especially concrete symbols.
Do not mention the graph, random walk, path, or prompt.
Do not copy the original poem.
Do not copy the example wording.
Return only the poem."""


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
        if not poem_content or not poem_content.strip():
            return {"poem": "", "message": f"OpenRouter returned empty content. Model: {OPENROUTER_MODEL}."}
        return {"poem": poem_content, "message": "Generated with OpenRouter."}
    except requests.exceptions.HTTPError as exc:
        error_detail = exc.response.text if hasattr(exc.response, "text") else str(exc)
        return {"poem": "", "message": f"API Error ({exc.response.status_code}): {error_detail}"}
    except Exception as exc:
        return {"poem": "", "message": f"Error: {exc}"}


def random_walk_on_visible_graph(nodes, edges, steps=5):
    if not nodes or not edges:
        return []

    node_ids = {node.get("id") for node in nodes if node.get("id")}
    adjacency = {node_id: set() for node_id in node_ids}
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source in node_ids and target in node_ids and source != target:
            adjacency[source].add(target)
            adjacency[target].add(source)

    starts = [node_id for node_id, neighbors in adjacency.items() if neighbors]
    if not starts:
        return []

    current = random.choice(starts)
    path = [current]
    visited = {current}
    for _ in range(steps):
        neighbors = [neighbor for neighbor in adjacency.get(current, set()) if neighbor not in visited]
        if not neighbors:
            break
        current = random.choice(neighbors)
        visited.add(current)
        path.append(current)
    return path


def labels_for_path(path, nodes):
    labels = {node["id"]: node.get("label", node["id"].split(":", 1)[-1]) for node in nodes}
    return [labels.get(node_id, node_id.split(":", 1)[-1]) for node_id in path]


def unique_walk_labels(labels):
    seen = set()
    unique = []
    for label in labels:
        key = str(label).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(label)
    return unique


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/index.html")
def index_html():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/styles.css")
def root_styles():
    return FileResponse(STATIC_DIR / "styles.css", media_type="text/css")


@app.get("/app.js")
def root_app_js():
    return FileResponse(STATIC_DIR / "app.js", media_type="application/javascript")


@app.get("/api/options")
def options():
    relations = filtered_relations()
    nodes = read_csv(GRAPH_NODES_PATH)
    poems = read_csv(POEMS_CLEAN_PATH)

    symbols = []
    emotions = []
    if not relations.empty:
        symbols = sorted(relations["source_symbol"].dropna().astype(str).unique().tolist())
        emotions = sorted(relations["target_emotion"].dropna().astype(str).unique().tolist())

    poem_titles = []
    if not poems.empty and "title" in poems.columns:
        poem_titles = sorted(poems["title"].dropna().astype(str).unique().tolist())

    authors = []
    if not nodes.empty:
        authors = sorted(nodes[nodes["type"] == "author"]["label"].dropna().astype(str).unique().tolist())
    elif not poems.empty and "author" in poems.columns:
        authors = sorted(poems["author"].dropna().astype(str).unique().tolist())

    return {
        "symbols": symbols,
        "emotions": emotions,
        "poems": poem_titles,
        "authors": authors,
    }


@app.post("/api/graph/search")
def graph_search(payload: GraphSearchRequest):
    center_id = node_id_for_search(payload.search_type, payload.query)
    nodes, edges = graph_records_for_node(center_id, top_k=payload.top_k)
    return {"center_id": center_id, "nodes": nodes, "edges": edges, "evidence": evidence_for_edges(edges)}


@app.post("/api/graph/expand")
def graph_expand(payload: ExpandRequest):
    new_nodes, new_edges = graph_records_for_node(payload.node_id, top_k=payload.top_k)
    node_map = {node["id"]: node for node in payload.current_nodes + new_nodes}
    edge_map = {(edge.get("source"), edge.get("target"), edge.get("type"), edge.get("poem_id")): edge for edge in payload.current_edges + new_edges}
    edges = list(edge_map.values())
    return {"center_id": payload.node_id, "nodes": list(node_map.values()), "edges": edges, "evidence": evidence_for_edges(edges)}


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest):
    text = payload.poem_text.strip()
    if not text:
        return {"symbols": [], "emotions": [], "relations": [], "nodes": [], "edges": [], "similar_poems": []}
    nlp = cached_nlp()
    symbols = extract_symbols_from_text(text, nlp)
    symbols = filter_symbol_records(symbols, nlp, emotion_words=emotion_category_set(), limit=MAX_USER_SYMBOLS)
    emotions = extract_emotions_from_text(text, nlp, load_emotion_lexicon())
    relations = extract_user_relations(text, symbols, emotions)
    nodes, edges = build_user_poem_graph(symbols, emotions, relations)
    return {
        "symbols": symbols,
        "emotions": emotions,
        "relations": relations,
        "nodes": nodes,
        "edges": edges,
        "similar_poems": find_similar_poems(text),
    }


@app.post("/api/generate")
def generate(payload: GenerateRequest):
    relations_df = filtered_relations()
    valid_symbols = [normalize_symbol_text(symbol) for symbol in payload.symbols if is_public_symbol_label(symbol)]
    valid_emotions = sorted(set(payload.emotions) & set(relations_df["target_emotion"].dropna().astype(str))) if not relations_df.empty else []
    selected = relations_df[(relations_df["source_symbol"].isin(valid_symbols)) | (relations_df["target_emotion"].isin(valid_emotions))] if not relations_df.empty else pd.DataFrame()
    relation_strings = [f"{row.source_symbol} -> {row.target_emotion}" for row in selected.head(10).itertuples(index=False)]
    snippets = []
    if not selected.empty and "context_snippet" in selected.columns:
        snippets_series = selected["context_snippet"].dropna().astype(str)
        snippets = snippets_series[~snippets_series.str.lower().isin(["", "nan", "none"])].head(5).tolist()
    prompt = build_generation_prompt(valid_symbols, valid_emotions, relation_strings, snippets, style=payload.style, length=payload.length)
    result = generate_poem_with_openrouter(prompt)
    result["similar_poems"] = find_similar_poems(result["poem"], top_k=5) if result.get("poem") else []
    return result


@app.post("/api/random-walk")
def random_walk_generate(payload: RandomWalkRequest):
    symbol_nodes = [
        node for node in payload.nodes
        if node.get("type") != "symbol" or is_public_symbol_label(node.get("label", symbol_label_from_id(node.get("id", ""))))
    ]
    node_ids = {node.get("id") for node in symbol_nodes}
    edges = [edge for edge in payload.edges if edge.get("source") in node_ids and edge.get("target") in node_ids]
    path = random_walk_on_visible_graph(symbol_nodes, edges, steps=payload.steps)
    walk_labels = unique_walk_labels(labels_for_path(path, symbol_nodes))
    if len(walk_labels) < 2:
        return {"path": walk_labels, "poem": "", "message": "The graph needs at least one symbol-emotion connection before a random walk can generate an alternate poem."}
    prompt = build_random_walk_prompt(payload.poem_text, walk_labels, payload.relations)
    result = generate_poem_with_openrouter(prompt)
    result["path"] = walk_labels
    return result
