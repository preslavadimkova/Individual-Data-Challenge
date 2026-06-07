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

POETIC_SYMBOLS = {
    "air", "angel", "animal", "apple", "arm", "ash", "baby", "beach", "bell", "bird",
    "blood", "body", "bone", "book", "branch", "bread", "breath", "bridge", "brother",
    "bull", "candle", "car", "cat", "chair", "child", "church", "city", "clock",
    "cloud", "coat", "color", "corner", "country", "cross", "crow", "cup", "curse",
    "daughter", "dawn", "devil", "dog", "door", "doorway",
    "dream", "dress", "drop", "dust", "earth", "eye", "face", "father", "field",
    "finger", "fire", "fish", "flame", "flower", "foot", "forest", "garden", "ghost",
    "glass", "gold", "grain", "grass", "grave", "ground", "hair", "hand", "hat",
    "hay", "head", "heart", "heath", "hill", "home", "horse", "house", "image",
    "island", "kitchen", "lamp", "land", "leaf", "leg", "lens", "letter", "light",
    "line", "lip", "mirror", "moon", "mother", "mountain", "mouth", "nature", "neck",
    "night", "ocean", "opera", "paper", "path", "photo", "plague", "poet", "print",
    "rain", "river", "road", "rock", "room", "root", "rose", "salt", "school", "sea",
    "season", "seed", "shadow", "ship", "shore", "shoulder", "silk", "sister", "skin",
    "sky", "snow", "son", "song", "sound", "space", "spirit", "star", "stone", "story",
    "street", "study", "sun", "table", "throat", "tide", "tongue", "trail", "tree",
    "valley", "wall", "water", "wave", "wheel", "window", "wind", "wing", "wood",
    "worker", "year",
}
GENERIC_SYMBOL_NOUNS = {
    "thing", "things", "time", "times", "day", "days", "way", "ways", "man", "men",
    "woman", "women", "person", "persons", "people", "life", "lives", "world", "worlds",
    "place", "places", "one", "ones", "someone", "everyone", "something",
    "nothing", "everything", "anything", "name", "names", "word", "words", "part",
    "parts", "kind", "sort", "type", "lot", "lots", "bit", "bits", "piece", "pieces",
    "use", "uses", "case", "fact", "idea", "ideas", "form", "forms", "effect", "effects",
    "result", "results", "problem", "problems", "question", "questions", "number",
    "numbers", "example", "examples", "point", "points", "matter", "matters", "object",
    "objects", "stuff", "self", "selves", "else", "today", "tomorrow", "yesterday",
}
TEMPORAL_SYMBOL_WORDS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "novemeber", "december", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday",
}
ABSTRACT_SYMBOL_SUFFIXES = ("ness", "tion", "sion", "ment", "ance", "ence", "ity", "ism", "ship", "hood", "acy", "ure")
BAD_SYMBOL_ENTITY_LABELS = {"DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY"}

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


def df_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    clean = df.replace({np.nan: None})
    return clean.to_dict("records")


def line_number_from_char(text, char_index):
    return text[:char_index].count("\n") + 1


def normalize_symbol_text(value):
    text = re.sub(r"\s+", " ", str(value).lower()).strip()
    return re.sub(r"^[^a-z]+|[^a-z]+$", "", text)


def emotion_lookup_from_lexicon(lexicon):
    lookup = {}
    for category, words in lexicon.items():
        for word in words:
            lookup.setdefault(str(word).lower(), set()).add(category)
    return lookup


def is_emotion_word(value):
    return str(value).lower().strip() in emotion_word_set()


def is_abstract_symbol(lemma):
    return lemma not in POETIC_SYMBOLS and any(lemma.endswith(suffix) for suffix in ABSTRACT_SYMBOL_SUFFIXES)


def is_valid_symbol_token(token, lemma, emotion_words):
    if not lemma or len(lemma) < 3 or not re.search(r"[a-z]", lemma):
        return False
    if token.pos_ not in {"NOUN", "PROPN"} or token.is_stop:
        return False
    if lemma in GENERIC_SYMBOL_NOUNS or lemma in TEMPORAL_SYMBOL_WORDS:
        return False
    if lemma in emotion_words and lemma not in POETIC_SYMBOLS:
        return False
    if is_abstract_symbol(lemma):
        return False
    if token.ent_type_ in BAD_SYMBOL_ENTITY_LABELS and lemma not in POETIC_SYMBOLS:
        return False
    return True


def symbol_candidate_score(item, frequency):
    symbol = item["symbol"]
    score = 0
    if symbol in POETIC_SYMBOLS:
        score += 3
    if item["source_method"] == "noun_chunk_head":
        score += 1
    if item["pos"] == "NOUN":
        score += 1
    if item["pos"] == "PROPN" and symbol not in POETIC_SYMBOLS:
        score -= 1
    if frequency > 1:
        score += 1
    return score


def extract_symbols_from_text(text, nlp):
    emotion_words = emotion_word_set()
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
        if score < 2:
            continue
        ranked_item = dict(item)
        ranked_item["score"] = score
        current = best_by_symbol.get(item["symbol"])
        if current is None or (ranked_item["score"], -ranked_item["start_token"]) > (current["score"], -current["start_token"]):
            best_by_symbol[item["symbol"]] = ranked_item

    ranked = sorted(best_by_symbol.values(), key=lambda item: (-item["score"], item["start_token"], item["symbol"]))
    rows = []
    for item in ranked[:MAX_USER_SYMBOLS]:
        item = dict(item)
        item["source_method"] = f"{item['source_method']}+ranked"
        item.pop("pos", None)
        item.pop("score", None)
        rows.append(item)
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
        edge_rows.append({"source": f"user_symbol:{relation['source_symbol']}", "target": f"user_emotion:{relation['target_emotion']}", "type": "POSSIBLE_NEAR_EMOTION", "weight": 1, "poem_id": "user_poem", "evidence": relation["context_snippet"]})
    return node_rows, edge_rows


def node_id_for_search(search_type, query):
    nodes = read_csv(GRAPH_NODES_PATH)
    poems = read_csv(POEMS_CLEAN_PATH)
    if nodes.empty:
        return ""
    value = query.lower().strip()
    if search_type in {"symbol", "emotion"}:
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
    return df_records(nodes[nodes["id"].isin(ids)].copy()), df_records(connected.copy())


def evidence_for_edges(edges):
    relations = read_csv(RELATIONS_PATH)
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
        if is_emotion_word(symbol) or symbol == emotion:
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
Do not replace concrete required words with only a mood or implication.
Do not copy the original poem.
Use the random walk as inspiration."""


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
    previous = None
    path = [current]
    for _ in range(steps):
        neighbors = list(adjacency.get(current, set()))
        if previous and len(neighbors) > 1:
            neighbors = [neighbor for neighbor in neighbors if neighbor != previous]
        if not neighbors:
            break
        previous = current
        current = random.choice(neighbors)
        path.append(current)
    return path


def labels_for_path(path, nodes):
    labels = {node["id"]: node.get("label", node["id"].split(":", 1)[-1]) for node in nodes}
    return [labels.get(node_id, node_id.split(":", 1)[-1]) for node_id in path]


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
    relations = read_csv(RELATIONS_PATH)
    if relations.empty:
        return {"symbols": [], "emotions": []}
    return {
        "symbols": sorted(relations["source_symbol"].dropna().astype(str).unique().tolist()),
        "emotions": sorted(relations["target_emotion"].dropna().astype(str).unique().tolist()),
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
    relations_df = read_csv(RELATIONS_PATH)
    selected = relations_df[(relations_df["source_symbol"].isin(payload.symbols)) | (relations_df["target_emotion"].isin(payload.emotions))] if not relations_df.empty else pd.DataFrame()
    relation_strings = [f"{row.source_symbol} -> {row.target_emotion}" for row in selected.head(10).itertuples(index=False)]
    snippets = []
    if not selected.empty and "context_snippet" in selected.columns:
        snippets_series = selected["context_snippet"].dropna().astype(str)
        snippets = snippets_series[~snippets_series.str.lower().isin(["", "nan", "none"])].head(5).tolist()
    prompt = build_generation_prompt(payload.symbols, payload.emotions, relation_strings, snippets, style=payload.style, length=payload.length)
    return generate_poem_with_openrouter(prompt)


@app.post("/api/random-walk")
def random_walk_generate(payload: RandomWalkRequest):
    path = random_walk_on_visible_graph(payload.nodes, payload.edges, steps=payload.steps)
    walk_labels = labels_for_path(path, payload.nodes)
    if len(walk_labels) < 2:
        return {"path": walk_labels, "poem": "", "message": "The graph needs at least one symbol-emotion connection before a random walk can generate an alternate poem."}
    prompt = build_random_walk_prompt(payload.poem_text, walk_labels, payload.relations)
    result = generate_poem_with_openrouter(prompt)
    result["path"] = walk_labels
    return result
