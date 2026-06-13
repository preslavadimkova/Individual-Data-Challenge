import re
from functools import lru_cache

import pandas as pd
from better_profanity import profanity


MIN_SYMBOL_OCCURRENCES = 3
MIN_SYMBOL_POEMS = 3
MIN_RELATION_COUNT = 3
MIN_RELATION_POEMS = 3
VALID_SYMBOL_POS = {"NOUN"}
BAD_SYMBOL_ENTITY_LABELS = {"DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY"}
ABSTRACT_SYMBOL_SUFFIXES = ("ness", "tion", "sion", "ment", "ance", "ence", "ity", "ism", "hood", "acy")

GENERIC_SYMBOL_NOUNS = {
    "thing", "things", "time", "times", "day", "days", "way", "ways", "man", "men",
    "woman", "women", "person", "persons", "people", "life", "lives", "world", "worlds",
    "place", "places", "one", "ones", "someone", "everyone", "something",
    "nothing", "everything", "anything", "name", "names", "word", "words", "part",
    "parts", "kind", "sort", "type", "lot", "lots", "bit", "bits", "piece", "pieces",
    "use", "uses", "case", "fact", "form", "forms", "effect", "effects", "result",
    "results", "problem", "problems", "question", "questions", "number", "numbers",
    "example", "examples", "point", "points", "matter", "matters", "object", "objects",
    "stuff", "self", "selves", "else", "today", "tomorrow", "yesterday",
}
TEMPORAL_SYMBOL_WORDS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "novemeber", "december", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday", "morning", "evening",
    "afternoon", "tonight",
}
ARCHAIC_NOISE_WORDS = {
    "thou", "thee", "thy", "thine", "doth", "hath", "hast", "shalt", "wilt", "wert",
    "ye", "yon", "tis", "twas", "ere",
}
ABSTRACT_META_WORDS = {
    "advice", "age", "allegory", "art", "beauty", "career", "concept", "content",
    "control", "discourse", "end", "faith", "glory", "history", "idea", "ideas",
    "language", "law", "meaning", "mind", "narrative", "plan", "poem", "poetry",
    "power", "reason", "scene", "sense", "state", "subject", "thought", "topic",
    "truth", "virtue", "wit", "work",
}
CLINICAL_OR_NOISE_BODY_WORDS = {
    "abdomen",
}


@lru_cache(maxsize=1)
def _load_profanity_words():
    profanity.load_censor_words()
    return True


def normalize_symbol_text(value):
    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value).lower()).strip()
    return re.sub(r"^[^a-z]+|[^a-z]+$", "", text)


def _is_profane(value):
    _load_profanity_words()
    return profanity.contains_profanity(value)


def _has_abstract_suffix(value):
    return any(len(value) > len(suffix) + 3 and value.endswith(suffix) for suffix in ABSTRACT_SYMBOL_SUFFIXES)


def basic_symbol_rejection_reason(value, emotion_words=None):
    symbol = normalize_symbol_text(value)
    emotion_words = emotion_words or set()

    if not symbol or len(symbol) < 3:
        return "too_short"
    if not re.fullmatch(r"[a-z]+", symbol):
        return "non_alpha"
    if symbol in GENERIC_SYMBOL_NOUNS:
        return "generic"
    if symbol in TEMPORAL_SYMBOL_WORDS:
        return "temporal"
    if symbol in ARCHAIC_NOISE_WORDS:
        return "archaic"
    if symbol in ABSTRACT_META_WORDS or _has_abstract_suffix(symbol):
        return "abstract"
    if symbol in CLINICAL_OR_NOISE_BODY_WORDS:
        return "clinical_noise"
    if symbol in emotion_words:
        return "emotion_word"
    if _is_profane(symbol):
        return "profane"
    return ""


def token_symbol_rejection_reason(token, symbol, emotion_words=None):
    reason = basic_symbol_rejection_reason(symbol, emotion_words=emotion_words)
    if reason:
        return reason
    if token.is_stop:
        return "stop_word"
    if token.pos_ not in VALID_SYMBOL_POS:
        return f"pos_{token.pos_.lower()}"
    if token.ent_type_ in BAD_SYMBOL_ENTITY_LABELS:
        return f"entity_{token.ent_type_.lower()}"
    return ""


def symbol_pos_rejection_reasons(symbols, nlp):
    normalized = [normalize_symbol_text(symbol) for symbol in symbols]
    unique_symbols = sorted({symbol for symbol in normalized if symbol})
    reasons = {}
    if not unique_symbols:
        return reasons

    for doc in nlp.pipe(unique_symbols, batch_size=512):
        token = doc[0] if doc else None
        symbol = doc.text
        if token is None:
            reasons[symbol] = "missing_pos"
        elif token.pos_ not in VALID_SYMBOL_POS:
            reasons[symbol] = f"pos_{token.pos_.lower()}"
        elif token.is_stop:
            reasons[symbol] = "stop_word"
        else:
            reasons[symbol] = ""
    return reasons


def symbol_quality_reasons(symbols, nlp, emotion_words=None):
    symbols = [normalize_symbol_text(symbol) for symbol in symbols]
    reasons = {symbol: basic_symbol_rejection_reason(symbol, emotion_words=emotion_words) for symbol in set(symbols)}
    needs_pos = [symbol for symbol, reason in reasons.items() if symbol and not reason]
    pos_reasons = symbol_pos_rejection_reasons(needs_pos, nlp)
    for symbol, reason in pos_reasons.items():
        if reason:
            reasons[symbol] = reason
    return reasons


def symbol_stats(df, symbol_col="symbol", poem_col="poem_id"):
    if df.empty or symbol_col not in df.columns:
        return pd.DataFrame(columns=[symbol_col, "occurrence_count", "poem_count"])
    grouped = (
        df.assign(_symbol=df[symbol_col].map(normalize_symbol_text))
        .dropna(subset=["_symbol"])
        .groupby("_symbol", as_index=False)
        .agg(
            occurrence_count=(symbol_col, "size"),
            poem_count=(poem_col, "nunique") if poem_col in df.columns else (symbol_col, "size"),
        )
        .rename(columns={"_symbol": symbol_col})
    )
    return grouped


def allowed_symbols_from_dataframe(
    df,
    nlp,
    emotion_words=None,
    symbol_col="symbol",
    poem_col="poem_id",
    min_occurrences=MIN_SYMBOL_OCCURRENCES,
    min_poems=MIN_SYMBOL_POEMS,
):
    stats = symbol_stats(df, symbol_col=symbol_col, poem_col=poem_col)
    supported = stats[
        (stats["occurrence_count"] >= min_occurrences)
        & (stats["poem_count"] >= min_poems)
    ][symbol_col].tolist()
    reasons = symbol_quality_reasons(supported, nlp, emotion_words=emotion_words)
    return {symbol for symbol in supported if not reasons.get(symbol)}


def filter_extracted_symbols(symbols_df, nlp, emotion_words=None):
    if symbols_df.empty:
        return symbols_df.copy()
    allowed = allowed_symbols_from_dataframe(
        symbols_df,
        nlp,
        emotion_words=emotion_words,
        symbol_col="symbol",
        poem_col="poem_id",
        min_occurrences=MIN_SYMBOL_OCCURRENCES,
        min_poems=MIN_SYMBOL_POEMS,
    )
    clean = symbols_df.copy()
    clean["symbol"] = clean["symbol"].map(normalize_symbol_text)
    if "lemma" in clean.columns:
        clean["lemma"] = clean["lemma"].map(normalize_symbol_text)
    return clean[clean["symbol"].isin(allowed)].reset_index(drop=True)


def filter_relations(relations_df, nlp, emotion_words=None):
    if relations_df.empty:
        return relations_df.copy()
    allowed = allowed_symbols_from_dataframe(
        relations_df,
        nlp,
        emotion_words=emotion_words,
        symbol_col="source_symbol",
        poem_col="poem_id",
        min_occurrences=MIN_RELATION_COUNT,
        min_poems=MIN_RELATION_POEMS,
    )
    clean = relations_df.copy()
    clean["source_symbol"] = clean["source_symbol"].map(normalize_symbol_text)
    return clean[clean["source_symbol"].isin(allowed)].reset_index(drop=True)


def filter_symbol_records(records, nlp, emotion_words=None, limit=None):
    if not records:
        return []
    reasons = symbol_quality_reasons([row.get("symbol", "") for row in records], nlp, emotion_words=emotion_words)
    rows = []
    seen = set()
    for row in records:
        symbol = normalize_symbol_text(row.get("symbol", ""))
        if not symbol or reasons.get(symbol) or symbol in seen:
            continue
        clean = dict(row)
        clean["symbol"] = symbol
        clean["lemma"] = normalize_symbol_text(clean.get("lemma", symbol))
        rows.append(clean)
        seen.add(symbol)
        if limit and len(rows) >= limit:
            break
    return rows
