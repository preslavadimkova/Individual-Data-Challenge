# Poetry Inspiration Knowledge Graph App: Pipeline Report

## Project Goal

This project builds a poetry inspiration application for new poets. The main idea is to discover how poets connect symbolic images with emotional language inside real poems. Instead of classifying each poem with one emotion label, the system extracts symbols and emotion words from the text and connects them when they appear close to each other. These repeated relations form a knowledge graph that users can search, explore, and use as inspiration for their own writing.

The project combines two approaches. The symbol-emotion graph is explainable and corpus-based: every relation comes from actual word positions in poems. A separate deep learning component, a fine-tuned MiniLM sentence-transformer, is used for semantic poem retrieval. This keeps the graph interpretable while still allowing the app to recommend similar poems when a user pastes their own poem.

## Data and Preprocessing

The project uses the Poetry Foundation dataset stored in data/raw/PoetryFoundationData.csv. The preprocessing stage standardizes the dataset into consistent columns for title, author, poem text, and tags. It also creates a unique poem identifier for each poem.

The cleaning step preserves the original poem text because line breaks and structure are important in poetry. At the same time, it creates a cleaned text version for NLP processing. Empty poems are removed, and basic features are added, including token count and line count. The processed dataset is saved as data/processed/poems_clean.csv, which becomes the base file for all later stages.

## Exploratory Data Analysis

The first notebook, notebooks/01_eda.ipynb, explores the corpus before model building. It checks the number of poems, number of authors, missing values, poem length statistics, frequent authors, and frequent tags. It also includes visualizations directly inside the notebook, such as poem length distribution, author counts, tag counts, frequent nouns, emotion categories, and common symbol-emotion relations.

This stage helps explain the dataset and supports later design choices. For example, the poem length distribution shows how much text the NLP pipeline must process, while author and tag frequencies show which voices and topics are most represented in the corpus.

## Symbol and Emotion Extraction

The second notebook, notebooks/02_symbol_emotion_extraction.ipynb, extracts the two main entity types: symbols and emotions.

Symbol extraction uses spaCy with the en_core_web_sm model. The pipeline identifies noun chunks and nouns or proper nouns. To make results cleaner, long noun phrases are reduced to their head nouns in the app logic. For example, instead of showing cold November night, the app shows night. The extraction also removes stopwords, pronouns, generic nouns, month names, very short terms, and words that are clearly emotion words. This prevents illogical outputs such as love or grief being treated as symbols.

Emotion extraction uses a poetic emotion lexicon stored in data/emotion_lexicon/poetic_emotion_lexicon.json. The lexicon includes many categories, such as grief, loneliness, longing, love, fear, hope, joy, anger, despair, nostalgia, wonder, peace, shame, guilt, anxiety, melancholy, resilience, confusion, regret, gratitude, faith, doubt, freedom, oppression, mortality, and transcendence. The lexicon is only used to detect emotional words. It does not manually say which symbols belong to which emotions.

The outputs are saved as data/processed/extracted_symbols.csv and data/processed/extracted_emotions.csv.

## Relation Extraction and Graph Construction

The third notebook, notebooks/03_graph_construction.ipynb, implements the core research logic. For each poem, the system compares the token positions of symbols and emotion words. A relation is created only when the distance is short enough. The current project uses a maximum distance of 10 tokens, meaning a symbol and emotion must appear near each other.

This short window makes the graph more precise. A relation such as moon with grief does not mean that moon always symbolizes grief. It means that in one or more poems, moon appeared near grief-related language. The relation is therefore suggestive and evidence-based, not a fixed interpretation.

The relation file stores the symbol, emotion, poem title, author, token distance, line distance, and a context snippet. It is saved as data/processed/symbol_emotion_edges.csv.

After relation extraction, the project builds a knowledge graph with four node types: symbols, emotions, poems, and authors. The graph includes edges for nearby emotions, poem symbols, poem emotion words, and authorship. The node and edge tables are saved as data/processed/graph_nodes.csv and data/processed/graph_edges.csv. A JSON version is saved as outputs/graphs/poetry_graph.json for app visualization.

## Deep Learning Retrieval Component

The fourth notebook, notebooks/04_sentence_transformer_training.ipynb, fine-tunes the all-MiniLM-L6-v2 sentence-transformer model.

The model is trained for poem similarity retrieval, not for generation and not for graph construction. Weak training pairs are created from the corpus. Poems are treated as similar when they share signals such as author, tags, dominant emotion categories, strong symbol-emotion relations, or shared symbols. Negative pairs are sampled from poems that do not share these features.

After training, the model is saved in models/sentence_transformer_poetry. Poem embeddings are saved to data/processed/poem_embeddings.npy, with metadata in data/processed/poem_embedding_metadata.csv. These embeddings let the app find similar poems for a user's pasted poem.

## Evaluation

The fifth notebook, notebooks/05_evaluation.ipynb, evaluates the pipeline using saved outputs and objective proxy annotations. These labels are not expert literary annotations, but they provide a transparent first validation layer. The evaluation folder keeps only the annotation files, while the notebook calculates metric tables live. The notebook reports symbol extraction precision, recall, and F1; emotion extraction precision, recall, and F1; relation quality on sampled symbol-emotion edges; token-distance threshold comparison; and retrieval quality in the top five results.

The current evaluation shows that emotion extraction and retrieval are the strongest parts of the system. Emotion extraction performs well because it is based on a controlled lexicon. Retrieval also performs well under the proxy relevance measure. Symbol extraction is weaker because poetic symbols are harder to define automatically, and relation quality is limited by the quality of extracted symbols and snippets. This result is useful because it identifies the main source of noise in the knowledge graph.

## Streamlit Application

The final application is implemented in app/streamlit_app.py. It has a landing page titled Find the Poet Inside You and three main sections.

In Search Graph, the user searches for a symbol, emotion, poem, or author. The result is shown as an interactive graph centered on the searched term. Direct connections are shown first, and clicking a node expands the graph from that node. A legend explains the colors, and example snippets from real poems show the evidence behind relations.

In Analyze My Poem, the user pastes their own poem. The app extracts found symbols and emotions, builds a small graph for the user poem, and retrieves similar poems. It also includes a random walk button. The random walk follows the user poem graph and then generates an alternate poem under the heading "How your poem could have also turned out." The prompt asks the model to keep the style of the pasted poem while using the random-walk words.

In Generate a Poem, the user selects multiple symbols and emotions from searchable dropdowns, chooses a style and length, and generates a new poem. Generation uses OpenRouter through the OPENROUTER_API_KEY environment variable. No local generation model is trained.

## Current Outputs

So far, the project has produced the main pipeline artifacts: cleaned poems, extracted symbols, extracted emotions, symbol-emotion relations, graph nodes and edges, graph JSON, poem similarity pairs, poem embeddings, a fine-tuned sentence-transformer model, evaluation files, and a working Streamlit app.

Overall, the project shows an end-to-end NLP pipeline for creative writing support. The corpus provides real poetic evidence, the graph makes symbolic-emotional patterns explorable, the sentence-transformer supports semantic poem recommendation, and the app turns the pipeline into an interactive poetry inspiration tool.

## Next Steps

The most important next improvement is symbol extraction. The system should continue filtering generic nouns, emotion words used as nouns, and abstract words that do not work well as poetic symbols. It could also use a stronger concrete-noun filter so objects, places, animals, natural images, and body images are preferred over vague concepts.

The second improvement is relation quality. Some symbol-emotion edges are noisy because a symbol and emotion may be close in token distance without forming a meaningful poetic association. Future versions should combine token distance with line distance, require clearer context snippets, remove symbol-emotion self-matches, and test more thresholds such as 5, 8, and 10 tokens.

The emotion lexicon should also be reviewed. It works well technically, but some words are ambiguous. Words such as "light," "still," "old," or "fire" can be emotional in some contexts and literal in others. A cleaner lexicon would reduce false emotion matches.

The evaluation can be strengthened with human labels. The current metrics use objective proxy annotations, which are useful for consistency, but a final version should manually annotate 20 to 30 poems for important symbols and 100 relation edges for meaningfulness. This would make the precision, recall, F1, and relation quality scores more persuasive.

Finally, retrieval evaluation can compare the fine-tuned MiniLM model with the original pretrained MiniLM model. A small manual rating set with relevance scores from 0 to 2 would show whether fine-tuning improves poem recommendations.
