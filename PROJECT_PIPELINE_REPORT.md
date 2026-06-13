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

Symbol extraction uses spaCy with the en_core_web_sm model. The pipeline identifies noun chunks and nouns, then reduces noun chunks to their head nouns so long phrases become cleaner symbolic candidates. For example, instead of keeping cold November night as a full phrase, the extractor keeps night. Candidates are extracted broadly first, then cleaned with a shared post-extraction filtering module used by both the notebooks and the app. The filter normalizes symbol text, blocks offensive terms with better-profanity, keeps only concrete common nouns using spaCy POS and morphology checks, rejects adjectives, verbs, pronouns, determiners, adverbs, auxiliary forms, and proper nouns, and removes abstract, meta, archaic, clinical, and low-support terms using lexical and corpus-statistical rules.

This design avoids a hardcoded symbol allowlist. It also prevents illogical or unsafe outputs, such as emotion words, abstract concepts, archaic pronouns, body-part noise, or offensive language being treated as public-facing poetic symbols. A small denylist is used only for non-profanity artifacts that external tools do not reliably catch, such as archaic forms.

Emotion extraction uses a poetic emotion lexicon stored in data/emotion_lexicon/poetic_emotion_lexicon.json. The lexicon includes many categories, such as grief, loneliness, longing, love, fear, hope, joy, anger, despair, nostalgia, wonder, peace, shame, guilt, anxiety, melancholy, resilience, confusion, regret, gratitude, faith, doubt, freedom, oppression, mortality, and transcendence. The lexicon was expanded to improve coverage of poetic emotional language. It is only used to detect emotional words. It does not manually say which symbols belong to which emotions.

The outputs are saved as data/processed/extracted_symbols.csv and data/processed/extracted_emotions.csv.

## Relation Extraction and Graph Construction

The third notebook, notebooks/03_graph_construction.ipynb, implements the core research logic. For each poem, the system compares the token positions of symbols and emotion words. A relation is created only when the distance is short enough. The current project uses a maximum distance of 10 tokens, meaning a symbol and emotion must appear near each other.

This short window makes the graph more precise. A relation such as moon with grief does not mean that moon always symbolizes grief. It means that in one or more poems, moon appeared near grief-related language. The relation is therefore suggestive and evidence-based, not a fixed interpretation.

The relation file stores the symbol, emotion, poem title, author, token distance, line distance, and a context snippet. Before saving, relations are filtered through the same symbol policy so low-quality or blocked symbols do not enter the downstream graph data. The cleaned relation table is saved as data/processed/symbol_emotion_edges.csv.

After relation extraction, the project builds a knowledge graph with four node types: symbols, emotions, poems, and authors. The graph includes edges for nearby emotions, poem symbols, poem emotion words, and authorship. Because graph construction starts from the cleaned relation table, blocked symbols are removed from graph_nodes.csv, graph_edges.csv, and the app visualization JSON instead of only being hidden in the interface. The node and edge tables are saved as data/processed/graph_nodes.csv and data/processed/graph_edges.csv. A JSON version is saved as outputs/graphs/poetry_graph.json for app visualization.

## Deep Learning Retrieval Component

The fourth notebook, notebooks/04_sentence_transformer_training.ipynb, fine-tunes the all-MiniLM-L6-v2 sentence-transformer model.

The model is trained for poem similarity retrieval, not for generation and not for graph construction. Weak training pairs are created from the corpus. Poems are treated as similar when they share signals such as author, tags, dominant emotion categories, strong symbol-emotion relations, or shared symbols. Negative pairs are sampled from poems that do not share these features.

After training, the model is saved in models/sentence_transformer_poetry. Poem embeddings are saved to data/processed/poem_embeddings.npy, with metadata in data/processed/poem_embedding_metadata.csv. These embeddings let the app find similar poems for a user's pasted poem.

## Evaluation

The fifth notebook, notebooks/05_evaluation.ipynb, evaluates the pipeline using saved outputs and objective proxy annotations. These labels are not expert literary annotations, but they provide a transparent first validation layer. The evaluation folder keeps only blind_gold_poem_sample.csv, while the notebook calculates metric tables live. The notebook currently reports symbol extraction precision, recall, and F1; emotion extraction precision, recall, and F1; and app search coverage for common graph queries. Retrieval and relation-quality samples are no longer part of the current evaluation notebook.

The current symbol extraction evaluation produced 108 true positives, 45 false positives, and 132 false negatives, giving 0.71 precision, 0.45 recall, and 0.55 F1. This shows that the stricter post-extraction cleanup makes the system reasonably precise but conservative. The app now surfaces cleaner and safer symbols, but it misses many valid gold symbols, so recall remains the main weakness.

The current emotion extraction evaluation produced 221 true positives, 34 false positives, and 0 false negatives, giving 0.87 precision, 1.00 recall, and 0.93 F1. This makes emotion extraction the strongest evaluated part of the pipeline. The controlled lexicon captures all gold emotion categories in the sample, while the false positives show that some broad emotional words can still trigger extra categories.

## FastAPI Browser Application

The final application is implemented as a FastAPI backend in app/api.py with a browser frontend in app/static. The interface is titled Find The Poet Inside You and has three main sections. The older Streamlit implementation remains in app/streamlit_app.py as a reference, but the FastAPI frontend is the main user interface.

The deployed application is available at https://preslavadimkova-data-challenge.hf.space, with the Hugging Face Space repository at https://huggingface.co/spaces/preslavadimkova/data_challenge.

The FastAPI app uses the same symbol cleanup policy as the notebooks as a defensive runtime layer. The options endpoint, graph search, graph expansion, poem analysis, random walk generation, and poem generation all avoid surfacing blocked symbol nodes. This keeps the app consistent even if an older processed CSV is still present locally before the notebooks are rerun.

In Search Graph, the user searches for a symbol, emotion, poem, or author. The result is shown as an interactive graph centered on the searched term. Direct connections are shown first, and clicking a node expands the graph from that node. A legend explains the colors, and example snippets from real poems show the evidence behind relations.

In Analyze My Poem, the user pastes their own poem. The app extracts found symbols and emotions, filters extracted symbols with the shared policy, builds a small graph for the user poem, and retrieves similar poems. The results remain hidden until the user analyzes a poem, then appear smoothly. The graph uses weighted lines, with thickness representing relation weight. It also includes a random walk button. The random walk only uses connected nodes, avoids repeated two-node bouncing, and displays the graph path separately from the generated poem. If there is no symbol-emotion connection, the interface explains that a connected path is needed instead of showing a failure.

In Generate a Poem, the user selects multiple approved symbols and emotions from custom searchable dropdowns, chooses a style and length, and generates a new poem. The autocomplete does not create chips from arbitrary typed text, which prevents unapproved symbols from entering prompts. After generation, the app also shows similar poems from the retrieval model for additional inspiration. Generation uses OpenRouter through the OPENROUTER_API_KEY environment variable. No local generation model is trained.

The application is also configured for Hugging Face Spaces with Docker. The Dockerfile runs the FastAPI app on port 7860, requirements-hf.txt installs the lighter deployment dependency set, and .dockerignore excludes notebooks, raw data, logs, and development-only files from the deployed image.

## Current Outputs

So far, the project has produced the main pipeline artifacts: cleaned poems, extracted symbols, extracted emotions, symbol-emotion relations, graph nodes and edges, graph JSON, poem similarity pairs, poem embeddings, a fine-tuned sentence-transformer model, an evaluation annotation file, and a working FastAPI browser app.

Overall, the project shows an end-to-end NLP pipeline for creative writing support. The corpus provides real poetic evidence, the graph makes symbolic-emotional patterns explorable, the sentence-transformer supports semantic poem recommendation, and the browser app turns the pipeline into an interactive poetry inspiration tool.
