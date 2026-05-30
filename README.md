# Find the Poet Inside You

This project is a poetry inspiration app built with natural language processing, a knowledge graph, semantic retrieval, and a Streamlit interface.

Instead of assigning one emotion label to a whole poem, the system extracts symbolic nouns and emotion words from poems. A symbol is connected to an emotion when it appears near an emotion word in the poem. These repeated, evidence-based relations form a graph of how poets place images near emotional language.

The graph is explainable and corpus-based. A relation such as moon with grief does not mean that moon always means grief. It means that moon appeared near grief-related language in one or more poems.

A separate sentence-transformer model is fine-tuned for poem retrieval. It helps the app find similar poems when a user pastes their own poem.

## Features

- Search for symbols, emotions, poems, and authors.
- Explore an interactive graph of direct and expanded connections.
- View real poem examples behind graph relations.
- Paste a poem and extract possible symbols and emotions.
- Build a small graph from the pasted poem.
- Retrieve similar poems using sentence-transformer embeddings.
- Generate a new poem through OpenRouter using selected symbols, emotions, or graph paths.

## Project Structure

- app
- app/streamlit_app.py
- data/emotion_lexicon
- data/processed
- data/raw
- models/sentence_transformer_poetry
- notebooks/01_eda.ipynb
- notebooks/02_symbol_emotion_extraction.ipynb
- notebooks/03_graph_construction.ipynb
- notebooks/04_sentence_transformer_training.ipynb
- notebooks/05_evaluation.ipynb
- outputs/evaluation
- outputs/graphs
- PROJECT_PIPELINE_REPORT.md
- README.md
- requirements.txt

Most pipeline code is inside the notebooks so the workflow is easy to trace. The Streamlit app is in app/streamlit_app.py.

The folders models/sentence_transformer_poetry/1_Pooling and models/sentence_transformer_poetry/2_Normalize are part of the saved SentenceTransformer model. The normalize folder can look empty, but it is referenced by modules.json and should stay in place.

## Setup

From inside the project folder, install the requirements and download the spaCy English model.

- pip install -r requirements.txt
- python -m spacy download en_core_web_sm

The raw dataset should be placed at data/raw/PoetryFoundationData.csv.

## Run The Pipeline

Run the notebooks in this order.

- notebooks/01_eda.ipynb
- notebooks/02_symbol_emotion_extraction.ipynb
- notebooks/03_graph_construction.ipynb
- notebooks/04_sentence_transformer_training.ipynb
- notebooks/05_evaluation.ipynb

Notebook 02 runs spaCy over the corpus and can take time. Notebook 04 fine-tunes MiniLM and creates poem embeddings, so it is usually the slowest.

The main generated files are listed below.

- data/processed/poems_clean.csv
- data/processed/extracted_symbols.csv
- data/processed/extracted_emotions.csv
- data/processed/symbol_emotion_edges.csv
- data/processed/graph_nodes.csv
- data/processed/graph_edges.csv
- data/processed/poem_similarity_pairs.csv
- data/processed/poem_embeddings.npy
- data/processed/poem_embedding_metadata.csv
- models/sentence_transformer_poetry
- outputs/graphs/poetry_graph.json
- outputs/evaluation

## Run The App

After notebooks 01, 02, and 03 are complete, graph search will work. After notebook 04 is complete, similar-poem retrieval will also work.

From the project folder, run the Streamlit app.

- streamlit run app/streamlit_app.py

On Windows PowerShell, first move into the project folder.

- cd path/to/poetry_graph_app
- streamlit run app/streamlit_app.py

## OpenRouter Generation

Poem generation uses OpenRouter. Set your key before launching the app.

- PowerShell environment variable name: OPENROUTER_API_KEY
- Example value: your key from OpenRouter
- Launch command: streamlit run app/streamlit_app.py

The default model is openai/gpt-oss-120b:free.

You can override the model with the OPENROUTER_MODEL environment variable.

If no key is set, the app still works for graph search, poem analysis, and retrieval. Generation will show that an API key is required.

## Evaluation

The evaluation notebook reports the following results.

- Symbol extraction precision, recall, and F1.
- Emotion extraction precision, recall, and F1.
- Symbol-emotion relation quality on sampled edges.
- Token-distance threshold comparison.
- Retrieval precision in the top five results and mean relevance in the top five results.

The evaluation folder keeps only the annotation files used to calculate these results.

- blind_gold_poem_sample.csv
- relation_annotation_sample.csv
- retrieval_evaluation_sample.csv

Metric tables are calculated inside the notebook and are not saved as separate CSV files.

The current labels are objective proxy annotations, not expert human annotations. They are useful for transparent first-pass validation and can be replaced with manual labels later.

## Notes For Git

The raw dataset, processed CSV files, embeddings, and saved model can be large. If needed, use .gitignore or Git LFS for large files. The notebooks and app code are the most important files for showing the project pipeline.
