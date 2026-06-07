# Find the Poet Inside You

This project is a poetry inspiration app built with natural language processing, a knowledge graph, semantic retrieval, and a FastAPI web interface.

Instead of assigning one emotion label to a whole poem, the system extracts symbolic nouns and emotion words from poems. A symbol is connected to an emotion when it appears near an emotion word in the poem. These repeated, evidence-based relations form a graph of how poets place images near emotional language.

The graph is explainable and corpus-based. A relation such as moon with grief does not mean that moon always means grief. It means that moon appeared near grief-related language in one or more poems.

A separate sentence-transformer model is fine-tuned for poem retrieval. It helps the app find similar poems when a user pastes their own poem.

## Features

- Search for symbols, emotions, poems, and authors.
- Explore an interactive graph of direct and expanded connections.
- View real poem examples behind the strongest graph relations.
- Paste a poem and extract possible symbols and emotions.
- Build a small graph from the pasted poem.
- Run a random walk only when the user poem has at least one symbol-emotion connection.
- Retrieve similar poems using sentence-transformer embeddings.
- Generate a new poem through OpenRouter using selected symbols, emotions, or graph paths.
- Use a faster pink/white browser frontend with a Cytoscape graph canvas instead of Streamlit reruns.

## Project Structure

- app
- app/api.py
- app/static
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

Most pipeline code is inside the notebooks so the workflow is easy to trace. The main web app is served by app/api.py, with frontend files in app/static. The old Streamlit app remains in app/streamlit_app.py as a reference implementation.

The folders models/sentence_transformer_poetry/1_Pooling and models/sentence_transformer_poetry/2_Normalize are part of the saved SentenceTransformer model. The normalize folder can look empty, but it is referenced by modules.json and should stay in place.

## Setup

From inside the project folder, install the requirements and download the spaCy English model.

```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The raw dataset should be placed at data/raw/PoetryFoundationData.csv.

## Run The Pipeline

Run the notebooks in this order.

- notebooks/01_eda.ipynb
- notebooks/02_symbol_emotion_extraction.ipynb
- notebooks/03_graph_construction.ipynb
- notebooks/04_sentence_transformer_training.ipynb
- notebooks/05_evaluation.ipynb

Notebook 02 runs spaCy over the corpus and can take time. It now uses a ranked symbol extractor that prefers noun-chunk heads, TF-IDF-supported terms, and a curated set of poetic image words while filtering generic, temporal, abstract, and emotion-only terms. Notebook 04 fine-tunes MiniLM and creates poem embeddings, so it is usually the slowest.

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

From the project folder, run the FastAPI app.

```powershell
cd "D:\Desktop\Sem6\notebook exercises\poetry_graph_app"
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000 in your browser.

During development, if port 8000 is already in use on Windows PowerShell, stop the old process first.

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty OwningProcess |
  Sort-Object -Unique |
  ForEach-Object { Stop-Process -Id $_ -Force }
```

The API sends no-cache headers for the static frontend files so browser refreshes pick up recent CSS and JavaScript changes.

## OpenRouter Generation

Poem generation uses OpenRouter. Set your key before launching the app.

- PowerShell environment variable name: OPENROUTER_API_KEY
- Example value: your key from OpenRouter
- Launch command: python -m uvicorn app.api:app --host 127.0.0.1 --port 8000

The default model is google/gemma-4-31b-it:free.

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

The current labels are objective proxy annotations, not expert human annotations. They are useful for transparent first-pass validation and can be replaced with manual labels later. After changes to the emotion lexicon or symbol extraction logic, rerun notebooks 02, 03, and then 05 so the processed graph files and evaluation metrics reflect the current pipeline.

## Notes For Git

The raw dataset, processed CSV files, embeddings, and saved model can be large. If needed, use .gitignore or Git LFS for large files. Runtime logs such as uvicorn-8000.log and uvicorn-8000.err.log are not needed and are ignored. The notebooks and app code are the most important files for showing the project pipeline.
