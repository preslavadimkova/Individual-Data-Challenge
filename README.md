---
title: Poetry Graph App
sdk: docker
app_port: 7860
---

# Find the Poet Inside You

This project is a poetry inspiration app built with natural language processing, a knowledge graph, semantic retrieval, and a FastAPI web interface.

Instead of assigning one emotion label to a whole poem, the system extracts symbolic nouns and emotion words from poems. A symbol is connected to an emotion when it appears near an emotion word in the poem. These repeated, evidence-based relations form a graph of how poets place images near emotional language.

The graph is explainable and corpus-based. A relation such as moon with grief does not mean that moon always means grief. It means that moon appeared near grief-related language in one or more poems.

A separate sentence-transformer model is fine-tuned for poem retrieval. It helps the app find similar poems when a user pastes their own poem.

Deployed app: https://preslavadimkova-data-challenge.hf.space

Hugging Face Space: https://huggingface.co/spaces/preslavadimkova/data_challenge

## Features

- Search for symbols, emotions, poems, and authors.
- Explore an interactive graph of direct and expanded connections.
- View real poem examples behind the strongest graph relations.
- Paste a poem and extract possible symbols and emotions.
- Build a small graph from the pasted poem.
- Run a random walk only when the user poem has at least one symbol-emotion connection, with repeated back-and-forth paths filtered out.
- Retrieve similar poems using sentence-transformer embeddings.
- Generate a new poem through OpenRouter using approved symbols, emotions, or graph paths.
- View similar poems after generation as extra inspiration.
- Use a faster pink/white browser frontend with a Cytoscape graph canvas instead of Streamlit reruns.

## Project Structure

- app
- app/api.py
- app/symbol_filter.py
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
- .dockerignore
- Dockerfile
- PROJECT_PIPELINE_REPORT.md
- README.md
- requirements-hf.txt
- requirements.txt

Most pipeline code is inside the notebooks so the workflow is easy to trace. The shared symbol cleanup policy lives in app/symbol_filter.py and is used by both the notebooks and the FastAPI app. The main web app is served by app/api.py, with frontend files in app/static. The old Streamlit app remains in app/streamlit_app.py as a reference implementation.

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

Notebook 02 runs spaCy over the corpus and can take time. It extracts symbol candidates broadly, then applies the shared post-extraction cleanup policy. This policy normalizes candidates, blocks offensive terms with better-profanity, keeps concrete common nouns using spaCy POS and morphology checks, rejects adjectives, verbs, pronouns, proper nouns, abstract/meta nouns, archaic forms, and other noise, and removes low-support symbols using corpus statistics. It does not use a hardcoded symbol allowlist.

Notebook 03 extracts symbol-emotion relations from the cleaned symbols, then filters relations again before graph construction so blocked symbols do not enter the graph nodes, graph edges, or app data. Notebook 04 fine-tunes MiniLM and creates poem embeddings, so it is usually the slowest.

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

After changing the symbol filtering policy, rerun notebooks 02 and 03 to regenerate the cleaned symbol table and graph files. Then rerun notebook 05 so the reported metrics match the stricter symbol set. Notebook 04 does not need to be rerun unless you want to retrain the retrieval model.

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

The app also applies the shared symbol filter defensively at runtime. Search results, graph expansion, analysis output, random walks, and generation inputs only surface symbols that pass the public symbol policy and exist in the cleaned graph data. The Generate autocomplete requires choosing an approved option instead of accepting arbitrary typed symbols.

## OpenRouter Generation

Poem generation uses OpenRouter. Set your key before launching the app.

- PowerShell environment variable name: OPENROUTER_API_KEY
- Example value: your key from OpenRouter
- Launch command: python -m uvicorn app.api:app --host 127.0.0.1 --port 8000

The default model is google/gemma-4-31b-it:free.

You can override the model with the OPENROUTER_MODEL environment variable.

If no key is set, the app still works for graph search, poem analysis, and retrieval. Generation will show that an API key is required.

## Hugging Face Docker Deployment

The repo is configured for Hugging Face Spaces with Docker. The README metadata sets `sdk: docker` and `app_port: 7860`, and the Dockerfile starts the FastAPI app with uvicorn on port 7860.

Use `requirements-hf.txt` for the deployed Space. It installs the CPU PyTorch wheel, spaCy, the spaCy English model, sentence-transformers, and better-profanity. The `.dockerignore` keeps notebooks, raw data, logs, and development-only files out of the Docker upload, while keeping the processed data and saved sentence-transformer model needed by the app.

In the Hugging Face Space settings, add these variables:

- `OPENROUTER_API_KEY`: add as a secret.
- `OPENROUTER_MODEL`: add as a normal variable if you want to override the default model.

After code or data changes, upload or push the updated project to the Space. Hugging Face will rebuild the Docker image automatically.

## Evaluation

The evaluation notebook currently reports the following results.

- Symbol extraction: 108 true positives, 45 false positives, 132 false negatives, 0.71 precision, 0.45 recall, and 0.55 F1.
- Emotion extraction: 221 true positives, 34 false positives, 0 false negatives, 0.87 precision, 1.00 recall, and 0.93 F1.
- App search coverage checks for common graph queries.

The evaluation folder keeps only the annotation file used to calculate the symbol and emotion metrics.

- blind_gold_poem_sample.csv

Metric tables are calculated inside the notebook and are not saved as separate CSV files. The current labels are objective proxy annotations, not expert human annotations. They are useful for transparent first-pass validation and can be replaced with manual labels later.

The current results show a clear trade-off. Symbol extraction is reasonably precise but conservative after the stricter cleanup rules, so it produces cleaner public-facing symbols while missing many valid gold symbols. Emotion extraction is the strongest evaluated component, with complete recall and high F1, although the 34 false positives show that the lexicon can still over-detect broad emotional language. After changes to the emotion lexicon, symbol extraction logic, or symbol cleanup thresholds, rerun notebooks 02, 03, and then 05 so the processed graph files and evaluation metrics reflect the current pipeline.

## Notes For Git

The raw dataset, processed CSV files, embeddings, and saved model can be large. If needed, use .gitignore or Git LFS for large files. Runtime logs such as uvicorn-8000.log and uvicorn-8000.err.log are not needed and are ignored. The notebooks and app code are the most important files for showing the project pipeline.
