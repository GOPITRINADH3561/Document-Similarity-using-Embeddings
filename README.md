# Document Similarity using Embeddings

## 📘 Overview

This project implements and evaluates different embedding techniques—TF-IDF, Word2Vec (trained and pretrained)—for document similarity in a movie recommendation system.

We processed movie metadata and built recommendation pipelines to suggest the top 5 similar movies for any given title using text similarity methods.

---

## 📦 Dataset

- **Source:** assignment2_data.csv (from Kaggle)
- **Total Movies:** ~4800
- **Columns Used:** `Title`, `Popularity`, `Tagline`, `Overview`
- **Target Column Created:** `Full_Overview` = Tagline + Overview

---

## ⚙️ Tasks and Methodologies

### 🔹 Task 1: Preprocessing

- Removed records missing both Tagline and Overview
- Combined Tagline + Overview into `Full_Overview`
- Cleaned text using regular expressions
- Tokenized using SpaCy (`en_core_web_sm`)
- Removed stopwords and applied optional lemmatization (used in this project)

### 🔹 Task 2: TF-IDF Vectorization (Sparse)

- Used `TfidfVectorizer` from scikit-learn to vectorize `Full_Overview`
- Built a recommendation function using cosine similarity
- Returned top 5 similar movies sorted by popularity

**Query Movies:**  
`Taken`, `Pulp Fiction`, `Mad Max`, `Rain Man`, `Bruce Almighty`

### 🔹 Task 3: Word2Vec Vectorization (Dense)

#### 3.1 Trained Word2Vec
- Trained using Gensim with:
  - Vector size: 200
  - Window: 10
  - Min count: 1
  - Iterations: 15
  - Method: skip-gram
- Averaged word vectors (centroid) for document-level embeddings

#### 3.2 Pretrained Word2Vec
- Used `word2vec-google-news-300` via `gensim.downloader`
- Similarity calculated the same way using centroids
- Compared pretrained vs trained results

### 🔹 Task 4: Word Analogies with t-SNE Visualization

- Visualized word analogies:
  - `king - man + woman ≈ queen`
  - `doctor - man + woman ≈ nurse`
  - `france - spain + madrid ≈ barcelona`
  - `florida - texas + austin ≈ miami`
- Used t-SNE for 2D projection
- Added 20 random words from SpaCy vocab to enrich plot context

---

## 📊 Evaluation Summary

- **TF-IDF**:
  - Vocabulary size: 24,272
  - Captures surface-level similarity
  - Occasionally makes less relevant recommendations

- **Trained Word2Vec**:
  - Captures semantic similarity better than TF-IDF
  - Requires tuning (vector size, window, iterations)

- **Pretrained Word2Vec**:
  - Performed best overall
  - Leverages knowledge from massive corpus (Google News)
  - Recommended for real-world semantic tasks

---

## 📁 Files Included

- `Coded_sol_part1.ipynb`: Tasks 1–3 (TF-IDF, trained Word2Vec, pretrained Word2Vec)
- `Coded_sol_part2.ipynb`: Task 4 (Word analogies with t-SNE)
- `data.csv`: Cleaned movie metadata
- `report.pdf`: Explanation of results, answers to analysis questions
- `README.md`: This file

---

## 🧠 Key Insights

- Pretrained vectors outperform both TF-IDF and trained Word2Vec in semantic tasks
- Lemmatization improves performance by reducing sparsity
- t-SNE adds interpretability to vector space reasoning but is non-deterministic

---

## 🚀 How to Run

1. Install required packages:
```bash
pip install pandas scikit-learn gensim spacy matplotlib seaborn
python -m spacy download en_core_web_sm
