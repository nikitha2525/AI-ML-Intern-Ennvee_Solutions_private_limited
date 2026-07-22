# 🎯 AI Mock Interview Platform

An intelligent, full-stack mock interview web application that evaluates candidate answers in real time using **NLP-based semantic similarity scoring**. The system combines **Sentence Transformers**, **NLTK preprocessing**, and **Cosine Similarity** to compare a candidate's spoken/typed response against ideal reference answers — going beyond simple keyword matching to understand actual *meaning*.

---

## 📌 Overview

Traditional interview prep tools rely on keyword matching, which fails to capture the semantic intent of an answer. This project solves that by:

- Encoding both the **user's answer** and the **ideal answer** into dense vector embeddings using a pretrained Sentence Transformer model (`paraphrase-MiniLM-L6-v2`)
- Cleaning and normalizing text using **NLTK** (tokenization, stopword removal, lemmatization)
- Computing **Cosine Similarity** between embeddings to generate a semantic closeness score
- Presenting results through a **Flask** backend, **PostgreSQL** database, and interactive **Chart.js** visualizations

---

## ✨ Features

- 🎤 **50 structured interview questions** across 5 categories: HR, Python, SQL, Machine Learning, and Data Science
- 🧠 **Semantic answer scoring** using Sentence-BERT embeddings instead of exact-match/keyword scoring
- 🧹 **NLTK preprocessing pipeline** — tokenization, stopword removal, lemmatization/stemming before scoring
- 📊 **Cosine similarity scoring engine** producing a 0–1 (or %) match score per answer
- 💾 **PostgreSQL** persistence — stores user sessions, answers, scores, and historical attempts
- 🔁 **Session-based mock interview flow** — sequential questions, timed responses, instant feedback
- 🗂️ **Category-wise performance breakdown** to identify weak areas
- 🌐 **Flask REST API** backend serving scoring and question logic

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Backend Framework | Flask (Python) |
| NLP Embeddings | Sentence Transformers (`paraphrase-MiniLM-L6-v2`) |
| Text Preprocessing | NLTK (tokenizer, stopwords, lemmatizer) |
| Similarity Engine | Cosine Similarity (scikit-learn / NumPy) |
| Database | PostgreSQL |
| Visualization | Chart.js |
| Frontend | HTML, CSS, JavaScript |

---

## 🧠 How the Scoring Works

```
User Answer ─┐
             ├─► NLTK Preprocessing ─► Sentence Transformer Encoding ─► Embedding Vector
Ideal Answer─┘                                                              │
                                                                             ▼
                                                          Cosine Similarity(user_vec, ideal_vec)
                                                                             │
                                                                             ▼
                                                                  Similarity Score (0–100%)
```

**Step-by-step:**
1. Raw text (user answer + reference/ideal answer) is cleaned using NLTK:
   - Lowercasing
   - Tokenization (`word_tokenize`)
   - Stopword removal (`nltk.corpus.stopwords`)
   - Lemmatization (`WordNetLemmatizer`)
2. Cleaned text is passed through the Sentence Transformer model to generate fixed-size dense embeddings.
3. Cosine similarity is computed between the two embedding vectors:

   ```
   cosine_similarity = (A · B) / (‖A‖ × ‖B‖)
   ```

4. The resulting score is normalized/scaled and mapped to a feedback band (e.g., Excellent / Good / Needs Improvement).

---

## 📁 Project Structure

```
ai-mock-interview/
│
├── app.py                     # Flask application entry point
├── requirements.txt           # Python dependencies
├── config.py                  # DB and app configuration
│
├── models/
│   └── db_models.py           # PostgreSQL ORM models (User, Session, Answer, Score)
│
├── nlp/
│   ├── preprocess.py          # NLTK cleaning pipeline
│   ├── embedder.py            # Sentence Transformer embedding logic
│   └── similarity.py          # Cosine similarity scoring engine
│
├── data/
│   └── questions.json         # 50 structured questions (HR, Python, SQL, ML, DS)
│
├── static/
│   ├── css/
│   ├── js/
│   └── charts/                # Chart.js dashboard scripts
│
├── templates/
│   ├── index.html
│   ├── interview.html
│   └── dashboard.html
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/nikitha2525/ai-mock-interview.git
cd ai-mock-interview
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt** should include:
```
flask
psycopg2-binary
sentence-transformers
nltk
scikit-learn
numpy
```

### 4. Download NLTK resources
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 5. Configure PostgreSQL
Update `config.py` with your database credentials:
```python
SQLALCHEMY_DATABASE_URI = "postgresql://username:password@localhost:5432/mock_interview_db"
```

### 6. Run the application
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

---

## 🧪 Example Usage

```python
from nlp.preprocess import clean_text
from nlp.embedder import get_embedding
from nlp.similarity import cosine_score

user_answer = "A primary key uniquely identifies each record in a table."
ideal_answer = "A primary key is a column that uniquely identifies every row in a database table."

cleaned_user = clean_text(user_answer)
cleaned_ideal = clean_text(ideal_answer)

score = cosine_score(get_embedding(cleaned_user), get_embedding(cleaned_ideal))
print(f"Similarity Score: {score * 100:.2f}%")
```

---

## 📊 Question Categories

| Category | Number of Questions |
|---|---|
| HR / Behavioral | 10 |
| Python | 10 |
| SQL | 10 |
| Machine Learning | 10 |
| Data Science | 10 |

---

## 🚀 Future Enhancements

- [ ] Add speech-to-text support for voice-based answers
- [ ] Integrate GPT-based feedback generation alongside similarity scoring
- [ ] Add difficulty-adaptive question selection
- [ ] Deploy on Render/Vercel with a managed PostgreSQL instance
- [ ] Add multi-language question support

---

## 👩‍💻 Author

**Nikitha**
B.Tech AI & Data Science | PSR Engineering College
[GitHub: nikitha2525](https://github.com/nikitha2525)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
