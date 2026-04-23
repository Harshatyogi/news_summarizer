# 📰 The Summarist — News Article Summarizer

A Streamlit app that performs **extractive summarization** on news articles using TF-IDF sentence scoring — **no external NLP libraries required**.

## Features
- ✦ Extractive summarization (selects real sentences, no paraphrasing)
- ✦ TF-IDF scoring + position-aware boosting
- ✦ Side-by-side original vs. summary comparison
- ✦ Highlighted selected sentences in original text
- ✦ Compression stats (words, sentences, % reduced)
- ✦ Adjustable summary length slider (10–60%)
- ✦ Optional sentence importance scores view
- ✦ Built-in sample articles (Tech News, Climate Report)

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

## How It Works

### Extractive Summarization Algorithm

1. **Sentence Splitting** — Splits article into sentences using regex (`.`, `!`, `?`).
2. **TF-IDF Scoring** — Computes term frequency × inverse document frequency for every word in each sentence.
3. **Position Boost** — Gives a 20% bonus to the first two sentences (news articles front-load key info).
4. **Top-N Selection** — Picks the highest-scoring sentences based on the summary ratio slider.
5. **Order Preservation** — Reassembles selected sentences in their original document order.

### No External Libraries
All NLP logic is implemented from scratch using only Python's `re`, `math`, and `collections` modules.

## Project Structure

```
news_summarizer/
├── app.py           # Streamlit app (UI + summarization logic)
├── requirements.txt # Only requires streamlit
└── README.md
```