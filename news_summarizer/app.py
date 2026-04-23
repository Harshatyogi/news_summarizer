import streamlit as st
import re
import math
from collections import defaultdict

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="News Summarizer",
    page_icon="📰",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Serif 4', serif;
}

.main { background-color: #faf8f3; }

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
}

.masthead {
    background: #1a1a2e;
    color: #f5e6c8;
    padding: 2.5rem 3rem 1.5rem;
    margin: -1rem -1rem 2rem -1rem;
    border-bottom: 4px double #c9a84c;
    text-align: center;
}
.masthead h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: 0.05em;
    color: #f5e6c8;
    margin: 0 0 0.2rem 0;
}
.masthead .tagline {
    font-size: 0.9rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #c9a84c;
    font-style: italic;
}

.stat-box {
    background: #1a1a2e;
    color: #f5e6c8;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    text-align: center;
    border-left: 4px solid #c9a84c;
}
.stat-box .stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
    color: #c9a84c;
    line-height: 1;
}
.stat-box .stat-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-top: 0.3rem;
    opacity: 0.8;
}

.article-box {
    background: #fff;
    border: 1px solid #e0d8c8;
    border-radius: 2px;
    padding: 1.6rem 2rem;
    line-height: 1.85;
    font-size: 1rem;
    color: #2c2c2c;
    box-shadow: 2px 2px 0 #e0d8c8;
}

.summary-box {
    background: #1a1a2e;
    color: #f5e6c8;
    border-radius: 2px;
    padding: 1.6rem 2rem;
    line-height: 1.85;
    font-size: 1rem;
    box-shadow: 2px 2px 0 #c9a84c;
}

.highlight-sentence {
    background: rgba(201, 168, 76, 0.18);
    border-left: 3px solid #c9a84c;
    padding: 0.15rem 0.4rem;
    margin: 0.2rem 0;
    display: inline;
}

.section-rule {
    border: none;
    border-top: 2px solid #1a1a2e;
    margin: 1.5rem 0 0.5rem;
}

.badge {
    display: inline-block;
    background: #c9a84c;
    color: #1a1a2e;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    margin-bottom: 0.7rem;
}

.top-sentence {
    border-left: 4px solid #c9a84c;
    padding: 0.6rem 1rem;
    margin: 0.5rem 0;
    font-style: italic;
    background: rgba(201, 168, 76, 0.08);
    border-radius: 0 2px 2px 0;
    font-size: 0.95rem;
}

footer { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Masthead ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <h1>📰 THE SUMMARIST</h1>
    <div class="tagline">Extractive News Article Summarization Engine</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACTIVE SUMMARIZATION — TF-IDF + Sentence Scoring (no external libs)
# ══════════════════════════════════════════════════════════════════════════════

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did didn't
do does doesn't doing don't down during each few for from further get got had hadn't
has hasn't have haven't having he he'd he'll he's her here here's hers herself him
himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other ought
our ours ourselves out over own same shan't she she'd she'll she's should shouldn't
so some such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when when's where
where's which while who who's whom why why's will with won't would wouldn't you you'd
you'll you're you've your yours yourself yourselves said also just like one new
""".split())


def tokenize(text: str) -> list[str]:
    return re.findall(r'\b[a-z]+\b', text.lower())


def split_sentences(text: str) -> list[str]:
    # Split on . ! ? followed by whitespace or end, keeping sentence intact
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if len(s.split()) >= 5]


def compute_tf(words: list[str]) -> dict:
    tf = defaultdict(int)
    for w in words:
        if w not in STOPWORDS:
            tf[w] += 1
    total = max(len(words), 1)
    return {w: c / total for w, c in tf.items()}


def compute_idf(sentences: list[str]) -> dict:
    N = len(sentences)
    doc_freq = defaultdict(int)
    for sent in sentences:
        words = set(tokenize(sent)) - STOPWORDS
        for w in words:
            doc_freq[w] += 1
    return {w: math.log(N / (1 + df)) for w, df in doc_freq.items()}


def score_sentences(sentences: list[str], idf: dict) -> list[tuple[float, str]]:
    scored = []
    for sent in sentences:
        words = tokenize(sent)
        tf = compute_tf(words)
        score = sum(tf.get(w, 0) * idf.get(w, 0) for w in tf)
        # Slight position bonus: first sentences get a nudge
        scored.append((score, sent))
    return scored


def summarize(text: str, ratio: float = 0.3) -> tuple[list[str], list[tuple[float, str]]]:
    sentences = split_sentences(text)
    if len(sentences) <= 2:
        return sentences, []

    idf = compute_idf(sentences)
    scored = score_sentences(sentences, idf)

    # Position boost: sentences 0 and 1 get +20% of max score
    max_score = max(s for s, _ in scored) if scored else 1
    boosted = []
    for i, (score, sent) in enumerate(scored):
        bonus = 0.2 * max_score if i < 2 else 0
        boosted.append((score + bonus, sent))

    n = max(1, round(len(sentences) * ratio))
    top_n = sorted(boosted, key=lambda x: x[0], reverse=True)[:n]

    # Preserve original document order
    top_sents_set = {s for _, s in top_n}
    ordered = [s for s in sentences if s in top_sents_set]

    all_scored = sorted(boosted, key=lambda x: x[0], reverse=True)
    return ordered, all_scored


def highlight_original(text: str, summary_sents: list[str]) -> str:
    sentences = split_sentences(text)
    summary_set = set(summary_sents)
    parts = []
    for sent in sentences:
        if sent in summary_set:
            parts.append(f'<span class="highlight-sentence">{sent}</span>')
        else:
            parts.append(sent)
    return ' '.join(parts)


def compression_ratio(original: str, summary: list[str]) -> float:
    orig_words = len(original.split())
    summ_words = sum(len(s.split()) for s in summary)
    return round((1 - summ_words / max(orig_words, 1)) * 100, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    ratio = st.slider("Summary length (% of sentences)", 10, 60, 30, 5) / 100
    show_scores = st.checkbox("Show sentence scores", value=False)
    show_highlight = st.checkbox("Highlight selected sentences", value=True)

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
**Extractive Summarization** selects the most important sentences from the
original article — no paraphrasing.

**Algorithm:**
- TF-IDF sentence scoring
- Position-aware boosting (lead sentences)
- Document-order reconstruction
    """)

    st.markdown("---")
    st.markdown("### 🗞️ Try a sample")
    sample_articles = {
        "— pick one —": "",
        "Tech News": """Artificial intelligence is transforming the way companies operate across every industry. 
Major technology firms are investing billions of dollars into developing large language models and generative AI tools. 
These systems can now write code, draft emails, analyze data, and even create artwork. 
However, experts warn that the rapid adoption of AI comes with significant risks, including job displacement and ethical concerns. 
Governments around the world are scrambling to create regulations that balance innovation with safety. 
The European Union recently passed landmark AI legislation that categorizes AI systems by risk level. 
High-risk applications, such as those used in healthcare or law enforcement, will face strict oversight. 
Meanwhile, researchers continue to push the boundaries of what AI can do, with multimodal models now handling text, images, and audio simultaneously. 
The race for AI supremacy is intensifying, with the United States and China investing heavily in next-generation systems. 
Startups are also entering the field, offering specialized AI tools for niche industries like legal services and financial analysis. 
As AI becomes more capable, questions about consciousness and machine rights are beginning to emerge in philosophical and scientific circles. 
The next decade will likely determine how deeply AI becomes embedded in everyday life.""",

        "Climate Report": """Scientists have issued a stark warning about the accelerating pace of climate change following the release of a major new study. 
The research, conducted over five years and involving more than 400 climate experts worldwide, found that global temperatures have risen faster in the past decade than at any point in recorded history. 
Ice sheets in Greenland and Antarctica are melting at unprecedented rates, contributing to rising sea levels that threaten coastal cities. 
The report highlights that extreme weather events, including hurricanes, wildfires, and floods, are becoming more frequent and more severe. 
Agricultural systems are under pressure as droughts devastate crop yields in parts of Africa and South Asia. 
World leaders are under mounting pressure to cut carbon emissions and transition to renewable energy sources. 
Solar and wind power have seen dramatic cost reductions, making clean energy more accessible than ever before. 
Yet fossil fuel consumption continues to rise globally, driven by growing economies in the developing world. 
The authors of the study call for immediate, coordinated action, warning that failure to act within the next decade could trigger irreversible tipping points. 
Innovations in carbon capture technology offer some hope, but experts say they cannot substitute for direct emissions reductions. 
Public awareness of climate issues has grown significantly, fueled by youth-led movements and high-profile disasters. 
The report concludes that while the challenge is enormous, it remains within human capacity to avert the worst outcomes.""",
    }
    choice = st.selectbox("Sample articles", list(sample_articles.keys()))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

default_text = sample_articles.get(choice, "") if choice != "— pick one —" else ""

article_input = st.text_area(
    "Paste your news article here ✍️",
    value=default_text,
    height=220,
    placeholder="Enter or paste a news article (minimum ~5 sentences for best results)...",
)

run_btn = st.button("✦ Summarize Article", use_container_width=True, type="primary")

if run_btn or (default_text and article_input):
    text = article_input.strip()
    if len(text.split()) < 30:
        st.warning("Please enter a longer article (at least ~30 words).")
    else:
        summary_sents, all_scores = summarize(text, ratio=ratio)

        orig_words = len(text.split())
        summ_words = sum(len(s.split()) for s in summary_sents)
        comp = compression_ratio(text, summary_sents)
        orig_sents = split_sentences(text)

        # ── Stats row ─────────────────────────────────────────────────────
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{orig_words}</div><div class="stat-label">Original Words</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{summ_words}</div><div class="stat-label">Summary Words</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{comp}%</div><div class="stat-label">Compression</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{len(summary_sents)}/{len(orig_sents)}</div><div class="stat-label">Sentences Kept</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Side-by-side comparison ────────────────────────────────────────
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown('<div class="badge">Original Article</div>', unsafe_allow_html=True)
            if show_highlight:
                highlighted = highlight_original(text, summary_sents)
                st.markdown(f'<div class="article-box">{highlighted}</div>', unsafe_allow_html=True)
                st.caption("🟡 Highlighted sentences are included in the summary")
            else:
                st.markdown(f'<div class="article-box">{text}</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="badge">Extractive Summary</div>', unsafe_allow_html=True)
            summary_html = " ".join(summary_sents)
            st.markdown(f'<div class="summary-box">{summary_html}</div>', unsafe_allow_html=True)
            st.caption(f"📉 Reduced to {100 - comp:.0f}% of original length")

        # ── Sentence scores ────────────────────────────────────────────────
        if show_scores and all_scores:
            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
            st.markdown("### 🔢 Sentence Importance Scores")
            st.caption("Top-scoring sentences (TF-IDF weighted)")
            top_display = all_scores[:min(6, len(all_scores))]
            for score, sent in top_display:
                in_summary = sent in set(summary_sents)
                tag = "✦ IN SUMMARY · " if in_summary else "○ "
                short = sent if len(sent) <= 110 else sent[:107] + "…"
                st.markdown(
                    f'<div class="top-sentence">{tag}<strong>{score:.4f}</strong> — {short}</div>',
                    unsafe_allow_html=True,
                )

else:
    st.info("👆 Paste an article above (or pick a sample from the sidebar), then click **Summarize Article**.")