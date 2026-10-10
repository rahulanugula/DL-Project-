"""
SMS Spam Detector — Streamlit Web Interface
Phase 6: Interactive web app for real-time spam detection.
"""

import streamlit as st
import numpy as np
import os
import pickle

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
}

.hero-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}

.hero-sub {
    text-align: center;
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

.result-spam {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    animation: pulse 0.6s ease-in-out;
}

.result-ham {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    animation: pulse 0.6s ease-in-out;
}

.result-icon  { font-size: 3rem;   margin-bottom: 0.5rem; }
.result-label { font-size: 1.8rem; font-weight: 700; color: white; }

.confidence-bar-label {
    color: #cbd5e1;
    font-size: 0.9rem;
    margin-top: 0.8rem;
}

@keyframes pulse {
    0%   { transform: scale(0.96); opacity: 0.6; }
    100% { transform: scale(1);    opacity: 1; }
}

.info-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 1rem;
    color: #94a3b8;
    font-size: 0.9rem;
}

.stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: white !important;
    font-size: 1rem !important;
}

div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-size: 1.1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, #1d4ed8);
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4);
}
</style>
""", unsafe_allow_html=True)


# ── Load model + vectorizer ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load the trained Keras model and rebuild the TextVectorization layer."""
    try:
        import tensorflow as tf

        # TF 2.16+ uses standalone keras; tf-keras provides backwards compat
        try:
            import tf_keras as keras_compat
            TextVectorization = keras_compat.layers.TextVectorization
            load_model = keras_compat.models.load_model
        except ImportError:
            from tensorflow.keras.layers import TextVectorization
            load_model = tf.keras.models.load_model

        base_dir    = os.path.join(os.path.dirname(__file__), "..")
        model_path  = os.path.join(base_dir, "models", "spam_model.keras")
        vocab_path  = os.path.join(base_dir, "models", "vocab.txt")
        meta_path   = os.path.join(base_dir, "models", "vec_meta.pkl")

        if not os.path.exists(model_path) or not os.path.exists(vocab_path):
            return None, None, "Model files not found. Please retrain the model."

        model = load_model(model_path)

        # Load hyperparams
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            max_tokens = meta["max_tokens"]
            max_len    = meta["max_len"]
        else:
            max_tokens, max_len = 10_000, 100

        # Rebuild vectorizer from saved vocabulary
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = [line.rstrip("\n") for line in f]

        vectorizer = TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=max_len,
            standardize="lower_and_strip_punctuation",
        )
        vectorizer.set_vocabulary(vocab)

        return model, vectorizer, None

    except Exception as e:
        return None, None, str(e)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">📱 SMS Spam Detector</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Deep Learning · LSTM · NLP — Detect spam messages in real time</p>',
    unsafe_allow_html=True,
)

# ── Model status ───────────────────────────────────────────────────────────────
model, vectorizer, load_error = load_artifacts()

if model is None:
    if load_error:
        st.error(f"❌ **Model load error:** {load_error}")
    else:
        st.warning(
            "⚠️ **Model not found.** Please run the training script first.",
            icon="🧠",
        )
    st.markdown("""
    <div class="info-card">
    <b>How to train the model:</b><br>
    <code>python -X utf8 notebooks/spam_detection.py</code><br><br>
    This saves <code>models/spam_model.keras</code> and <code>models/vectorizer.pkl</code>.<br>
    Then come back here and refresh the page.
    </div>
    """, unsafe_allow_html=True)

# ── Example messages ───────────────────────────────────────────────────────────
st.markdown("#### 💬 Try an example")
col1, col2 = st.columns(2)

spam_examples = [
    "Congratulations! You've won a FREE iPhone. Click now!",
    "URGENT: £1000 prize waiting. Call 09061743939 NOW!",
    "You have been selected for a cash reward. Claim today!",
]
ham_examples = [
    "Hey, are we still meeting at 6 tonight?",
    "Can you send me the notes from today's class?",
    "Don't forget mum's birthday is this weekend!",
]

with col1:
    st.markdown("🚨 **Spam samples**")
    for ex in spam_examples:
        if st.button(ex[:42] + "…", key=f"spamex_{ex[:8]}"):
            st.session_state["user_message"] = ex

with col2:
    st.markdown("✅ **Ham samples**")
    for ex in ham_examples:
        if st.button(ex[:42] + "…", key=f"hamex_{ex[:8]}"):
            st.session_state["user_message"] = ex

st.markdown("---")

# ── Input area ─────────────────────────────────────────────────────────────────
default_text = st.session_state.get("user_message", "")
user_input   = st.text_area(
    "✍️ Enter your message below",
    value=default_text,
    height=120,
    placeholder="Type or paste an SMS message here...",
    key="message_input",
)

check_btn = st.button("🔍  CHECK MESSAGE", use_container_width=True)

# ── Prediction ─────────────────────────────────────────────────────────────────
if check_btn:
    if not user_input.strip():
        st.error("Please enter a message first.")
    elif model is None:
        st.error("Model not loaded. Train the model first (see instructions above).")
    else:
        with st.spinner("Analysing message..."):
            import time
            time.sleep(0.35)
            # Vectorize the message then predict
            vec_input  = vectorizer(np.array([user_input.strip()])).numpy()
            prediction = model.predict(vec_input, verbose=0)[0][0]

        confidence = float(prediction)
        is_spam    = confidence > 0.5
        spam_pct   = round(confidence * 100, 1)
        ham_pct    = round((1 - confidence) * 100, 1)

        st.markdown("<br>", unsafe_allow_html=True)

        if is_spam:
            st.markdown(f"""
            <div class="result-spam">
                <div class="result-icon">🚨</div>
                <div class="result-label">SPAM MESSAGE</div>
                <div class="confidence-bar-label">Spam confidence: <b>{spam_pct}%</b></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-ham">
                <div class="result-icon">✅</div>
                <div class="result-label">NOT SPAM</div>
                <div class="confidence-bar-label">Ham confidence: <b>{ham_pct}%</b></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.metric("🚨 Spam Probability", f"{spam_pct}%")
        c2.metric("✅ Ham Probability",  f"{ham_pct}%")
        st.progress(confidence)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small style='color:#475569'>SMS Spam Detection · Deep Learning Project · LSTM + Keras</small></center>",
    unsafe_allow_html=True,
)
