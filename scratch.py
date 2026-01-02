import streamlit as st
import librosa
import tensorflow as tf
import time
import numpy as np
import pandas as pd
import os
import hashlib
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

st.set_page_config(
    page_title="Speech-to-Text + Sentiment",
    layout="wide",
)

st.markdown(
    """
<style>
* { box-sizing: border-box; }

.block-container {
  max-width: 1120px;
  padding-top: 2.0rem;
  padding-bottom: 2.0rem;
}

.panel {
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.03);
  border-radius: 18px;
  padding: 16px;
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.section-title {
  font-size: 1.02rem;
  font-weight: 650;
}

.muted {
  opacity: 0.72;
  font-size: 0.92rem;
}

.kv {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.tag {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 999px;
  padding: 6px 10px;
  font-size: 0.86rem;
}

.stTextArea textarea { border-radius: 14px; }
.stExpander { border-radius: 14px; }
[data-testid="stMetricValue"] { font-size: 1.55rem; }

div:empty { display: none; }
</style>
""",
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""
if "last_run_id" not in st.session_state:
    st.session_state.last_run_id = None
if "last_text_id" not in st.session_state:
    st.session_state.last_text_id = None

FW_MODEL_ID = "small.en"
FW_DEVICE = "cpu"
FW_COMPUTE = "int8"

FT_DIR = "finetuned_sentiment"
FT_TOKENIZER_DIR = os.path.join(FT_DIR, "tokenizer")
FT_MODEL_DIR = os.path.join(FT_DIR, "model")

HITL_DIR = "hitl_data"
HITL_FILE = os.path.join(HITL_DIR, "labels.csv")

LABELS_UI = ["Negative", "Neutral", "Positive"]

def stable_text_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def stable_run_id(audio_bytes):
    h = hashlib.sha256()
    h.update(audio_bytes)
    h.update(FW_MODEL_ID.encode())
    return h.hexdigest()[:16]

def append_hitl_row(row):
    os.makedirs(HITL_DIR, exist_ok=True)
    df_new = pd.DataFrame([row])
    if os.path.exists(HITL_FILE):
        df_old = pd.read_csv(HITL_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["id"], keep="last")
    else:
        df_all = df_new
    df_all.to_csv(HITL_FILE, index=False)

@st.cache_resource
def load_whisper():
    return WhisperModel(FW_MODEL_ID, device=FW_DEVICE, compute_type=FW_COMPUTE)

@st.cache_resource
def load_sentiment():
    tokenizer = AutoTokenizer.from_pretrained(FT_TOKENIZER_DIR)
    model = TFAutoModelForSequenceClassification.from_pretrained(FT_MODEL_DIR)
    return tokenizer, model

@st.cache_data
def load_audio(file_bytes):
    import io
    audio, _ = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True)
    return audio

def transcribe(audio):
    segments, _ = whisper_model.transcribe(audio, language="en")
    return " ".join(s.text.strip() for s in segments)

def analyze_sentiment(text):
    enc = sent_tokenizer(text, return_tensors="tf", truncation=True, max_length=512)
    out = sent_model(**enc)
    probs = tf.nn.softmax(out.logits[0], axis=-1).numpy()
    return LABELS_UI[int(np.argmax(probs))], probs

whisper_model = load_whisper()
sent_tokenizer, sent_model = load_sentiment()

st.markdown("# Speech-to-Text + Sentiment")
st.markdown("<div class='muted'>Manual transcription • Human labeling • CSV logging</div>", unsafe_allow_html=True)

tabs = st.tabs(["Analyze", "History"])

with tabs[0]:
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        with st.container():
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Audio</div>", unsafe_allow_html=True)

            uploaded = st.file_uploader(
                "Upload",
                type=["mp3", "wav", "m4a"],
                label_visibility="collapsed",
                key="audio_upload",
            )

            if uploaded:
                st.audio(uploaded)
            else:
                st.markdown("<div class='muted'>Upload an audio file</div>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            start = c1.button("Start transcription", disabled=uploaded is None)
            reset = c2.button("Reset", disabled=uploaded is None and not st.session_state.last_transcript)

            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        with st.container():
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Configuration</div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="kv">
                  <div class="tag">Whisper: small.en</div>
                  <div class="tag">Device: cpu</div>
                  <div class="tag">Compute: int8</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    if reset:
        st.session_state.last_transcript = ""
        st.session_state.last_run_id = None
        st.session_state.last_text_id = None

    text = ""

    if start and uploaded:
        audio_bytes = uploaded.getvalue()
        run_id = stable_run_id(audio_bytes)
        if run_id != st.session_state.last_run_id:
            with st.spinner("Transcribing..."):
                audio = load_audio(audio_bytes)
                text = transcribe(audio)
            st.session_state.last_transcript = text
            st.session_state.last_run_id = run_id
            st.session_state.last_text_id = stable_text_id(text)
        else:
            text = st.session_state.last_transcript
    elif st.session_state.last_transcript:
        text = st.session_state.last_transcript

    if text:
        pred_label, probs = analyze_sentiment(text)
        conf = float(np.max(probs))
        tid = st.session_state.last_text_id

        tcol, scol = st.columns([1.6, 1], gap="large")

        with tcol:
            with st.container():
                st.markdown("<div class='panel'>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Transcript</div>", unsafe_allow_html=True)
                st.text_area(
                    "Transcript",
                    text,
                    height=260,
                    label_visibility="collapsed",
                    key=f"transcript_{tid}",
                )
                st.markdown("</div>", unsafe_allow_html=True)

        with scol:
            with st.container():
                st.markdown("<div class='panel'>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Sentiment</div>", unsafe_allow_html=True)
                st.metric("Model result", pred_label)
                st.markdown(f"<div class='muted'>Confidence: {conf:.3f}</div>", unsafe_allow_html=True)

                human = st.radio(
                    "Human label",
                    LABELS_UI,
                    index=LABELS_UI.index(pred_label),
                    horizontal=True,
                    key=f"human_{tid}",
                )

                if st.button("Save to labels.csv", use_container_width=True):
                    append_hitl_row({
                        "id": tid,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "text": text,
                        "pred_label": pred_label.lower(),
                        "pred_conf": conf,
                        "p_neg": float(probs[0]),
                        "p_neu": float(probs[1]),
                        "p_pos": float(probs[2]),
                        "human_label": human.lower(),
                        "whisper_model": FW_MODEL_ID,
                    })
                    st.success("Saved")

                st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    with st.container():
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Run history</div>", unsafe_allow_html=True)

        if not st.session_state.history:
            st.markdown("<div class='muted'>No runs yet</div>", unsafe_allow_html=True)

        for i, r in enumerate(st.session_state.history):
            with st.expander(r["timestamp"]):
                st.text_area(
                    "Transcript",
                    r["text"],
                    height=140,
                    label_visibility="collapsed",
                    key=f"hist_{i}_{r['run_id']}",
                )

        st.markdown("</div>", unsafe_allow_html=True)
