import streamlit as st
import pandas as pd
from textblob import TextBlob
from deepface import DeepFace
import cv2
import tempfile

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("emotions_data.csv")

emotion_df = load_data()

# -----------------------------
# Text Emotion Detection
# -----------------------------
def detect_text_emotion(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.5:
        return "Happy"
    elif polarity > 0.1:
        return "Hopeful"
    elif polarity < -0.5:
        return "Depressed"
    elif polarity < -0.1:
        return "Sad"
    else:
        return "Neutral"

# -----------------------------
# Face Emotion Detection
# -----------------------------
def detect_face_emotion(image_path):
    result = DeepFace.analyze(
        img_path=image_path,
        actions=['emotion'],
        enforce_detection=False
    )
    return result[0]['dominant_emotion']

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(emotion):
    row = emotion_df[emotion_df["Emotion"].str.lower() == emotion.lower()]
    if not row.empty:
        return row.iloc[0]["Song"], row.iloc[0]["Quote"]
    else:
        return "No song found", "No quote found"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Emotion Detection & Song Recommendation", page_icon="🎵")
st.title("🎵 Emotion Detection & Song Recommendation System")

st.markdown("Detect emotion from **text or face** and get a **song + motivational quote**.")

# =============================
# TEXT EMOTION SECTION
# =============================
st.header("📝 Emotion Detection from Text")

user_text = st.text_area("Enter how you feel:")

if st.button("Detect Emotion from Text"):
    if user_text.strip() != "":
        emotion = detect_text_emotion(user_text)
        song, quote = recommend(emotion)

        st.success(f"Detected Emotion: **{emotion}**")
        st.write(f"🎶 **Song:** {song}")
        st.write(f"💬 **Quote:** {quote}")
    else:
        st.warning("Please enter some text.")

# =============================
# FACE EMOTION SECTION
# =============================
st.header("📷 Emotion Detection from Face Image")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    if st.button("Detect Emotion from Face"):
        with st.spinner("Analyzing face emotion..."):
            emotion = detect_face_emotion(temp_path)
            song, quote = recommend(emotion)

            st.success(f"Detected Emotion: **{emotion}**")
            st.write(f"🎶 **Song:** {song}")
            st.write(f"💬 **Quote:** {quote}")

# -----------------------------
st.markdown("---")
st.caption("AI Project | Emotion-Based Recommendation System")
