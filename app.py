import streamlit as st
import pandas as pd
from textblob import TextBlob
from deepface import DeepFace
import tempfile

# ----------------------
# Page config
# ----------------------
st.set_page_config(
    page_title="Emotion Detection & Recommendation",
    page_icon="😊"
)
# ... the rest of your code
