"""Streamlit frontend for the OrnithoAI bird species classifier."""

import io
import logging
import os

import requests
import streamlit as st
from PIL import Image

# Setup simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_app")

# Page config
st.set_page_config(
    page_title="OrnithoAI - Bird Classifier",
    page_icon="🐦",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- PANEL BOCZNY (SIDEBAR) ---
with st.sidebar:
    st.header("ℹ️ O aplikacji")
    st.info(
        "Projekt zaliczeniowy z przedmiotu SUML. "
        "Aplikacja wykorzystuje model głębokiego uczenia "
        "do klasyfikacji obrazów i rozpoznawania gatunków ptaków."
    )
    st.divider()
    st.subheader("🛠️ Zespół projektowy")
    st.write("- Michał Pavlovs s26701")
    st.write("- Kacper Kowieski s27794")
    st.write("- Michał Wereszczyński s27570")

# --- GŁÓWNY NAGŁÓWEK ---
st.title("🐦 OrnithoAI")
st.caption("Deep Learning Bird Species Classifier")
st.write("---")

# --- SEKCJA INSTRUKTAŻOWA ---
st.markdown("""
### Odkryj sekrety ornitologii 🌿
Wgraj wyraźne zdjęcie ptaka, a nasz system przeanalizuje jego sylwetkę i upierzenie,
aby błyskawicznie dopasować go do bazy znanych gatunków.
""")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.success("📸 **1. Wgraj**\n\nDodaj zdjęcie w formacie JPG lub PNG.")
with col_b:
    st.warning("🧠 **2. Przeanalizuj**\n\nBackend wyciąga cechy obrazu.")
with col_c:
    st.info("📊 **3. Sprawdź**\n\nOtrzymujesz top dopasowań.")

st.write("---")

# --- POŁĄCZENIE Z BACKENDEM ---
# Retrieve backend endpoint from environment
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Diagnostic check for backend connection
try:
    health_response = requests.get(f"{BACKEND_URL}/health", timeout=3)
    BACKEND_CONNECTED = health_response.status_code == 200
except requests.exceptions.RequestException:
    BACKEND_CONNECTED = False

# --- LOGIKA GŁÓWNA ---
if not BACKEND_CONNECTED:
    st.error(
        "⚠️ Connection Error: Unable to connect to the backend server. "
        "Please ensure the backend container is running and healthy."
    )
else:
    # Main file upload area
    uploaded_file = st.file_uploader("Upload a bird image to classify", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Load and display image
            image = Image.open(uploaded_file)

            # Use two equal columns for a clean side-by-side layout
            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                st.subheader("Uploaded Image")
                # Scale image nicely keeping aspect ratio
                st.image(image, width="content")

            with col2:
                st.subheader("Classification Results")

                # Convert image to bytes for POST request
                img_byte_arr = io.BytesIO()
                # Determine correct format (default JPEG)
                img_format = image.format if image.format else "JPEG"
                image.save(img_byte_arr, format=img_format)
                img_byte_arr = img_byte_arr.getvalue()

                # Prepare payload
                files = {"image": (uploaded_file.name, img_byte_arr, f"image/{img_format.lower()}")}

                with st.spinner("Analyzing plumage and details..."):
                    try:
                        response = requests.post(f"{BACKEND_URL}/predict", files=files, timeout=30)

                        if response.status_code == 200:
                            data = response.json()
                            predictions = data.get("predictions", [])

                            if predictions:
                                # Top prediction highlighted in a metric card
                                top_prediction = predictions[0]
                                st.metric(
                                    label="Top Species Match",
                                    value=top_prediction["species"].title(),
                                    delta=f"{top_prediction['confidence'] * 100:.1f}% Confidence"
                                )

                                st.write("---")
                                st.write("**Top Species Candidates**")

                                # Show top-5 candidates with native Streamlit progress bars
                                for pred in predictions:
                                    species = pred["species"].title()
                                    confidence = pred["confidence"]

                                    # Native progress bar with clean labels
                                    st.write(f"**{species}** • {confidence * 100:.1f}%")
                                    st.progress(confidence)
                            else:
                                st.warning("No predictions returned from the server.")
                        else:
                            st.error(
                                f"Backend API Error (Status {response.status_code}): "
                                f"{response.text}"
                            )
                    except requests.exceptions.Timeout:
                        st.error("Request timed out. The backend took too long to respond.")
                    except (requests.exceptions.RequestException, ValueError) as e:
                        st.error(f"Inference request failed: {str(e)}")

        except OSError as e:
            st.error(f"Could not open uploaded image: {str(e)}")

stream_version = st.__version__
logger.info("Streamlit App loaded using version %s", stream_version)
