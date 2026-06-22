"""Modern, premium Streamlit frontend for the OrnithoAI bird species classifier."""

import io
import json
import logging
import os
import requests
import streamlit as st
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_app")

# Page config - use wide layout for premium feel and side-by-side comparison
st.set_page_config(
    page_title="OrnithoAI - Klasyfikator Ptaków",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS INJECTION ---
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global font override */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Gradient header */
    .header-title {
        background: linear-gradient(135deg, #10b981, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0px;
        text-align: center;
    }
    
    .header-subtitle {
        color: #9ca3af;
        font-weight: 400;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 25px;
    }

    /* Cards */
    .premium-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        margin-bottom: 25px;
    }

    @media (prefers-color-scheme: light) {
        .premium-card {
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid rgba(0, 0, 0, 0.05);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.05);
        }
        .header-subtitle {
            color: #4b5563;
        }
    }

    /* Step Cards Grid */
    .step-grid {
        display: flex;
        gap: 20px;
        margin-bottom: 30px;
        flex-wrap: wrap;
    }

    .step-box {
        flex: 1;
        min-width: 250px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .step-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
        border-color: rgba(16, 185, 129, 0.3);
    }

    @media (prefers-color-scheme: light) {
        .step-box {
            background: rgba(0, 0, 0, 0.02);
            border: 1px solid rgba(0, 0, 0, 0.04);
        }
        .step-box:hover {
            border-color: rgba(16, 185, 129, 0.3);
        }
    }

    .step-icon {
        font-size: 2rem;
        margin-bottom: 10px;
    }

    .step-title {
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 8px;
    }

    .step-desc {
        font-size: 0.9rem;
        color: #9ca3af;
    }

    @media (prefers-color-scheme: light) {
        .step-desc {
            color: #6b7280;
        }
    }

    /* Top Winner Match */
    .winner-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(6, 182, 212, 0.15));
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(16, 185, 129, 0.1);
    }

    .winner-badge {
        display: inline-block;
        background: #10b981;
        color: white;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 4px 12px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }

    .winner-name {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 5px 0 15px 0;
        background: linear-gradient(135deg, #ffffff, #e2e8f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    @media (prefers-color-scheme: light) {
        .winner-name {
            background: linear-gradient(135deg, #1f2937, #4b5563);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    }

    .winner-conf {
        font-size: 1.5rem;
        font-weight: 700;
        color: #06b6d4;
    }

    /* Custom progress bar styles */
    .custom-bar-container {
        margin-bottom: 16px;
    }

    .custom-bar-labels {
        display: flex;
        justify-content: space-between;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 6px;
    }

    .custom-bar-outer {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
        width: 100%;
    }

    @media (prefers-color-scheme: light) {
        .custom-bar-outer {
            background-color: rgba(0, 0, 0, 0.07);
        }
    }

    .custom-bar-inner {
        height: 100%;
        border-radius: 8px;
        transition: width 0.8s cubic-bezier(0.1, 0.8, 0.2, 1);
    }
    
    /* Sidebar enhancements */
    .sidebar-section {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    @media (prefers-color-scheme: light) {
        .sidebar-section {
            background: rgba(0, 0, 0, 0.02);
            border: 1px solid rgba(0, 0, 0, 0.03);
        }
    }

    /* Grid layout for Encyclopedia search results */
    .encyclopedia-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 16px;
        margin-top: 20px;
    }

    .encyclopedia-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s, border-color 0.2s;
    }

    .encyclopedia-card:hover {
        transform: translateY(-2px);
        border-color: rgba(16, 185, 129, 0.3);
        background: rgba(255, 255, 255, 0.05);
    }

    @media (prefers-color-scheme: light) {
        .encyclopedia-card {
            background: rgba(0, 0, 0, 0.015);
            border: 1px solid rgba(0, 0, 0, 0.04);
        }
        .encyclopedia-card:hover {
            background: rgba(0, 0, 0, 0.025);
            border-color: rgba(16, 185, 129, 0.3);
        }
    }

    .encyclopedia-card-title {
        font-weight: 700;
        font-size: 1.1rem;
        color: #10b981;
        margin-bottom: 4px;
    }

    .encyclopedia-card-sub {
        font-size: 0.85rem;
        color: #9ca3af;
        font-style: italic;
        margin-bottom: 12px;
    }

    @media (prefers-color-scheme: light) {
        .encyclopedia-card-sub {
            color: #6b7280;
        }
    }

    .encyclopedia-card-actions {
        display: flex;
        gap: 10px;
        font-size: 0.85rem;
    }

    .encyclopedia-card-actions a {
        text-decoration: none;
        color: #06b6d4;
        font-weight: 600;
    }

    .encyclopedia-card-actions a:hover {
        color: #0891b2;
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

# Run CSS Injection
inject_custom_css()

# --- BACKEND CONNECTION SETUP ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Diagnostic check for backend connection
@st.cache_data(ttl=5)
def check_backend_health(url):
    try:
        response = requests.get(f"{url}/health", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

backend_healthy = check_backend_health(BACKEND_URL)

# --- BIRD NAMES DICTIONARY MAPPING ---
names_file = os.path.join(os.path.dirname(__file__), "bird_names_pl.json")
try:
    with open(names_file, encoding="utf-8") as f:
        BIRD_NAMES_MAPPING = json.load(f)
except Exception as e:
    logger.error("Failed to load bird names mapping: %s", str(e))
    BIRD_NAMES_MAPPING = {}

FAMILY_TRANSLATIONS = {
    "Albatross": "Albatrosy (Albatrosses)",
    "Auklet": "Auklety (Auklets)",
    "Blackbird": "Kosy i Kacyki (Blackbirds)",
    "Bunting": "Lazurniki i Trznadle (Buntings)",
    "Cardinal": "Kardynały (Cardinals)",
    "Catbird": "Miauczyki (Catbirds)",
    "Cormorant": "Kormorany (Cormorants)",
    "Flycatcher": "Muchołówki (Flycatchers)",
    "Gull": "Mewy (Gulls)",
    "Hummingbird": "Kolibry (Hummingbirds)",
    "Jay": "Sójki (Jays)",
    "Kingfisher": "Zimorodki (Kingfishers)",
    "Loon": "Nury (Loons)",
    "Nuthatch": "Kowaliki (Nuthatches)",
    "Oriole": "Wilgi (Orioles)",
    "Pelican": "Pelikany (Pelicans)",
    "Pewee": "Kontusie (Pewees)",
    "Puffin": "Maskonury (Puffins)",
    "Sparrow": "Pasówki i Wróble (Sparrows)",
    "Swallow": "Jaskółkowate (Swallows)",
    "Tanager": "Tanagry (Tanagers)",
    "Tern": "Rybitwy (Terns)",
    "Thrasher": "Drozdniki (Thrashers)",
    "Vireo": "Wireonki (Vireos)",
    "Warbler": "Lasówki (Warblers)",
    "Woodpecker": "Dzięciołowate (Woodpeckers)",
    "Wren": "Strzyżyki (Wrens)"
}

# --- PANEL BOCZNY (SIDEBAR) ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1452570053594-1b985d6ea890?w=300&auto=format&fit=crop&q=80", width="content")
    st.markdown('<div class="header-title" style="font-size: 1.8rem; text-align: left;">OrnithoAI</div>', unsafe_allow_html=True)
    st.caption("Inteligentny asystent rozpoznawania ptaków")
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### ℹ️ O aplikacji")
    st.write(
        "Aplikacja wykorzystuje sieć neuronową ResNet dostrojoną dwufazowo "
        "na zbiorze **CUB-200-2011** w celu klasyfikacji gatunków ptaków "
        "na podstawie zdjęć."
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🛠️ Zespół projektowy (SUML)")
    st.markdown("- **Michał Pavlovs** (s26701)")
    st.markdown("- **Kacper Kowieski** (s27794)")
    st.markdown("- **Michał Wereszczyński** (s27570)")
    st.markdown('</div>', unsafe_allow_html=True)

    # Status połączenia
    if backend_healthy:
        st.success("🟢 Połączono z API")
    else:
        st.error("🔴 Brak połączenia z API")

    # Reset button
    if st.button("🔄 Resetuj aplikację", use_container_width=True):
        st.session_state.active_image_bytes = None
        st.session_state.active_image_name = None
        st.rerun()

# --- GŁÓWNY NAGŁÓWEK ---
st.markdown('<div class="header-title">🐦 OrnithoAI</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Przeanalizuj sylwetkę i upierzenie ptaka przy użyciu Deep Learning</div>', unsafe_allow_html=True)

# --- SEKCJA INSTRUKTAŻOWA (STEP CARDS) ---
st.markdown("""
<div class="step-grid">
    <div class="step-box">
        <div class="step-icon">📸</div>
        <div class="step-title">1. Wybierz zdjęcie</div>
        <div class="step-desc">Wgraj własne zdjęcie JPG/PNG lub wybierz jeden z gotowych przykładów poniżej.</div>
    </div>
    <div class="step-box">
        <div class="step-icon">🧠</div>
        <div class="step-title">2. Analiza głęboka</div>
        <div class="step-desc">Model ResNet analizuje detale upierzenia, kształt dzioba i sylwetkę ptaka.</div>
    </div>
    <div class="step-box">
        <div class="step-icon">📊</div>
        <div class="step-title">3. Wyniki & Edukacja</div>
        <div class="step-desc">Zobacz 5 najbardziej pasujących gatunków i dowiedz się o nich więcej w Wikipedii.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- TABS LAYOUT ---
tab_classifier, tab_encyclopedia = st.tabs(["🔍 Klasyfikator AI", "📚 Baza gatunków (200 klas)"])

with tab_classifier:
    # --- ZDJĘCIA PRZYKŁADOWE (EXAMPLES) ---
    EXAMPLES = {
        "Miauczyk szary": {
            "url": "https://images.unsplash.com/photo-1734630155396-2c476389a948?w=600&auto=format&fit=crop&q=80",
            "description": "Miauczyk szary",
            "filename": "miauczyk.jpg"
        },
        "Europejski rudzik": {
            "url": "https://images.unsplash.com/photo-1591608971362-f08b2a75731a?q=80&w=880&auto=format&fit=crop&q=80",
            "description": "Drobny ptak z czerwoną piersią",
            "filename": "rudzik.jpg"
        },
        "Ziębnik purpurowy": {
            "url": "https://images.unsplash.com/photo-1663204166873-3ecd09eb3d91?w=600&auto=format&fit=crop&q=80",
            "description": "Ziębnik purpurowy",
            "filename": "finch.jpg"
        },
        "Kruk zwyczajny": {
            "url": "https://images.unsplash.com/photo-1682467612877-f9b5e55ff2af?q=80&w=764&auto=format&fit&q=80",
            "description": "Kruk zwyczajny",
            "filename": "kruk.jpg"
        }
    }

    # Session state initialization
    if "active_image_bytes" not in st.session_state:
        st.session_state.active_image_bytes = None
    if "active_image_name" not in st.session_state:
        st.session_state.active_image_name = None

    st.write("### ⚡ Szybki test na przykładach")
    cols = st.columns(len(EXAMPLES))
    for idx, (name, info) in enumerate(EXAMPLES.items()):
        with cols[idx]:
            st.image(info["url"], width="content", caption=info["description"])
            if st.button(f"Wybierz: {name}", key=f"ex_btn_{idx}", use_container_width=True):
                with st.spinner("Pobieranie zdjęcia przykładowego..."):
                    try:
                        res = requests.get(info["url"], timeout=10)
                        if res.status_code == 200:
                            st.session_state.active_image_bytes = res.content
                            st.session_state.active_image_name = info["filename"]
                            st.rerun()
                        else:
                            st.error("Nie udało się pobrać zdjęcia przykładowego.")
                    except Exception as e:
                        st.error(f"Błąd pobierania: {str(e)}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.write("---")
    st.markdown("<br>", unsafe_allow_html=True)

    # Helper function to format species names nicely
    def format_species_name(name: str) -> str:
        # Handle folder name fallback like "119.House_Sparrow"
        if "." in name:
            name = name.split(".", 1)[-1]
        return name.replace("_", " ").strip().capitalize()

    # Helper to render custom colored progress bars
    def render_progress_bar(label: str, confidence: float):
        # Determine color scale
        if confidence >= 0.70:
            color = "linear-gradient(90deg, #10b981, #059669)" # Strong Green
        elif confidence >= 0.35:
            color = "linear-gradient(90deg, #f59e0b, #d97706)" # Amber Orange
        else:
            color = "linear-gradient(90deg, #6b7280, #4b5563)" # Soft Grey
            
        percentage = confidence * 100
        st.markdown(
            f'<div class="custom-bar-container">'
            f'<div class="custom-bar-labels"><span>{label}</span><span>{percentage:.1f}%</span></div>'
            f'<div class="custom-bar-outer">'
            f'<div class="custom-bar-inner" style="width: {percentage}%; background: {color};"></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # --- SEKCJA GŁÓWNA (WGRYWANIE / ANALIZA) ---
    if not backend_healthy:
        st.error(
            "⚠️ Błąd połączenia: Nie można nawiązać komunikacji z serwerem backendowym API.\n\n"
            "Upewnij się, że kontener backendu działa i pomyślnie wczytał pliki wag modelu."
        )
    else:
        # File Uploader
        uploaded_file = st.file_uploader(
            "Przeciągnij i upuść lub wybierz z dysku zdjęcie ptaka", 
            type=["jpg", "jpeg", "png"],
            help="Obsługiwane formaty: PNG, JPG, JPEG."
        )
        
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            if st.session_state.active_image_bytes != file_bytes:
                st.session_state.active_image_bytes = file_bytes
                st.session_state.active_image_name = uploaded_file.name
                st.rerun()

        # If we have an active image (uploaded or selected from example)
        if st.session_state.active_image_bytes is not None:
            try:
                image = Image.open(io.BytesIO(st.session_state.active_image_bytes))
                
                # Create two columns (left: image card, right: results card)
                main_col1, main_col2 = st.columns([1, 1], gap="large")
                
                with main_col1:
                    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                    st.markdown("### 📷 Podgląd zdjęcia")
                    st.image(image, width="content")
                    st.markdown(f"**Nazwa pliku:** `{st.session_state.active_image_name}`")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with main_col2:
                    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                    st.markdown("### 📊 Wyniki klasyfikacji")
                    
                    # Convert image to bytes for API post
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format="JPEG")
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    files = {
                        "image": (
                            st.session_state.active_image_name, 
                            img_byte_arr, 
                            "image/jpeg"
                        )
                    }
                    
                    with st.spinner("Analiza szczegółów upierzenia przez model AI..."):
                        try:
                            response = requests.post(f"{BACKEND_URL}/predict", files=files, timeout=30)
                            
                            if response.status_code == 200:
                                data = response.json()
                                predictions = data.get("predictions", [])
                                
                                if predictions:
                                    # Render Top Match Winner Card
                                    top_pred = predictions[0]
                                    top_name_raw = top_pred["species"]
                                    top_name = format_species_name(top_name_raw)
                                    top_confidence = top_pred["confidence"]
                                    
                                    st.markdown(
                                        f'<div class="winner-card">'
                                        f'<div class="winner-badge">Najbardziej dopasowany</div>'
                                        f'<div class="winner-name">{top_name}</div>'
                                        f'<div class="winner-conf">{top_confidence * 100:.1f}% pewności</div>'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                                    
                                    # Other candidates list
                                    st.markdown("#### Prawdopodobne gatunki:")
                                    for pred in predictions:
                                        name_fmt = format_species_name(pred["species"])
                                        render_progress_bar(name_fmt, pred["confidence"])
                                        
                                    st.markdown("<br>", unsafe_allow_html=True)
                                    
                                    # Wikipedia button
                                    wiki_url = f"https://pl.wikipedia.org/wiki/Special:Search?search={top_name}"
                                    st.link_button(
                                        f"📖 Dowiedz się więcej o: {top_name} (Wikipedia)",
                                        wiki_url,
                                        use_container_width=True,
                                        type="primary"
                                    )
                                    
                                else:
                                    st.warning("Serwer nie zwrócił żadnych predykcji.")
                            else:
                                st.error(
                                    f"Błąd API Backend (Status {response.status_code}): "
                                    f"{response.text}"
                                )
                        except requests.exceptions.Timeout:
                            st.error("Przekroczono limit czasu połączenia. Backend nie odpowiedział w wyznaczonym czasie.")
                        except Exception as e:
                            st.error(f"Nie udało się wysłać zapytania: {str(e)}")
                            
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Nie udało się załadować obrazu: {str(e)}")

with tab_encyclopedia:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("### 📚 Encyklopedia obsługiwanych gatunków")
    st.write(
        "Nasz model potrafi zidentyfikować **200 różnych klas ptaków**. Poniżej możesz wyszukiwać "
        "gatunki po nazwie polskiej lub angielskiej, przejść do artykułów na Wikipedii oraz obejrzeć galerie zdjęć."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Search bar
    search_query = st.text_input("🔍 Wyszukaj ptaka po nazwie polskiej lub angielskiej", "")

    # Clean and filter bird names
    filtered_birds = []
    for key, pl_name in BIRD_NAMES_MAPPING.items():
        eng_name = key.split(".", 1)[-1].replace("_", " ")
        if not search_query or search_query.lower() in pl_name.lower() or search_query.lower() in eng_name.lower():
            filtered_birds.append((pl_name, eng_name))

    # Show search results
    if search_query:
        st.subheader(f"🔍 Wyniki wyszukiwania ({len(filtered_birds)})")
        if filtered_birds:
            # HTML card display
            html_cards = []
            for pl, eng in filtered_birds:
                wiki_link = f"https://pl.wikipedia.org/wiki/Special:Search?search={pl}"
                google_link = f"https://www.google.com/search?tbm=isch&q={pl}"
                card_html = (
                    f'<div class="encyclopedia-card">'
                    f'<div>'
                    f'<div class="encyclopedia-card-title">{pl}</div>'
                    f'<div class="encyclopedia-card-sub">{eng}</div>'
                    f'</div>'
                    f'<div class="encyclopedia-card-actions">'
                    f'<a href="{wiki_link}" target="_blank">📖 Wikipedia</a>'
                    f'<a href="{google_link}" target="_blank">📷 Google Grafika</a>'
                    f'</div>'
                    f'</div>'
                )
                html_cards.append(card_html)
            
            # Draw grid
            st.markdown(f'<div class="encyclopedia-grid">{"".join(html_cards)}</div>', unsafe_allow_html=True)
        else:
            st.info("Brak gatunków odpowiadających Twojemu wyszukiwaniu.")
    else:
        # Default view: Group by family
        st.subheader("📁 Kategorie ptaków w naszej bazie")
        
        families = {}
        for key, pl_name in BIRD_NAMES_MAPPING.items():
            eng_name = key.split(".", 1)[-1].replace("_", " ")
            family = eng_name.split()[-1] if eng_name.split() else "Inne"
            if family not in families:
                families[family] = []
            families[family].append((pl_name, eng_name))

        # Sort families alphabetically
        sorted_families = sorted(families.items())
        
        # Split into columns for balanced layout
        cols = st.columns(2, gap="medium")
        for idx, (family, members) in enumerate(sorted_families):
            col_idx = idx % 2
            family_pl = FAMILY_TRANSLATIONS.get(family, f"{family}s")
            
            with cols[col_idx]:
                with st.expander(f"🪶 {family_pl} ({len(members)} gatunków)"):
                    for pl, eng in sorted(members):
                        wiki_link = f"https://pl.wikipedia.org/wiki/Special:Search?search={pl}"
                        google_link = f"https://www.google.com/search?tbm=isch&q={pl}"
                        st.markdown(
                            f"**{pl}** — *{eng}* &nbsp;&nbsp;|&nbsp;&nbsp; "
                            f"[Wikipedia 📖]({wiki_link}) &nbsp;&nbsp;|&nbsp;&nbsp; "
                            f"[Google Grafika 📷]({google_link})"
                        )
                        st.divider()

# Log streamlit load
stream_version = st.__version__
logger.info("Streamlit App loaded using version %s", stream_version)
