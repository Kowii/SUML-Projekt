# OrnithoAI - Klasyfikator Gatunków Ptaków

OrnithoAI to modułowa, konteneryzowana aplikacja uczenia maszynowego przeznaczona do klasyfikacji gatunków ptaków. Cały stos technologiczny wykorzystuje **FastAPI** do obsługi wysokowydajnych interfejsów REST API, **Streamlit** do prezentacji czystego i przyjaznego dla użytkownika interfejsu (frontend), **PyTorch** do serwowania modeli AI, a także **Docker Compose** do pełnej orkiestracji usług.

---

## 🌟 Kluczowe Funkcjonalności

1. **Pretrenowany Model Bazowy ImageNet**: Usługa treningowa (`trainer`) pobiera pretrenowaną architekturę bazową (domyślnie `resnet50`) za pomocą `torchvision.models`, dostraja ją dwufazowo na zbiorze danych **CUB-200-2011** (obejmującym 200 gatunków ptaków) i zapisuje wagi modelu w celu umożliwienia natychmiastowego wnioskowania.
2. **Dynamiczny, Nieblokujący Start Backends**: Podczas uruchamiania backend w bezpieczny sposób odpytuje o współdzielone pliki modelu z wykorzystaniem standardowego modułu `logging`. Jeśli wagi nie zostaną odnalezione w ciągu **5 minut (300 s)**, sekwencja startowa przerywa działanie, zapobiegając zakleszczeniom (deadlockom).
3. **Nieblokująca Pętla Wnioskowania**: Zadania przetwarzania obrazu i propagacji w przód w PyTorch są delegowane do osobnych wątków roboczych za pomocą `asyncio.to_thread`. Zapobiega to blokowaniu głównej pętli zdarzeń FastAPI, która może bez przeszkód przetwarzać kolejne żądania współbieżne.
4. **Automatyczne Przełączanie na CUDA**: Usługa wnioskowania automatycznie wykrywa obecność procesorów graficznych kompatybilnych z CUDA (`cuda`). Obliczenia są automatycznie kierowane na kartę GPU, a w przypadku jej braku system przezroczyście przełącza się na pracę na procesorze (CPU).
5. **Niezależna Architektura**: Konfiguracje modelu, katalogi, pliki wag oraz adresy URL usług API są w pełni sparametryzowane poprzez standardowy plik zmiennych środowiskowych `.env`. Pozwala to na łatwą zmianę architektury modelu (np. z `resnet50` na `resnet18`) bez modyfikowania kodu rutingu API czy interfejsu.
6. **Docker Healthchecks**: W pliku `docker-compose.yml` zaimplementowano standardowe mechanizmy sprawdzania stanu kontenerów, dzięki czemu aplikacja Streamlit (frontend) uruchamia się dopiero wtedy, gdy FastAPI backend zgłosi status `healthy`.

---

## 📂 Struktura Katalogów Projektu

```
├── data/                          # Przygotowanie i transformacja danych
│   ├── __init__.py
│   ├── dataset.py                 # BirdDataset, augmentacje TRAIN/VAL, random_split
│   ├── bird_names_pl.json         # Mapowanie angielskich nazw klas na polskie dla 200 klas
│   └── raw/                       # Miejsce na surowe obrazy CUB-200-2011
│
├── backend/
│   ├── app/                       # REST API (FastAPI)
│   │   ├── __init__.py
│   │   ├── main.py                # Punkt wejścia FastAPI
│   │   ├── config.py              # Ładowanie ustawień przy użyciu Pydantic Settings
│   │   ├── schemas.py             # Schematy Pydantic dla struktur wejściowych/wyjściowych API
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── health.py          # GET /health
│   │   │   └── predict.py         # POST /predict (asyncio.to_thread — nieblokujące)
│   │   └── services/
│   │       ├── __init__.py
│   │       └── inference.py       # Wnioskowanie z wykrywaniem CUDA/CPU, bezpieczeństwo wątkowe
│   ├── train/                     # Trenowanie i fine-tuning
│   │   ├── __init__.py
│   │   └── train.py               # Dwufazowy fine-tuning: głowica a potem pełny backbone (ResNet + AdamW + AMP)
│   ├── .dockerignore
│   ├── Dockerfile                 # Python 3.13, FastAPI na porcie 8000
│   ├── Dockerfile.trainer         # Kontener jednorazowy do trenowania modelu
│   └── requirements.txt           # fastapi, uvicorn, torch, torchvision, pillow
│
├── frontend/                      #  Interfejs użytkownika (Streamlit)
│   ├── app.py                     # UI z podglądem obrazu, wykresami top-5 i encyklopedią gatunków
│   ├── .dockerignore
│   ├── Dockerfile                 # Python 3.13, Streamlit na porcie 8501
│   └── requirements.txt           # streamlit, requests, pillow
│
├── docs/images/                   # Zrzuty ekranu do dokumentacji
├── models/                        # Wagi modelu (generowane przez trainer, nieśledzone przez git)
├── docker-compose.yml             # Orkiestracja: trainer → backend (healthcheck) → frontend
├── pyproject.toml                 # Konfiguracja Ruff (linter/formatter) i Pylint
├── run.sh                         # Lokalny skrypt uruchomieniowy Linux/macOS
├── .env.example                   # Szablon zmiennych środowiskowych — kopiuj jako .env
└── README.md                      # Ten plik
```
---

## ⚙️ Konfiguracja Środowiska (zmienne .env)

Przed pierwszym uruchomieniem skopiuj szablon:
```bash
cp .env.example .env
```

Plik `.env` kontroluje pełne zachowanie aplikacji bez modyfikowania kodu:

| Zmienna | Domyślna wartość | Opis |
|---------|-----------------|------|
| `MODEL_DIR` | `/models` | Katalog przechowywania wag i pliku klas |
| `MODEL_FILE` | `bird_classifier.pt` | Nazwa pliku wag modelu PyTorch |
| `CLASSES_FILE` | `classes.json` | Plik JSON z nazwami 200 klas |
| `MODEL_NAME` | `resnet50` | Architektura modelu (`resnet50` lub `resnet18`) |
| `BATCH_SIZE` | `64` | Rozmiar batcha treningowego |
| `PHASE1_EPOCHS` | `5` | Liczba epok fazy 1 (tylko głowica) |
| `PHASE2_EPOCHS` | `20` | Liczba epok fazy 2 (pełny fine-tuning) |
| `PHASE1_LR` | `1e-3` | Learning rate fazy 1 |
| `PHASE2_LR` | `1e-4` | Learning rate fazy 2 |
| `BACKEND_URL` | `http://localhost:8000` | Adres backendu używany przez frontend Streamlit |
| `SKIP_TRAIN` | `true` | Pomiń trenowanie przy starcie (gdy wagi już istnieją) |

---

## 🤖 Model ML — Szczegóły

### Architektura

Model bazuje na pretrenowanej sieci **ResNet-50** (wagi ImageNet) dostrojonej dwufazowo na zbiorze **CUB-200-2011** (200 gatunków ptaków, 11 788 obrazów):

- **Faza 1 — trening głowicy** (5 epok, lr=1e-3): zamrożony backbone, trenowana jedynie nowa warstwa `nn.Linear(2048 → 200)`.
- **Faza 2 — pełny fine-tuning** (20 epok, lr=1e-4): odblokowanie wszystkich parametrów, scheduler `CosineAnnealingLR`, optymalizator `AdamW` z `weight_decay=1e-4`, `CrossEntropyLoss` z label smoothing=0.1.

### Augmentacje danych (warstwa `data/dataset.py`)

Zbiór treningowy stosuje augmentacje zapobiegające przeuczeniu: losowe przycięcie (`RandomResizedCrop`), odbicia poziome, rotacja ±20°, zmiana jasności/kontrastu/nasycenia, losowe wymazywanie (`RandomErasing`).

### Wyniki trenowania

Trenowanie przeprowadzono na GPU (CUDA), czas: ~14 minut. Pełny raport per-epoka generowany jest automatycznie do pliku `models/training_report.md`.

| Metryka | Wartość |
|---------|---------|
| Dokładność walidacyjna (Top-1) | **84,09%** |
| Liczba klas | 200 gatunków |
| Zbiór treningowy | 9 431 próbek |
| Zbiór walidacyjny | 2 357 próbek |
| Najlepsza epoka | 25 (faza 2, epoka 20) |

---

## 📦 Zbiór Danych

Projekt opiera się na zbiorze danych **CUB-200-2011** (Caltech-UCSD Birds) zawierającym 11 788 obrazów reprezentujących 200 gatunków ptaków.

1. Pobierz dane z [oficjalnego źródła](https://www.vision.caltech.edu/datasets/cub_200_2011/) lub za pomocą polecenia:
   ```bash
   wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
   tar -xzf CUB_200_2011.tgz -C data/raw/
   ```
2. Oczekiwana struktura folderów: ` data/raw/CUB_200_2011/images/<folder_klasy>/*.jpg`

> **Uwaga:** Zbiór danych ani wagi wytrenowanego modelu nie są przechowywane w repozytorium ze względu na ich duży rozmiar. Przed uruchomieniem backendu należy uruchomić moduł treningowy w celu wygenerowania pliku `models/bird_classifier.pt` (lub ustawić `SKIP_TRAIN=false` w `.env`).

---

## 🚀 Uruchomienie Aplikacji

### 🐳 Opcja A — Docker (rekomendowana)

Upewnij się, że na Twoim komputerze są zainstalowane narzędzia [Docker](https://www.docker.com/) oraz [Docker Compose](https://docs.docker.com/compose/).

Skopiuj szablon zmiennych środowiskowych:
```bash
cp .env.example .env
```

Zbuduj i uruchom wszystkie kontenery jednym poleceniem:
```bash
docker compose up --build -d
```

Powoduje to uruchomienie następującej sekwencji zdarzeń:
1. **Generowanie Modelu**: Usługa `trainer` wykonuje skrypt `backend/train/train.py`, pobiera sieć bazową `resnet50`, przeprowadza uczenie, zapisuje wagi w udostępnionym katalogu `./models` i kończy pracę (chyba że zmienna `SKIP_TRAIN` jest ustawiona na `true`).
2. **Uruchomienie API**: Kontener `backend` oczekuje na pojawienie się pliku wag z procesu uczenia, ładuje go na dostępny sprzęt (CUDA lub CPU) i zgłasza status gotowości (healthy).
3. **Uruchomienie Frontendu**: Kontener `frontend` z aplikacją Streamlit startuje w momencie, gdy backend zgłosi pomyślne uruchomienie i stan zdrowy.

**Zatrzymanie usług z zachowaniem wag modelu:**
```bash
docker compose down
```

**Całkowite usunięcie plików wag modelu:**
```bash
rm -rf models/
```

---

### 🐍 Opcja B — Lokalnie (Python 3.13 + venv)

**Wymagania wstępne:** Python 3.13, menedżer pakietów `pip`.

Utwórz i aktywuj wirtualne środowisko, a następnie zainstaluj niezbędne zależności:
```bash
python3.13 -m venv .venv_suml
source .venv_suml/bin/activate          # Systemy Linux/macOS

pip install -r backend/requirements.txt -r frontend/requirements.txt
```

Skopiuj szablon zmiennych środowiskowych:
```bash
cp .env.example .env
```

**Uruchomienie dla systemów Linux/macOS** — uruchom wszystko jednym skryptem:
```bash
./run.sh
```


Skrypt automatycznie uruchomi proces uczenia, poczeka na gotowość serwera API (backend), a na końcu włączy interfejs Streamlit (frontend).

---

## 🌐 Adresy Usług

| Usługa | Adres URL |
|---------|-----------|
| Streamlit Frontend | http://localhost:8501 |
| FastAPI Swagger UI (Dokumentacja API) | http://localhost:8000/docs |
| Sprawdzenie Stanu Backend (Health Check) | http://localhost:8000/health |

---

## 🔍 Weryfikacja i Rozwiązywanie Problemów

**Sprawdzenie statusu kontenerów (Docker):**
```bash
docker compose ps
```
Zarówno `bird-classifier-backend`, jak i `bird-classifier-frontend` powinny mieć status `healthy`.

**Podgląd logów aplikacji (Docker):**
```bash
docker logs bird-classifier-trainer
docker logs -f bird-classifier-backend
```

---

## 📊 Format Danych Wejściowych i Wyjściowych

### Dane Wejściowe (Input)
Aplikacja akceptuje obrazy przedstawiające ptaki.
* **Formaty plików:** `.jpg`, `.jpeg`, `.png`.
* **Przesyłanie przez interfejs (Streamlit):** Poprzez wgranie pliku w oknie przeglądarki (drag and drop) lub kliknięcie jednego z gotowych przykładów zdjęć zintegrowanych w aplikacji.
* **Przesyłanie przez API (`POST /predict`):** Plik musi zostać przekazany jako żądanie typu `multipart/form-data` pod kluczem o nazwie `image`. Przesyłany plik musi posiadać typ MIME rozpoczynający się od prefiksu `image/` (np. `image/jpeg`).

### Format Wyniku (Output)
Po przesłaniu pliku następuje przetworzenie obrazu (zmiana rozmiaru do formatu dopasowanego do ResNet, normalizacja wartości pikseli) i przesłanie go do modelu sieci neuronowej.
1. **Odpowiedź z API backendowego:** Serwer zwraca strukturę w formacie JSON z listą najbardziej prawdopodobnych klas (gatunków) wraz z poziomem ufności (wartość od 0.0 do 1.0):
   ```json
   {
     "predictions": [
       {
         "species": "012.Yellow_billed_Cuckoo",
         "confidence": 0.9427
       },
       {
         "species": "011.Rusty_Blackbird",
         "confidence": 0.0215
       }
     ]
   }
   ```
2. **Prezentacja na interfejsie użytkownika:**
   * **Karta Głównego Dopasowania:** Prezentuje nazwę gatunku o najwyższym współczynniku pewności przetłumaczoną na język polski (przy pomocy pliku `data/bird_names_pl.json`) oraz oryginalną nazwę angielską wraz z procentowym wskaźnikiem pewności.
   * **Wykresy Prawdopodobieństwa (Top-5):** Lista 5 najbardziej prawdopodobnych gatunków zobrazowana kolorowymi paskami postępu (Zielony ≥ 70%, Pomarańczowy 35–69%, Szary < 35%).
   * **Integracja z Wikipedią:** Przycisk "Dowiedz się więcej" automatycznie przekierowuje użytkownika do artykułu na angielskiej Wikipedii dedykowanego wybranemu gatunkowi.
   * **Baza Gatunków (Encyklopedia):** Osobna zakładka interfejsu umożliwia przeglądanie pełnego spisu 200 obsługiwanych klas ptaków podzielonych na rodziny, wyszukiwanie po nazwach polskich i angielskich oraz przejście do Google Grafika lub Wikipedii.

---

## 📸 Jak Używać Aplikacji — Instrukcja Krok po Kroku

### Krok 1: Uruchomienie i Panel Główny
Po poprawnym uruchomieniu kontenerów lub skryptów lokalnych otwórz przeglądarkę pod adresem [http://localhost:8501](http://localhost:8501). Zobaczysz główny panel aplikacji z informacjami o projekcie po lewej stronie oraz trzema krokami instrukcji na środku.

![Krok 1 - Panel główny aplikacji](docs/images/krok1.png)

### Krok 2: Wybór lub Przesłanie Obrazu
Możesz przetestować aplikację klikając przycisk wyboru pod jednym z gotowych zdjęć przykładowych w sekcji **"Szybki test na przykładach"**, bądź przeciągnąć i upuścić własne zdjęcie ptaka (w formacie JPG, JPEG lub PNG) do obszaru wgrywania plików.

![Krok 2 - Wgrywanie zdjęcia](docs/images/krok2.png)

### Krok 3: Analiza i Wyniki
Aplikacja natychmiast wyśle zdjęcie do backendu, a po prawej stronie wyświetli wyniki klasyfikacji. Zobaczysz gatunek o najwyższym dopasowaniu (wraz z polską i angielską nazwą) oraz listę alternatywnych gatunków z wykresami słupkowymi. Kliknij przycisk **"Dowiedz się więcej"**, aby przejść do artykułu na Wikipedii.

![Krok 3 - Odczytanie wyników analizy](docs/images/krok3.png)

### Krok 4: Przeglądanie Encyklopedii Gatunków
Przejdź do zakładki **"Baza gatunków (200 klas)"**, aby wyszukiwać ptaki według słów kluczowych lub przeglądać pełne drzewo systematyczne zintegrowane w bazie. Z tego poziomu możesz bezpośrednio szukać zdjęć ptaków w Google Grafika lub czytać o nich w encyklopedii.

![Krok 4 - Przeglądanie encyklopedii](docs/images/krok4.png)