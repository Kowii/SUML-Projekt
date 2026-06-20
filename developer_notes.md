# Notatki deweloperskie

## Ruff — linter i formatter

### Co to jest?

Ruff to narzędzie do statycznej analizy i formatowania kodu Python, napisane w Rust. Zastępuje w jednym narzędziu takie rzeczy jak `flake8`, `isort` czy `black`, działając przy tym wielokrotnie szybciej.

W tym projekcie używamy go do pilnowania zgodności z PEP8 i spójności stylu kodu. Konfiguracja znajduje się w `pyproject.toml`.

> **Uwaga:** Ruff **nie zastępuje pylinta** — wymóg projektu zaliczeniowego to wynik pylint ≥ 8. Ruff służy do bieżącej pracy i formatowania, pylint uruchamiamy osobno przed oddaniem.

---

### Instalacja

```bash
pip install ruff
```

Lub jeśli korzystasz z venv projektu (zalecane):

```bash
.venv_suml/bin/pip install ruff
```

---

### Najważniejsze komendy

**Sprawdź błędy w całym projekcie:**
```bash
ruff check .
```

**Sprawdź konkretny plik:**
```bash
ruff check backend/app/main.py
```

**Napraw automatycznie to, co się da:**
```bash
ruff check . --fix
```

**Sformatuj kod (odpowiednik `black`):**
```bash
ruff format .
```

**Sprawdź co zostałoby zmienione bez faktycznej zmiany:**
```bash
ruff format . --diff
```

**Sprawdź i napraw, a potem sformatuj — wszystko naraz:**
```bash
ruff check . --fix && ruff format .
```

---

### Konfiguracja w projekcie (`pyproject.toml`)

```toml
[tool.ruff]
target-version = "py313"   # wersja Pythona
line-length = 100           # max długość linii

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP"]
# E  — błędy stylu (pycodestyle)
# F  — błędy logiczne (pyflakes), np. nieużywane importy
# W  — ostrzeżenia stylu
# I  — kolejność importów (isort)
# UP — sugestie unowocześnienia składni (pyupgrade)

[tool.ruff.format]
quote-style = "double"     # cudzysłowy podwójne
indent-style = "space"     # wcięcia spacjami
```

Żeby wyłączyć konkretną regułę w jednej linii:
```python
x = 1  # noqa: E501
```

Żeby wyłączyć dla całego pliku — dodaj na początku:
```python
# ruff: noqa: E501
```

---

### Pylint — sprawdzenie przed oddaniem

Pylint uruchamiamy ręcznie żeby sprawdzić wynik (wymagane ≥ 8):

```bash
pylint backend/app backend/train frontend/app.py
```

Konfiguracja pylinta jest też w `pyproject.toml`:
```toml
[tool.pylint."messages control"]
disable = ["C0114", "C0115", "C0116"]   # pomija brak docstringów
```

---

## Trening modelu — jak to działa

### ResNet50 — co to jest?

ResNet50 to gotowa sieć neuronowa wytrenowana przez Meta na **1,2 miliona zdjęć** (ImageNet — koty, psy, samochody, itd.). Ma 50 warstw i "wie" jak rozpoznawać kształty, tekstury, kontury. My jej nie tworzymy od zera — **pobieramy gotowe wagi**.

Wyobraź sobie to tak: ResNet50 to kucharz, który umie gotować wszystko. Chcemy go "przekwalifikować" żeby specjalizował się tylko w ptakach.

---

### Dwufazowy fine-tuning — co dokładnie robimy

**Faza 1 — trenujemy TYLKO ostatnią warstwę (głowę klasyfikatora)**

```
[ResNet50 zamrożony — wagi nie zmieniają się]
         ↓
[Nowa warstwa FC: 2048 → 200 klas ptaków]  ← tylko to się uczy
```

ResNet50 ma na końcu warstwę `fc` (fully connected), która pierwotnie klasyfikuje 1000 klas ImageNet. Wycinamy ją i wstawiamy nową — 200 wyjść (200 gatunków ptaków z CUB-200-2011). Reszta sieci jest **zamrożona** — jej wagi się nie zmieniają.

Trwa ~5 epok, szybko, niska stopa uczenia `1e-3`.

---

**Faza 2 — odmrażamy CAŁĄ sieć i fine-tunujemy wszystko**

```
[ResNet50 odmrożony — wszystkie wagi mogą się zmieniać]
         ↓
[Warstwa FC: 2048 → 200 klas]
```

Teraz cała sieć się uczy, ale bardzo powoli (`1e-4` — 10x mniejsza stopa). Chodzi o to żeby delikatnie "dostroić" już nauczone cechy do ptaków, a nie zniszczyć tego co ResNet50 już wie.

Trwa ~20 epok.

---

### Dlaczego tak a nie inaczej?

| Podejście | Problem |
|---|---|
| Trening od zera | Potrzeba milionów zdjęć i tygodni GPU |
| Fine-tuning tylko głowy | Szybko, ale mniej dokładny |
| **Dwufazowy fine-tuning** | **Najlepszy balans — to co robimy** |

Dataset CUB-200-2011 ma tylko ~60 zdjęć na klasę (razem ~12 000). Przy takim małym zbiorze trening od zera dałby fatalny wynik. Dlatego korzystamy z wiedzy którą ResNet50 już ma.