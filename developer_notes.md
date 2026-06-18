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