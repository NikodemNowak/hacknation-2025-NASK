# Dane Bez Twarzy - Rozwiązanie Zespołu all_in()

Biblioteka do anonimizacji tekstów w języku polskim dla modelu PLLUM.  
Zastępuje dane wrażliwe tokenami (np. `{name}`, `{city}`) oraz wspiera generację danych syntetycznych.

### Nowy pipeline (hybrydowy)
- Krok 1: **RegEx** – twarde dane o stałym formacie (PESEL, email, konta).
- Krok 2: **AI NER (HerBERT)** – kontekstowe encje (imiona, miasta, firmy) -> tagi.
- Krok 3: **LLM Synthesis (PLLuM)** – opcjonalne podmiany tagów na realistyczne dane.

## Wymagania

- Python 3.9+
- System Linux/macOS/Windows

## Instalacja

### Krok 1: Sklonuj repozytorium

```bash
git clone https://github.com/NikodemNowak/hacknation-2025-NASK.git
cd hacknation-2025-NASK
```

### Krok 2: Utwórz wirtualne środowisko

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# lub: venv\Scripts\activate  # Windows
```

### Krok 3: Zainstaluj bibliotekę

```bash
pip install -r requirements.txt
pip install -e .
```

### Krok 4: Pobierz modele (wymagane dla trybu offline)

```bash
python download_models.py
```

> ⚠️ **UWAGA:** Ten krok jest wymagany przed użyciem w trybie offline!

## Użycie

### Podstawowa anonimizacja

```python
from anonymizer import Anonymizer

# Inicjalizacja (modele ładują się automatycznie)
model = Anonymizer()
# albo z własnym modelem NER (fine-tune HerBERT):
# model = Anonymizer(ner_model_path="/path/do/modelu")

# Anonimizacja tekstu
text = "Nazywam się Jan Kowalski, PESEL 90010112345."
result = model.anonymize(text)
print(result)
# Wyjście: "Nazywam się Jan Kowalski, PESEL {pesel}."
```

### Generacja danych syntetycznych

```python
from anonymizer import Anonymizer

model = Anonymizer(use_synthetic=True)

# Najpierw anonimizacja
text = "Mieszkam w Warszawie przy ulicy Długiej 5."
anonymized = model.anonymize(text)
# "Mieszkam w {city} przy ulicy {address}."

# Potem synteza (PLLuM -> realistyczne dane)
synthetic = model.synthesize(anonymized)
# "Mieszkam w Poznaniu przy ulicy Krótkiej 10."
```

### Przetwarzanie wsadowe

```python
from anonymizer import Anonymizer

model = Anonymizer()
texts = [
    "PESEL: 90010112345",
    "Email: jan@example.com",
    "Telefon: +48 123 456 789"
]

results = model.anonymize_batch(texts)
```

### Konfiguracja modeli
- PLLuM API: ustaw zmienne `.env` / środowiskowe `PLLUM_API_KEY`, `PLLUM_BASE_URL` (opcjonalnie), `PLLUM_MODEL_NAME` (opcjonalnie).
- Ścieżka do własnego NER (HerBERT): podaj w kodzie `Anonymizer(ner_model_path="/path/do/modelu")` lub ustaw `NER_MODEL_PATH=/path/do/modelu`.
- Tryb CPU/GPU: domyślnie działa na CPU, jeśli dostępny jest GPU zostanie użyty automatycznie przez pipeline HF.

## Obsługiwane typy danych

### Warstwa RegEx (gotowa)

| Tag | Opis | Przykład |
|-----|------|----------|
| `{pesel}` | PESEL | 90010112345 |
| `{email}` | Adres e-mail | jan@example.com |
| `{phone}` | Numer telefonu | +48 123 456 789 |
| `{bank-account}` | Numer konta IBAN | PL12 3456 7890... |
| `{credit-card-number}` | Karta kredytowa | 1234 5678 9012 3456 |
| `{document-number}` | Numer dowodu | ABC123456 |

### Warstwa NER (HerBERT)

| Tag | Opis |
|-----|------|
| `{name}` | Imię |
| `{surname}` | Nazwisko |
| `{city}` | Miasto |
| `{address}` | Adres |
| `{company}` | Nazwa firmy |
| `{date}` | Data |

## Architektura

```
anonymizer/
├── __init__.py       # Eksportuje Anonymizer
├── core.py           # Główna klasa Anonymizer (hybrydowa)
├── regex_layer.py    # Warstwa RegEx (szybka, stałe formaty)
├── ner_layer.py      # Warstwa NER (kontekstowa, HerBERT + HF pipeline)
├── synthetic.py      # Generator danych syntetycznych
└── utils.py          # Stałe i funkcje pomocnicze
```

## Testy

```bash
pytest tests/ -v
```

## Użyte technologie

- **RegEx** - dla danych o stałym formacie (PESEL, email, telefon)
- **PLLUM** - model `CYFRAGOVPL/pllum-12b-nc-chat-250715` jako główny model NLP
- **LangChain** - integracja z API modelu PLLUM

## Licencja

MIT License - Hackathon NASK 2025
