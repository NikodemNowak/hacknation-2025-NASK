# Dane Bez Twarzy - Rozwiązanie Zespołu all_in()

Biblioteka do anonimizacji tekstów w języku polskim dla modelu PLLUM.  
Zastępuje dane wrażliwe tokenami (np. `[name]`, `[city]`) oraz wspiera generację danych syntetycznych.

## Pipeline (hybrydowy)

1. **RegEx** – twarde dane o stałym formacie (PESEL, email, konta, telefony, daty).
2. **AI NER (HerBERT)** – kontekstowe encje (imiona, miasta, firmy) -> tagi.
3. **LLM Refinement (PLLuM)** – opcjonalna walidacja i dopełnienie tagowania.
4. **LLM Synthesis (PLLuM)** – opcjonalne podmiany tagów na realistyczne dane syntetyczne.

## Wymagania

- Python 3.9+
- System Linux/macOS/Windows

## Instalacja

```bash
pip install git+https://github.com/NikodemNowak/hacknation-2025-NASK.git
```

## Użycie

### Podstawowa anonimizacja

```python
from anonymizer import Anonymizer

# Inicjalizacja - model NER pobierze się automatycznie przy pierwszym użyciu
anon = Anonymizer()

# Anonimizacja tekstu
text = "Nazywam się Jan Kowalski, mieszkam w Warszawie, PESEL 90010112345."
result = anon.anonymize(text)
print(result)
# Output: "Nazywam się [name] [surname], mieszkam w [city], [pesel]."
```

### Z nawiasami klamrowymi (styl {tag})

```python
from anonymizer import Anonymizer

anon = Anonymizer(use_brackets=False)  # użyj {tag} zamiast [tag]

text = "Jan Kowalski, tel: +48 123 456 789"
result = anon.anonymize(text)
# Output: "Jan Kowalski, tel: {phone}"
```

### Generacja danych syntetycznych (z PLLuM API)

```python
from anonymizer import Anonymizer

# Wymaga ustawienia PLLUM_API_KEY w .env lub zmiennej środowiskowej
anon = Anonymizer(use_synthetic=True)

# Najpierw anonimizacja
text = "Mieszkam w Warszawie przy ulicy Długiej 5."
anonymized = anon.anonymize(text)
# "[city] przy ulicy [address]."

# Potem synteza (PLLuM -> realistyczne dane)
synthetic = anon.synthesize(anonymized)
# "Mieszkam w Poznaniu przy ulicy Krótkiej 10."
```

### Przetwarzanie wsadowe

```python
from anonymizer import Anonymizer

anon = Anonymizer()
texts = [
    "PESEL: 90010112345",
    "Email: jan@example.com",
    "Telefon: +48 123 456 789"
]

results = anon.anonymize_batch(texts)
for orig, res in zip(texts, results):
    print(f"{orig} -> {res}")
```

### Konfiguracja

| Zmienna środowiskowa | Opis |
|---------------------|------|
| `PLLUM_API_KEY` | Klucz API do PLLuM (wymagany dla syntezy) |
| `PLLUM_BASE_URL` | URL API PLLuM (opcjonalnie) |
| `PLLUM_MODEL_NAME` | Nazwa modelu PLLuM (opcjonalnie) |

## Obsługiwane typy danych

### Warstwa RegEx

| Tag | Opis | Przykład |
|-----|------|----------|
| `[pesel]` | PESEL | 90010112345 |
| `[email]` | Adres e-mail | jan@example.com |
| `[phone]` | Numer telefonu | +48 123 456 789 |
| `[bank-account]` | Numer konta IBAN | PL12 3456 7890... |
| `[credit-card-number]` | Karta kredytowa | 1234 5678 9012 3456 |
| `[document-number]` | Numer dowodu | ABC123456 |
| `[date]` | Data | 15.03.2024 |

### Warstwa NER (HerBERT)

| Tag | Opis |
|-----|------|
| `[name]` | Imię |
| `[surname]` | Nazwisko |
| `[city]` | Miasto |
| `[address]` | Adres |
| `[company]` | Nazwa firmy |
| `[school-name]` | Nazwa szkoły |
| `[job-title]` | Stanowisko |
| `[age]` | Wiek |
| `[date-of-birth]` | Data urodzenia |
| `[sex]` | Płeć |
| `[religion]` | Religia |
| `[political-view]` | Pogląd polityczny |
| `[ethnicity]` | Pochodzenie etniczne |
| `[sexual-orientation]` | Orientacja seksualna |
| `[health]` | Dane zdrowotne |
| `[relative]` | Informacja o krewnych |
| `[username]` | Nazwa użytkownika |
| `[secret]` | Poufne dane |

## Architektura

```
anonymizer/
├── __init__.py       # Eksportuje Anonymizer, PLLUMClient, etc.
├── core.py           # Główna klasa Anonymizer (hybrydowa)
├── regex_layer.py    # Warstwa RegEx (szybka, stałe formaty)
├── ner_layer.py      # Warstwa NER (kontekstowa, HerBERT)
├── synthetic.py      # Generator danych syntetycznych
├── pllum_client.py   # Klient PLLuM (API + offline)
└── utils.py          # Stałe i funkcje pomocnicze
```

## Modele

- **NER**: [`Nikod3m/hacknation-2025-NASK-herbert-ner-v2`](https://huggingface.co/Nikod3m/hacknation-2025-NASK-herbert-ner-v2) - fine-tuned HerBERT
- **LLM**: `CYFRAGOVPL/pllum-12b-nc-chat-250715` - dla syntezy i rafinacji

## Testy

```bash
pytest tests/ -v
```
