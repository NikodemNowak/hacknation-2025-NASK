6# Dane Bez Twarzy - Rozwiązanie Zespołu all_in()

Biblioteka do anonimizacji tekstów w języku polskim dla modelu PLLUM.  
Zastępuje dane wrażliwe tokenami (np. `{name}`, `{city}`) oraz wspiera generację danych syntetycznych.

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
from pllum_anonymizer import Anonymizer

# Inicjalizacja (modele ładują się automatycznie)
model = Anonymizer()

# Anonimizacja tekstu
text = "Nazywam się Jan Kowalski, PESEL 90010112345."
result = model.anonymize(text)
print(result)
# Wyjście: "Nazywam się Jan Kowalski, PESEL {pesel}."
```

### Generacja danych syntetycznych

```python
from pllum_anonymizer import Anonymizer

model = Anonymizer()

# Najpierw anonimizacja
text = "Mieszkam w Warszawie przy ulicy Długiej 5."
anonymized = model.anonymize(text)
# "Mieszkam w Warszawie przy ulicy Długiej 5."

# Potem synteza (gdy NER będzie zaimplementowany)
synthetic = model.synthesize(anonymized)
# "Mieszkam w Krakowie przy ulicy Krótkiej 10."
```

### Przetwarzanie wsadowe

```python
from pllum_anonymizer import Anonymizer

model = Anonymizer()
texts = [
    "PESEL: 90010112345",
    "Email: jan@example.com",
    "Telefon: +48 123 456 789"
]

results = model.anonymize_batch(texts)
```

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

### Warstwa NER (placeholder)

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
pllum_anonymizer/
├── __init__.py       # Eksportuje Anonymizer
├── core.py           # Główna klasa Anonymizer (hybrydowa)
├── regex_layer.py    # Warstwa RegEx (szybka, stałe formaty)
├── ner_layer.py      # Warstwa NER (kontekstowa, placeholder)
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
