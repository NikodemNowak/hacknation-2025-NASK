"""
Stałe i funkcje pomocnicze dla biblioteki pllum_anonymizer.
"""

from typing import Dict, List, Set


# =============================================================================
# TAGI ANONIMIZACJI - zgodne z wymogami konkursu "Dane bez twarzy"
# =============================================================================

# Tagi obsługiwane przez warstwę regex (dane o stałym formacie)
REGEX_TAGS: Set[str] = {
    "pesel",              # PESEL (11 cyfr)
    "email",              # Adresy e-mail
    "phone",              # Numery telefonów
    "bank-account",       # Numery kont bankowych (IBAN)
    "credit-card-number", # Numery kart kredytowych
    "document-number",    # Numery dowodów osobistych
}

# Tagi obsługiwane przez warstwę NER (wymagają modelu NLP)
NER_TAGS: Set[str] = {
    "name",               # Imiona
    "surname",            # Nazwiska
    "city",               # Miasta
    "address",            # Adresy (ulica, numer domu)
    "company",            # Nazwy firm/organizacji
    "age",                # Wiek
    "sex",                # Płeć
    "date",               # Daty
    "date-of-birth",      # Data urodzenia
    "religion",           # Religia
    "political-view",     # Poglądy polityczne
    "health",             # Informacje zdrowotne
    "relative",           # Relacje rodzinne
    "nationality",        # Narodowość
    "sexual-orientation", # Orientacja seksualna
    "country",            # Kraj
    "voivodeship",        # Województwo
    "district",           # Powiat
    "zip-code",           # Kod pocztowy
}

# Wszystkie obsługiwane tagi
ALL_TAGS: Set[str] = REGEX_TAGS | NER_TAGS


# =============================================================================
# FUNKCJE POMOCNICZE
# =============================================================================

def format_tag(tag: str, use_brackets: bool = False) -> str:
    """
    Formatuje tag anonimizacji.
    
    Args:
        tag: Nazwa tagu (np. "name", "pesel")
        use_brackets: Jeśli True, używa nawiasów kwadratowych [tag],
                     jeśli False, używa klamrowych {tag}
    
    Returns:
        Sformatowany tag
    """
    if use_brackets:
        return f"[{tag}]"
    return f"{{{tag}}}"


def is_valid_tag(tag: str) -> bool:
    """Sprawdza czy tag jest prawidłowy."""
    return tag in ALL_TAGS


def normalize_text(text: str) -> str:
    """
    Normalizuje tekst do przetwarzania.
    
    - Usuwa nadmiarowe białe znaki
    - Normalizuje znaki specjalne
    """
    # Zamień wiele spacji na jedną
    import re
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_tag_description(tag: str) -> str:
    """Zwraca opis tagu po polsku."""
    descriptions = {
        "pesel": "PESEL",
        "email": "Adres e-mail",
        "phone": "Numer telefonu",
        "bank-account": "Numer konta bankowego",
        "credit-card-number": "Numer karty kredytowej",
        "document-number": "Numer dowodu osobistego",
        "name": "Imię",
        "surname": "Nazwisko",
        "city": "Miasto",
        "address": "Adres",
        "company": "Nazwa firmy",
        "age": "Wiek",
        "sex": "Płeć",
        "date": "Data",
        "date-of-birth": "Data urodzenia",
        "religion": "Religia",
        "political-view": "Poglądy polityczne",
        "health": "Informacje zdrowotne",
        "relative": "Relacje rodzinne",
        "nationality": "Narodowość",
        "sexual-orientation": "Orientacja seksualna",
        "country": "Kraj",
        "voivodeship": "Województwo",
        "district": "Powiat",
        "zip-code": "Kod pocztowy",
    }
    return descriptions.get(tag, tag)
