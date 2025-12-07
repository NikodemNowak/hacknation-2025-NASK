"""Constants and helper functions for anonymizer."""

from typing import Dict, List, Set


# =============================================================================
# Anonymization tags (aligned with competition requirements)
# =============================================================================

# Tags handled by regex layer (fixed-format data)
REGEX_TAGS: Set[str] = {
    "pesel",  # PESEL (11 cyfr)
    "email",  # Adresy e-mail
    "phone",  # Numery telefonów
    "bank-account",  # Numery kont bankowych (IBAN)
    "credit-card-number",  # Numery kart kredytowych
    "document-number",  # Numery dowodów osobistych / paszportów
    "date",  # Daty (ogólne)
}

# Tags handled by NER/LLM (contextual)
NER_TAGS: Set[str] = {
    # Dane identyfikacyjne osobowe
    "name",
    "surname",
    "age",
    "date-of-birth",
    "sex",
    "religion",
    "political-view",
    "ethnicity",
    "sexual-orientation",
    "health",
    "relative",
    # Lokalizacja / kontakt
    "city",
    "address",
    # Dokumenty / identyfikatory
    "company",
    "school-name",
    "job-title",
    # Inne
    "nationality",
    "country",
    "voivodeship",
    "district",
    "zip-code",
    "username",
    "secret",
}

# All supported tags
ALL_TAGS: Set[str] = REGEX_TAGS | NER_TAGS


# =============================================================================
# Helper functions
# =============================================================================


def format_tag(tag: str, use_brackets: bool = False) -> str:
    """
    Format a tag with chosen bracket style.
    """
    if use_brackets:
        return f"[{tag}]"
    return f"{{{tag}}}"


def is_valid_tag(tag: str) -> bool:
    """Check if tag is supported."""
    return tag in ALL_TAGS


def normalize_text(text: str) -> str:
    """
    Normalize text:
    - collapse multiple spaces
    - trim whitespace
    """
    # Zamień wiele spacji na jedną
    import re

    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_tag_description(tag: str) -> str:
    """Return human-friendly tag description."""
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
