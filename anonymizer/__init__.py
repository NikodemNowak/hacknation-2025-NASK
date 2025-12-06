"""
anonymizer - Narzędzie do anonimizacji danych dla modelu PLLUM.

Biblioteka do anonimizacji tekstów w języku polskim. Zastępuje dane wrażliwe
tokenami (np. {name}, {city}) oraz wspiera generację danych syntetycznych.

Użycie:
    from anonymizer import Anonymizer
    
    model = Anonymizer()
    text = "Nazywam się Jan Kowalski, PESEL 90010112345."
    result = model.anonymize(text)
    # Wynik: "Nazywam się {name} {surname}, PESEL {pesel}."
"""

from .core import Anonymizer
from .regex_layer import RegexAnonymizer
from .synthetic import SyntheticGenerator
from .pllum_client import PLLUMClient

__version__ = "0.1.0"
__author__ = "all_in()"

__all__ = [
    "Anonymizer",
    "RegexAnonymizer",
    "SyntheticGenerator",
    "PLLUMClient",
    "__version__",
]
