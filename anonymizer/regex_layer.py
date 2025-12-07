"""
Warstwa RegEx do anonimizacji danych o stałym formacie.

Obsługiwane typy danych:
- PESEL -> {pesel}
- E-mail -> {email}
- Numer telefonu -> {phone}
- Numer konta bankowego (IBAN) -> {bank-account}
- Numer karty kredytowej -> {credit-card-number}
- Numer dowodu osobistego -> {document-number}

Użycie:
    from anonymizer.regex_layer import RegexAnonymizer
    
    anonymizer = RegexAnonymizer()
    text = "Mój PESEL to 90010112345, a email jan@example.com"
    result = anonymizer.anonymize(text)
    # "Mój PESEL to {pesel}, a email {email}"
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .utils import format_tag


@dataclass
class AnonymizationResult:
    """Wynik anonimizacji tekstu."""
    original_text: str
    anonymized_text: str
    replacements: List[Tuple[str, str, int]]  # (original, tag, position)


class RegexAnonymizer:
    """
    Klasa do anonimizacji danych osobowych w tekście za pomocą wyrażeń regularnych.
    
    Obsługuje:
    - PESEL (11 cyfr)
    - E-mail
    - Numery telefonów (polskie, różne formaty)
    - Numery kont bankowych (IBAN)
    - Numery kart kredytowych
    - Numery dowodów osobistych (polskie)
    """
    
    def __init__(self, use_brackets: bool = False):
        """
        Inicjalizacja anonimizera.
        
        Args:
            use_brackets: Jeśli True, używa nawiasów kwadratowych [tag],
                         jeśli False, używa klamrowych {tag}
        """
        self.use_brackets = use_brackets
        self._compile_patterns()
    
    def _format_tag(self, tag: str) -> str:
        """Formatuje tag zgodnie z wybranym stylem."""
        return format_tag(tag, self.use_brackets)
    
    def _compile_patterns(self):
        """Kompiluje wszystkie wzorce regex."""
        
        # PESEL - 11 cyfr, może mieć różne zniekształcenia
        self.pesel_pattern = re.compile(
            r'\b'
            r'(?:'
                r'[0-9oOlI!|]{11}'
            r')'
            r'\b',
            re.IGNORECASE
        )
        
        # Email - standardowy format z domeną
        self.email_pattern = re.compile(
            r'[a-zA-Z0-9._%+\-łśżźćńóęą|!]{1,64}'
            r'@'
            r'[a-zA-Z0-9.\-łśżźćńóęą|]{1,255}'
            r'\.'
            r'[a-zA-Z]{2,10}',
            re.IGNORECASE
        )
        
        # Telefon - różne polskie formaty (+ jest częścią tagu)
        self.phone_pattern = re.compile(
            r'(?:\+\s*)?'  # opcjonalny + na początku (włączony do matcha)
            r'(?:'
                r'(?:[4hA]\s*[8B]\s+)?'  # opcjonalne 48
                r'(?:[0-9oOlI!|BSsAaEeGgqQhHzZ]{2,3}[\s\-\.]*){3,4}[0-9oOlI!|BSsAaEeGgqQhHzZ]{2,3}'
            r')',
            re.IGNORECASE
        )
        
        # Numer konta bankowego (IBAN polski) - 26 cyfr
        self.bank_account_pattern = re.compile(
            r'\b'
            r'(?:'
                r'(?:PL\s*)?'
                r'(?:[0-9oOlI!|\s]{2,4}[\s\-]?){6,7}'
            r')'
            r'\b',
            re.IGNORECASE
        )
        
        # Karta kredytowa - 16 cyfr
        self.credit_card_pattern = re.compile(
            r'\b'
            r'(?:'
                r'[0-9]{4}[\s\-]?'
                r'[0-9]{4}[\s\-]?'
                r'[0-9]{4}[\s\-]?'
                r'[0-9]{4}'
            r')'
            r'\b'
        )
        
        # Numer dowodu osobistego (polski)
        self.document_number_pattern = re.compile(
            r'\b'
            r'(?:'
                r'[A-Za-z]{2,3}\s*[0-9oOlI!|]{4,6}'
                r'|'
                r'[0-9oOlI!|]{4}[\-\s][0-9oOlI!|]{4}[\-\s][0-9oOlI!|]{4}'
            r')'
            r'\b',
            re.IGNORECASE
        )
    
    def _is_valid_pesel(self, text: str) -> bool:
        """Sprawdza czy tekst może być PESELem."""
        cleaned = text.upper()
        cleaned = cleaned.replace('O', '0').replace('I', '1').replace('L', '1')
        cleaned = cleaned.replace('!', '1').replace('|', '1')
        
        if not re.match(r'^\d{11}$', cleaned):
            return False
        return True
    
    def _is_valid_phone(self, text: str) -> bool:
        """Sprawdza czy tekst może być numerem telefonu."""
        cleaned = re.sub(r'[\s\-\.\(\)\+]', '', text)
        cleaned = cleaned.upper()
        
        replacements = {
            'O': '0', 'I': '1', 'L': '1', '!': '1', '|': '1',
            'B': '8', 'S': '5', 'A': '4', 'E': '3', 'G': '9',
            'Q': '9', 'H': '4', 'Z': '2'
        }
        for char, digit in replacements.items():
            cleaned = cleaned.replace(char, digit)
        
        if cleaned.startswith('48'):
            cleaned = cleaned[2:]
        
        if not re.match(r'^\d{9}$', cleaned):
            return False
        return True
    
    def _is_valid_email(self, text: str) -> bool:
        """Sprawdza czy tekst może być emailem."""
        if '@' not in text or '.' not in text.split('@')[-1]:
            return False
        return True
    
    def _is_valid_bank_account(self, text: str) -> bool:
        """Sprawdza czy tekst może być numerem konta bankowego."""
        cleaned = re.sub(r'[\s\-]', '', text.upper())
        if cleaned.startswith('PL'):
            cleaned = cleaned[2:]
        cleaned = cleaned.replace('O', '0').replace('I', '1').replace('L', '1')
        return len(cleaned) >= 20 and re.match(r'^\d+$', cleaned)
    
    def _is_valid_credit_card(self, text: str) -> bool:
        """Sprawdza czy tekst może być numerem karty kredytowej."""
        cleaned = re.sub(r'[\s\-]', '', text)
        return len(cleaned) == 16 and cleaned.isdigit()
    
    def _is_valid_document_number(self, text: str) -> bool:
        """Sprawdza czy tekst może być numerem dowodu osobistego."""
        cleaned = re.sub(r'[\s\-]', '', text.upper())
        
        if re.match(r'^[A-Z]{2,3}\d{4,7}$', cleaned):
            return True
        if re.match(r'^\d{12}$', cleaned):
            return True
        return False

    def anonymize(self, text: str) -> str:
        """
        Anonimizuje tekst zastępując dane osobowe odpowiednimi tagami.
        
        Kolejność zastępowania (od najbardziej specyficznych):
        1. Email (najbardziej charakterystyczny - zawiera @)
        2. Numer konta bankowego (najdłuższy - 26 cyfr)
        3. Karta kredytowa (16 cyfr)
        4. PESEL (11 cyfr)
        5. Numer dowodu osobistego (różne formaty)
        6. Telefon (9 cyfr, ale różne formaty)
        
        Args:
            text: Tekst do anonimizacji
            
        Returns:
            Zanonimizowany tekst
        """
        result = text
        
        # 1. Email
        result = self._replace_emails(result)
        
        # 2. Numer konta bankowego
        result = self._replace_bank_accounts(result)
        
        # 3. Karta kredytowa
        result = self._replace_credit_cards(result)
        
        # 4. PESEL
        result = self._replace_pesels(result)
        
        # 5. Numer dowodu
        result = self._replace_document_numbers(result)
        
        # 6. Telefon
        result = self._replace_phones(result)
        
        return result
    
    def _replace_emails(self, text: str) -> str:
        """Zamienia emaile na tag."""
        def replace(match):
            if self._is_valid_email(match.group(0)):
                return self._format_tag('email')
            return match.group(0)
        return self.email_pattern.sub(replace, text)
    
    def _replace_bank_accounts(self, text: str) -> str:
        """Zamienia numery kont na tag."""
        def replace(match):
            if self._is_valid_bank_account(match.group(0)):
                return self._format_tag('bank-account')
            return match.group(0)
        return self.bank_account_pattern.sub(replace, text)
    
    def _replace_credit_cards(self, text: str) -> str:
        """Zamienia numery kart na tag."""
        def replace(match):
            if self._is_valid_credit_card(match.group(0)):
                return self._format_tag('credit-card-number')
            return match.group(0)
        return self.credit_card_pattern.sub(replace, text)
    
    def _replace_pesels(self, text: str) -> str:
        """Zamienia numery PESEL na tag."""
        def replace(match):
            if self._is_valid_pesel(match.group(0)):
                return self._format_tag('pesel')
            return match.group(0)
        return self.pesel_pattern.sub(replace, text)
    
    def _replace_document_numbers(self, text: str) -> str:
        """Zamienia numery dowodów na tag."""
        def replace(match):
            if self._is_valid_document_number(match.group(0)):
                return self._format_tag('document-number')
            return match.group(0)
        return self.document_number_pattern.sub(replace, text)
    
    def _replace_phones(self, text: str) -> str:
        """Zamienia numery telefonów na tag."""
        def replace(match):
            if self._is_valid_phone(match.group(0)):
                return self._format_tag('phone')
            return match.group(0)
        return self.phone_pattern.sub(replace, text)
    
    def anonymize_detailed(self, text: str) -> AnonymizationResult:
        """
        Anonimizuje tekst i zwraca szczegółowy wynik z listą zamian.
        
        Returns:
            AnonymizationResult z oryginalnym tekstem, zanonimizowanym i listą zamian
        """
        anonymized = self.anonymize(text)
        
        replacements = []
        
        for match in self.email_pattern.finditer(text):
            if self._is_valid_email(match.group(0)):
                replacements.append((match.group(0), 'email', match.start()))
        
        for match in self.pesel_pattern.finditer(text):
            if self._is_valid_pesel(match.group(0)):
                replacements.append((match.group(0), 'pesel', match.start()))
        
        for match in self.phone_pattern.finditer(text):
            if self._is_valid_phone(match.group(0)):
                replacements.append((match.group(0), 'phone', match.start()))
        
        return AnonymizationResult(
            original_text=text,
            anonymized_text=anonymized,
            replacements=replacements
        )


def anonymize_text(text: str, use_brackets: bool = False) -> str:
    """
    Funkcja pomocnicza do szybkiej anonimizacji tekstu.
    
    Args:
        text: Tekst do anonimizacji
        use_brackets: Czy używać nawiasów kwadratowych [tag] (True) 
                     czy klamrowych {tag} (False)
    
    Returns:
        Zanonimizowany tekst
    """
    anonymizer = RegexAnonymizer(use_brackets=use_brackets)
    return anonymizer.anonymize(text)
