"""
RegEx layer for fixed-format data anonymization.

Supported types:
- PESEL -> {pesel}
- E-mail -> {email}
- Phone number -> {phone}
- Bank account (IBAN) -> {bank-account}
- Credit card -> {credit-card-number}
- Document number -> {document-number}
- Dates -> {date}

Usage:
    from anonymizer.regex_layer import RegexAnonymizer

    anonymizer = RegexAnonymizer()
    text = "Mój PESEL to 90010112345, a email jan@example.com"
    result = anonymizer.anonymize(text)
    # "Mój PESEL to [pesel], a email [email]"
"""

import re
from dataclasses import dataclass
from typing import List, Tuple

from .utils import format_tag

OCR_REPLACEMENTS = {
    'O': '0',
    'o': '0',
    'I': '1',
    'i': '1',
    'l': '1',
    'L': '1',
    '!': '1',
    '|': '1',
    'Z': '2',
    'z': '2',
    'E': '3',
    'e': '3',
    'A': '4',
    'a': '4',
    'H': '4',
    'h': '4',
    'S': '5',
    's': '5',
    'G': '6',
    'g': '6',
    'T': '7',
    't': '7',
    'B': '8',
    'b': '8',
    'Q': '9',
    'q': '9',
}

DIGIT_LIKE = r'0-9oOiIlL!|zZeEaAhHsStTbBgGqQ'


@dataclass
class AnonymizationResult:
    """Result of anonymization."""

    original_text: str
    anonymized_text: str
    replacements: List[Tuple[str, str, int]]


def clean_to_digits(text: str) -> str:
    """Convert OCR-distorted characters to digits."""
    result = text
    for char, digit in OCR_REPLACEMENTS.items():
        result = result.replace(char, digit)
    return result


class RegexAnonymizer:
    """
    Anonymizes fixed-format PII with regular expressions.

    Supports:
    - PESEL (11 digits)
    - E-mail
    - Phone numbers (PL formats)
    - Bank accounts (IBAN)
    - Credit cards
    - Document numbers (PL)
    - Dates
    """

    def __init__(self, use_brackets: bool = False):
        """
        Initialize anonymizer.

        Args:
            use_brackets: If True, uses square brackets [tag],
                         if False, uses curly braces {tag}
        """
        self.use_brackets = use_brackets
        self._compile_patterns()

    def _format_tag(self, tag: str) -> str:
        """Formats tag according to bracket style."""
        return format_tag(tag, self.use_brackets)

    def _compile_patterns(self):
        """Compile all regex patterns."""

        self.pesel_pattern = re.compile(
            rf'\b[{DIGIT_LIKE}]{{11}}\b', re.IGNORECASE
        )

        # Dates (general): dd.mm.yyyy, dd-mm-yyyy, yyyy-mm-dd, dd/mm/yyyy
        self.date_pattern = re.compile(
            r'\b(?:'
            r'(?:[0-3]?\d[.\-\/][0-1]?\d[.\-\/](?:19|20)?\d{2})'
            r'|'
            r'(?:\d{4}[.\-\/][0-1]?\d[.\-\/][0-3]?\d)'
            r')\b'
        )

        # Email - standard format with domain
        self.email_pattern = re.compile(
            r'[a-zA-Z0-9._%+\-łśżźćńóęą|!]{1,64}'
            r'@'
            r'[a-zA-Z0-9.\-łśżźćńóęą|]{1,255}'
            r'\.'
            r'[a-zA-Z]{2,10}',
            re.IGNORECASE,
        )

        # Phone - various Polish formats (+ is part of tag)
        # Format: +48 XXX XXX XXX or XXX XXX XXX or XXX-XXX-XXX etc.
        # Also handles OCR distortions (letters instead of digits)
        self.phone_pattern = re.compile(
            rf'(?:\+\s*)?'  # optional +
            rf'(?:[{DIGIT_LIKE}]{{2}}[\s\-\.]*)?'  # optional 48
            rf'[{DIGIT_LIKE}]{{2,3}}[\s\-\.]*'  # first segment
            rf'[{DIGIT_LIKE}]{{2,3}}[\s\-\.]*'  # second segment
            rf'[{DIGIT_LIKE}]{{2,3}}',  # third segment
            re.IGNORECASE,
        )

        # Bank account number (Polish IBAN) - 26 digits with OCR distortions
        self.bank_account_pattern = re.compile(
            rf'\b'
            rf'(?:PL\s*)?'  # optional PL prefix
            rf'(?:[{DIGIT_LIKE}]{{2,4}}[\s\-]?){{5,7}}'  # groups of 2-4 digits
            rf'[{DIGIT_LIKE}]{{1,4}}'  # last group
            rf'\b',
            re.IGNORECASE,
        )

        # Credit card - 16 digits with OCR distortions
        self.credit_card_pattern = re.compile(
            rf'\b'
            rf'[{DIGIT_LIKE}]{{4}}[\s\-]?'
            rf'[{DIGIT_LIKE}]{{4}}[\s\-]?'
            rf'[{DIGIT_LIKE}]{{4}}[\s\-]?'
            rf'[{DIGIT_LIKE}]{{4}}'
            rf'\b',
            re.IGNORECASE,
        )

        # ID card number (Polish)
        # Format: ABC123456 or 1234-5678-9012
        self.document_number_pattern = re.compile(
            rf'\b'
            rf'(?:'
            rf'[A-Za-z]{{2,3}}\s*[{DIGIT_LIKE}]{{4,6}}'
            rf'|'
            rf'[{DIGIT_LIKE}]{{4}}[\-\s][{DIGIT_LIKE}]{{4}}[\-\s][{DIGIT_LIKE}]{{4}}'
            rf')'
            rf'\b',
            re.IGNORECASE,
        )

    def _is_valid_pesel(self, text: str) -> bool:
        """Validate if text can be a PESEL."""
        cleaned = re.sub(r'[\s\-]', '', text)
        cleaned = clean_to_digits(cleaned)

        if not re.match(r'^\d{11}$', cleaned):
            return False
        return True

    def _is_valid_phone(self, text: str) -> bool:
        """Validate if text can be a phone number."""
        cleaned = re.sub(r'[\s\-\.\(\)\+]', '', text)
        cleaned = clean_to_digits(cleaned)

        # Remove prefix 48 if present
        if cleaned.startswith('48') and len(cleaned) > 9:
            cleaned = cleaned[2:]

        # Polish number is 9 digits
        if len(cleaned) == 9 and cleaned.isdigit():
            return True
        return False

    def _is_valid_email(self, text: str) -> bool:
        """Validate if text can be an email."""
        if '@' not in text or '.' not in text.split('@')[-1]:
            return False
        return True

    def _is_valid_bank_account(self, text: str) -> bool:
        """Validate if text can be a bank account."""
        cleaned = re.sub(r'[\s\-]', '', text.upper())
        if cleaned.startswith('PL'):
            cleaned = cleaned[2:]
        cleaned = clean_to_digits(cleaned)

        # Polish IBAN has 26 digits (24 + 2 check digits)
        # But we also accept shorter formats (20-26 digits)
        if len(cleaned) >= 20 and len(cleaned) <= 28 and cleaned.isdigit():
            return True
        return False

    def _is_valid_credit_card(self, text: str) -> bool:
        """Validate if text can be a credit card."""
        cleaned = re.sub(r'[\s\-]', '', text)
        cleaned = clean_to_digits(cleaned)
        return len(cleaned) == 16 and cleaned.isdigit()

    def _is_valid_document_number(self, text: str) -> bool:
        """Validate if text can be a document number."""
        cleaned = re.sub(r'[\s\-]', '', text.upper())

        # Format: ABC123456 (3 letters + 6 digits) or AB1234567 (2 letters + 7 digits)
        if re.match(r'^[A-Z]{2,3}[0-9]{4,7}$', clean_to_digits(cleaned)):
            return True
        # Format: 1234-5678-9012 (12 digits)
        cleaned_digits = clean_to_digits(cleaned)
        if re.match(r'^\d{12}$', cleaned_digits):
            return True
        return False

    def anonymize(self, text: str) -> str:
        """
        Anonymize text by replacing fixed-format PII with tags.

        Order (most specific first):
        1. Email
        2. Bank account (26 digits)
        3. Credit card (16 digits)
        4. Dates
        5. PESEL
        6. Document number
        7. Phone
        """
        result = text

        # 1. Email
        result = self._replace_emails(result)

        # 2. Bank account number
        result = self._replace_bank_accounts(result)

        # 3. Credit card
        result = self._replace_credit_cards(result)

        # 4. Dates
        result = self._replace_dates(result)

        # 5. PESEL
        result = self._replace_pesels(result)

        # 6. Document number
        result = self._replace_document_numbers(result)

        # 7. Phone
        result = self._replace_phones(result)

        return result

    def _replace_emails(self, text: str) -> str:
        """Replace emails with tag."""

        def replace(match):
            if self._is_valid_email(match.group(0)):
                return self._format_tag('email')
            return match.group(0)

        return self.email_pattern.sub(replace, text)

    def _replace_bank_accounts(self, text: str) -> str:
        """Replace bank accounts with tag."""

        def replace(match):
            if self._is_valid_bank_account(match.group(0)):
                return self._format_tag('bank-account')
            return match.group(0)

        return self.bank_account_pattern.sub(replace, text)

    def _replace_credit_cards(self, text: str) -> str:
        """Replace credit cards with tag."""

        def replace(match):
            if self._is_valid_credit_card(match.group(0)):
                return self._format_tag('credit-card-number')
            return match.group(0)

        return self.credit_card_pattern.sub(replace, text)

    def _replace_dates(self, text: str) -> str:
        """Replace dates with {date} tag."""

        def replace(match):
            return self._format_tag('date')

        return self.date_pattern.sub(replace, text)

    def _replace_pesels(self, text: str) -> str:
        """Replace PESEL numbers with tag."""

        def replace(match):
            if self._is_valid_pesel(match.group(0)):
                return self._format_tag('pesel')
            return match.group(0)

        return self.pesel_pattern.sub(replace, text)

    def _replace_document_numbers(self, text: str) -> str:
        """Replace document numbers with tag."""

        def replace(match):
            if self._is_valid_document_number(match.group(0)):
                return self._format_tag('document-number')
            return match.group(0)

        return self.document_number_pattern.sub(replace, text)

    def _replace_phones(self, text: str) -> str:
        """Replace phone numbers with tag."""

        def replace(match):
            if self._is_valid_phone(match.group(0)):
                return self._format_tag('phone')
            return match.group(0)

        return self.phone_pattern.sub(replace, text)

    def anonymize_detailed(self, text: str) -> AnonymizationResult:
        """Return detailed anonymization result with replacements list."""
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
            replacements=replacements,
        )


def anonymize_text(text: str, use_brackets: bool = False) -> str:
    """
    Helper function for quick text anonymization.

    Args:
        text: Text to anonymize
        use_brackets: Whether to use square brackets [tag] (True)
                     or curly braces {tag} (False)

    Returns:
        Anonymized text
    """
    anonymizer = RegexAnonymizer(use_brackets=use_brackets)
    return anonymizer.anonymize(text)
