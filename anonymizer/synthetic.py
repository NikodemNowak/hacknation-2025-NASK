"""
Moduł do generacji danych syntetycznych.

Zastępuje tokeny anonimizacji ({name}, {city}, etc.) realistycznymi,
ale fikcyjnymi danymi. Domyślnie wykorzystuje PLLuM, aby zastąpić tagi
wiarygodnymi danymi, a w razie braku dostępu do API korzysta z lokalnych
zbiorów przykładowych danych.

Prompt dla PLLuM:
"Jesteś asystentem danych. W podanym tekście zastąp wszystkie tagi typu
{name}, {city} realistycznymi polskimi danymi, zachowując odmianę
 gramatyczną. Nie zmieniaj reszty tekstu."
"""

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .pllum_client import PLLUMClient

DEFAULT_PROMPT = (
    "Jesteś asystentem danych. W podanym tekście zastąp wszystkie tagi typu "
    "{{name}}, {{city}} realistycznymi polskimi danymi, zachowując odmianę gramatyczną. "
    "Nie zmieniaj reszty tekstu.\n\nTekst:\n{input_text}\n\nWynik:"
)


@dataclass
class SyntheticData:
    """Zbiór danych syntetycznych dla różnych kategorii."""

    names: List[str]
    surnames: List[str]
    cities: List[str]
    streets: List[str]
    companies: List[str]


class SyntheticGenerator:
    """
    Generator danych syntetycznych.

    Zastępuje tokeny anonimizacji realistycznymi danymi (LLM lub lokalnie),
    zachowując poprawność gramatyczną tekstu.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        use_llm: bool = True,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """
        Inicjalizacja generatora.

        Args:
            seed: Ziarno dla generatora losowego (dla powtarzalności)
            use_llm: Czy używać PLLuM do syntezy (fallback do lokalnych danych)
            api_key: Klucz API PLLuM (fallback do .env/ENV)
            base_url: Adres API PLLuM
            model_name: Nazwa modelu PLLuM
            prompt: Niestandardowy prompt dla PLLuM
        """
        if seed is not None:
            random.seed(seed)

        self.use_llm = use_llm
        self.prompt_template = prompt or DEFAULT_PROMPT
        self._client_params = {
            "api_key": api_key,
            "base_url": base_url,
            "model_name": model_name,
        }
        self._pllum_client: Optional[PLLUMClient] = None

        self._init_data()

    def _ensure_client(self) -> None:
        """Leniewo inicjalizuje klienta PLLuM."""
        if self._pllum_client is not None or not self.use_llm:
            return

        kwargs = {k: v for k, v in self._client_params.items() if v is not None}
        self._pllum_client = PLLUMClient(**kwargs)

    def _init_data(self):
        """Inicjalizuje zbiory danych syntetycznych (fallback offline)."""

        # Polskie imiona (mianownik)
        self.names_male = [
            "Adam",
            "Piotr",
            "Jan",
            "Michał",
            "Krzysztof",
            "Andrzej",
            "Tomasz",
            "Paweł",
            "Marcin",
            "Jakub",
            "Marek",
            "Łukasz",
            "Mateusz",
            "Wojciech",
            "Robert",
            "Kamil",
            "Sebastian",
            "Filip",
        ]

        self.names_female = [
            "Anna",
            "Maria",
            "Katarzyna",
            "Małgorzata",
            "Agnieszka",
            "Barbara",
            "Ewa",
            "Krystyna",
            "Elżbieta",
            "Magdalena",
            "Joanna",
            "Monika",
            "Aleksandra",
            "Natalia",
            "Karolina",
        ]

        self.names = self.names_male + self.names_female

        # Polskie nazwiska
        self.surnames = [
            "Nowak",
            "Kowalski",
            "Wiśniewski",
            "Wójcik",
            "Kowalczyk",
            "Kamiński",
            "Lewandowski",
            "Zieliński",
            "Szymański",
            "Woźniak",
            "Dąbrowski",
            "Kozłowski",
            "Jankowski",
            "Mazur",
            "Wojciechowski",
            "Kwiatkowski",
            "Krawczyk",
            "Kaczmarek",
            "Piotrowski",
            "Grabowski",
        ]

        # Polskie miasta
        self.cities = [
            "Warszawa",
            "Kraków",
            "Łódź",
            "Wrocław",
            "Poznań",
            "Gdańsk",
            "Szczecin",
            "Bydgoszcz",
            "Lublin",
            "Białystok",
            "Katowice",
            "Gdynia",
            "Częstochowa",
            "Radom",
            "Sosnowiec",
            "Toruń",
            "Kielce",
            "Rzeszów",
            "Gliwice",
            "Zabrze",
        ]

        # Polskie ulice
        self.streets = [
            "Główna",
            "Ogrodowa",
            "Kościelna",
            "Polna",
            "Leśna",
            "Krótka",
            "Długa",
            "Słoneczna",
            "Szkolna",
            "Lipowa",
            "Parkowa",
            "Kwiatowa",
            "Zielona",
            "Brzozowa",
            "Kolejowa",
            "Wolności",
            "Wojska Polskiego",
            "3 Maja",
            "Mickiewicza",
            "Piłsudskiego",
        ]

        # Firmy (nazwy fikcyjne)
        self.companies = [
            "TechPol Sp. z o.o.",
            "Innova Systems S.A.",
            "Global Trade Sp. z o.o.",
            "MediCare Polska",
            "EkoSolutions",
            "DataSoft Sp. z o.o.",
            "PrimeTech S.A.",
            "SmartBiz Polska",
            "ProServices Sp. z o.o.",
            "NextGen Solutions",
            "Alpha Consulting",
            "Beta Industries",
        ]

        # Mapowanie tagów na generators
        self._generators: Dict[str, callable] = {
            "name": self._gen_name,
            "surname": self._gen_surname,
            "city": self._gen_city,
            "address": self._gen_address,
            "company": self._gen_company,
            "phone": self._gen_phone,
            "email": self._gen_email,
            "pesel": self._gen_pesel,
            "date": self._gen_date,
            "age": self._gen_age,
            "bank-account": self._gen_bank_account,
            "credit-card-number": self._gen_credit_card,
            "document-number": self._gen_document_number,
        }

    def _gen_name(self) -> str:
        """Generuje losowe imię."""
        return random.choice(self.names)

    def _gen_surname(self) -> str:
        """Generuje losowe nazwisko."""
        return random.choice(self.surnames)

    def _gen_city(self) -> str:
        """Generuje losowe miasto."""
        return random.choice(self.cities)

    def _gen_address(self) -> str:
        """Generuje losowy adres."""
        street = random.choice(self.streets)
        number = random.randint(1, 150)
        apartment = random.choice(["", f"/{random.randint(1, 50)}"])
        return f"ul. {street} {number}{apartment}"

    def _gen_company(self) -> str:
        """Generuje losową nazwę firmy."""
        return random.choice(self.companies)

    def _gen_phone(self) -> str:
        """Generuje losowy numer telefonu."""
        prefix = random.choice(["50", "51", "53", "60", "66", "69", "72", "78", "79", "88"])
        number = "".join([str(random.randint(0, 9)) for _ in range(7)])
        return f"+48 {prefix}{number[:3]} {number[3:6]} {number[6:]}"

    def _gen_email(self) -> str:
        """Generuje losowy email."""
        name = random.choice(self.names).lower()
        surname = random.choice(self.surnames).lower()
        domain = random.choice(["gmail.com", "wp.pl", "onet.pl", "o2.pl", "interia.pl"])
        separator = random.choice([".", "_", ""])
        return f"{name}{separator}{surname}@{domain}"

    def _gen_pesel(self) -> str:
        """Generuje losowy PESEL (syntetyczny, nie walidowany)."""
        year = random.randint(50, 99)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        series = random.randint(1000, 9999)
        checksum = random.randint(0, 9)
        return f"{year:02d}{month:02d}{day:02d}{series}{checksum}"

    def _gen_date(self) -> str:
        """Generuje losową datę."""
        day = random.randint(1, 28)
        month = random.randint(1, 12)
        year = random.randint(1980, 2023)
        return f"{day:02d}.{month:02d}.{year}"

    def _gen_age(self) -> str:
        """Generuje losowy wiek."""
        return str(random.randint(18, 80))

    def _gen_bank_account(self) -> str:
        """Generuje losowy numer konta."""
        digits = "".join([str(random.randint(0, 9)) for _ in range(26)])
        return f"PL{digits[:2]} {digits[2:6]} {digits[6:10]} {digits[10:14]} {digits[14:18]} {digits[18:22]} {digits[22:26]}"

    def _gen_credit_card(self) -> str:
        """Generuje losowy numer karty."""
        digits = "".join([str(random.randint(0, 9)) for _ in range(16)])
        return f"{digits[:4]} {digits[4:8]} {digits[8:12]} {digits[12:16]}"

    def _gen_document_number(self) -> str:
        """Generuje losowy numer dowodu."""
        letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=3))
        numbers = "".join([str(random.randint(0, 9)) for _ in range(6)])
        return f"{letters}{numbers}"

    def _has_tags(self, text: str) -> bool:
        return bool(re.search(r"\{[a-z\-]+\}", text))

    def _replace_tags_locally(self, anonymized_text: str) -> str:
        """Fallback: zastępuje tagi lokalnie na bazie gotowych słowników."""
        tag_pattern = re.compile(r"\{([a-z\-]+)\}")

        def replace_tag(match):
            tag = match.group(1)
            if tag in self._generators:
                return self._generators[tag]()
            return match.group(0)

        return tag_pattern.sub(replace_tag, anonymized_text)

    def synthesize(self, anonymized_text: str) -> str:
        """
        Zastępuje tokeny anonimizacji danymi syntetycznymi.
        Używa PLLuM do generacji danych, a w razie braku dostępu – lokalnego fallbacku.
        """
        if not anonymized_text or not self._has_tags(anonymized_text):
            return anonymized_text

        if self.use_llm:
            try:
                self._ensure_client()
                if self._pllum_client:
                    prompt = self.prompt_template.format(input_text=anonymized_text)
                    response = self._pllum_client.generate(prompt)
                    if response:
                        return response.strip()
            except Exception:
                # Jeśli PLLuM jest niedostępny, przechodzimy na fallback
                pass

        return self._replace_tags_locally(anonymized_text)

    def synthesize_batch(self, texts: List[str]) -> List[str]:
        """Syntetyzuje wiele tekstów."""
        return [self.synthesize(text) for text in texts]
