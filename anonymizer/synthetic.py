"""
Synthetic data module.

Replaces anonymization tokens ({name}, {city}, etc.) with realistic but
fake data. By default uses PLLuM; if API is unavailable, falls back to
local sample data.

PLLuM prompt:
"You are a data assistant. In the given text, replace all tokens like
{name}, {city} with realistic Polish data, preserving grammatical form.
Do not change the rest of the text."
"""

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .pllum_client import PLLUMClient

DEFAULT_PROMPT = (
    "Jesteś asystentem danych po polsku. Zamień tokeny (np. {name}, {surname}, "
    "{age}, {city}, {address}, {company}, {phone}, {email}, {pesel}, {date}) "
    "na realistyczne, lecz fikcyjne dane w poprawnych formach gramatycznych.\n"
    "- Zachowaj sens zdania; nie dodawaj nic poza podstawieniem tokenów.\n"
    "- Ustal płeć z kontekstu (zaimki 'ona/on', zawody żeńskie/męskie, słowa typu 'Mama', 'Pan').\n"
    "- Odmieniaj imiona i nazwiska zgodnie z przypadkiem i rodzajem: "
    "np. 'Mama {surname}' -> 'Mama Kowalskiego'; 'z {name}' -> 'z Piotrem' / 'z Anną'.\n"
    "- Jeśli występują osobno {name} i {surname}, wybierz pasującą parę (ta sama osoba) "
    "i odmieniaj każde wystąpienie poprawnie.\n"
    "- Ten sam token w tekście ma być spójny (ta sama wartość w kolejnych wystąpieniach, odmieniona gdy trzeba).\n"
    "- Dla {age} użyj liczby, nie słów; 'ma {age} lat' powinno brzmieć naturalnie.\n"
    "- Nie dodawaj cudzysłowów, komentarzy, wyjaśnień ani nowych zdań.\n"
    "\nPrzykłady:\n"
    "Tekst: Mama {surname} ma na imię {name}. O niej mówią, że jest pracowita.\n"
    "Wynik: Mama Kowalskiego ma na imię Anna. O niej mówią, że jest pracowita.\n"
    "Tekst: Mama {surname} ma na imię {name}. Jest ona {age} lat. Jest nauczycielką.\n"
    "Wynik: Mama Nowaka ma na imię Maria. Jest ona 38 lat. Jest nauczycielką.\n"
    "Tekst: Spotkałem się z {name} w {city} w dniu {date}.\n"
    "Wynik: Spotkałem się z Piotrem w Krakowie w dniu 14.03.2021.\n"
    "\nTekst:\n{input_text}\n\nWynik:"
)


@dataclass
class SyntheticData:
    """Container for synthetic sample data."""

    names: List[str]
    surnames: List[str]
    cities: List[str]
    streets: List[str]
    companies: List[str]


class SyntheticGenerator:
    """
    Generates synthetic replacements for tokens (LLM or local fallback)
    while preserving grammar.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        use_llm: bool = True,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        prompt: Optional[str] = None,
        use_brackets: bool = False,
        offline: Optional[bool] = None,
    ):
        """
        Initialize generator.

        Args:
            seed: Random seed (for reproducibility)
            use_llm: Use PLLuM; fallback to local data if not available
            api_key: PLLuM API key (env fallback)
            base_url: PLLuM API base URL
            model_name: PLLuM model name
            prompt: Custom PLLuM prompt
        """
        if seed is not None:
            random.seed(seed)

        self.use_llm = use_llm
        self.use_brackets = use_brackets
        self.offline = offline
        self.prompt_template = prompt or DEFAULT_PROMPT
        self._client_params = {
            "api_key": api_key,
            "base_url": base_url,
            "model_name": model_name,
        }
        self._pllum_client: Optional[PLLUMClient] = None

        self._init_data()

    def _ensure_client(self) -> None:
        """Lazy-init PLLuM client."""
        if self._pllum_client is not None or not self.use_llm:
            return

        kwargs = {
            k: v for k, v in self._client_params.items() if v is not None
        }
        # Auto-switch to offline if no API key provided.
        resolved_offline = (
            self.offline
            if self.offline is not None
            else not kwargs.get("api_key")
        )
        kwargs["offline"] = resolved_offline
        self._pllum_client = PLLUMClient(**kwargs)

    def _init_data(self):
        """Initialize synthetic datasets (offline fallback)."""

        # Polish first names
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

        # Polish surnames
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

        # Polish cities
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

        # Polish streets
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

        # Companies (fictional)
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

        # Map tags to generators
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
        """Generate random first name."""
        return random.choice(self.names)

    def _gen_surname(self) -> str:
        """Generate random surname."""
        return random.choice(self.surnames)

    def _gen_city(self) -> str:
        """Generate random city."""
        return random.choice(self.cities)

    def _gen_address(self) -> str:
        """Generate random address."""
        street = random.choice(self.streets)
        number = random.randint(1, 150)
        apartment = random.choice(["", f"/{random.randint(1, 50)}"])
        return f"ul. {street} {number}{apartment}"

    def _gen_company(self) -> str:
        """Generate random company."""
        return random.choice(self.companies)

    def _gen_phone(self) -> str:
        """Generate random phone."""
        prefix = random.choice(
            ["50", "51", "53", "60", "66", "69", "72", "78", "79", "88"]
        )
        number = "".join([str(random.randint(0, 9)) for _ in range(7)])
        return f"+48 {prefix}{number[:3]} {number[3:6]} {number[6:]}"

    def _gen_email(self) -> str:
        """Generate random email."""
        name = random.choice(self.names).lower()
        surname = random.choice(self.surnames).lower()
        domain = random.choice(
            ["gmail.com", "wp.pl", "onet.pl", "o2.pl", "interia.pl"]
        )
        separator = random.choice([".", "_", ""])
        return f"{name}{separator}{surname}@{domain}"

    def _gen_pesel(self) -> str:
        """Generate random PESEL (synthetic, not validated)."""
        year = random.randint(50, 99)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        series = random.randint(1000, 9999)
        checksum = random.randint(0, 9)
        return f"{year:02d}{month:02d}{day:02d}{series}{checksum}"

    def _gen_date(self) -> str:
        """Generate random date."""
        day = random.randint(1, 28)
        month = random.randint(1, 12)
        year = random.randint(1980, 2023)
        return f"{day:02d}.{month:02d}.{year}"

    def _gen_age(self) -> str:
        """Generate random age."""
        return str(random.randint(18, 80))

    def _gen_bank_account(self) -> str:
        """Generate random bank account."""
        digits = "".join([str(random.randint(0, 9)) for _ in range(26)])
        return f"PL{digits[:2]} {digits[2:6]} {digits[6:10]} {digits[10:14]} {digits[14:18]} {digits[18:22]} {digits[22:26]}"

    def _gen_credit_card(self) -> str:
        """Generate random credit card."""
        digits = "".join([str(random.randint(0, 9)) for _ in range(16)])
        return f"{digits[:4]} {digits[4:8]} {digits[8:12]} {digits[12:16]}"

    def _gen_document_number(self) -> str:
        """Generate random document number."""
        letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=3))
        numbers = "".join([str(random.randint(0, 9)) for _ in range(6)])
        return f"{letters}{numbers}"

    def _has_tags(self, text: str) -> bool:
        """Detect tags in either {} or [] style."""
        return bool(re.search(r"\{[a-z\-]+\}|\[[a-z\-]+\]", text))

    def _normalize_to_curly(self, text: str) -> str:
        """
        Convert [tag] to {tag} so LLM prompt stays consistent and can
        reuse the same value for repeated tags.
        """
        return re.sub(r"\[([a-z\-]+)\]", r"{\1}", text)

    def _replace_tags_locally(self, anonymized_text: str) -> str:
        """Fallback: replace tags locally using sample dictionaries."""
        tag_pattern = re.compile(r"\{([a-z\-]+)\}|\[([a-z\-]+)\]")

        def replace_tag(match):
            tag = match.group(1) or match.group(2)
            if tag in self._generators:
                return self._generators[tag]()
            return match.group(0)

        return tag_pattern.sub(replace_tag, anonymized_text)

    def synthesize(self, anonymized_text: str) -> str:
        """
        Replace tokens with synthetic data.
        Uses PLLuM; falls back to local generation when unavailable.
        """
        if not anonymized_text or not self._has_tags(anonymized_text):
            return anonymized_text

        if self.use_llm:
            try:
                self._ensure_client()
                if self._pllum_client:
                    prompt_input = self._normalize_to_curly(anonymized_text)
                    prompt = self.prompt_template.format(
                        input_text=prompt_input
                    )
                    response = self._pllum_client.generate(prompt)
                    if response:
                        return response.strip()
            except Exception:
                pass

        return self._replace_tags_locally(anonymized_text)

    def synthesize_batch(self, texts: List[str]) -> List[str]:
        """Synthesize many texts."""
        return [self.synthesize(text) for text in texts]
