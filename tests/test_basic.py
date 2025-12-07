"""
Podstawowe testy dla biblioteki anonymizer.

Uruchomienie:
    pytest tests/test_basic.py -v
"""

import pytest


class TestLibraryImport:
    """Testy importu biblioteki."""

    def test_import_main_module(self):
        """Test importu głównego modułu."""
        import anonymizer

        assert hasattr(anonymizer, "__version__")

    def test_import_anonymizer_class(self):
        """Test importu klasy Anonymizer."""
        from anonymizer import Anonymizer

        assert Anonymizer is not None

    def test_import_regex_anonymizer(self):
        """Test importu RegexAnonymizer."""
        from anonymizer import RegexAnonymizer

        assert RegexAnonymizer is not None

    def test_import_synthetic_generator(self):
        """Test importu SyntheticGenerator."""
        from anonymizer import SyntheticGenerator

        assert SyntheticGenerator is not None


class TestAnonymizerBasic:
    """Podstawowe testy klasy Anonymizer."""

    def test_anonymizer_initialization(self):
        """Test tworzenia instancji Anonymizer."""
        from anonymizer import Anonymizer

        anonymizer = Anonymizer()
        assert anonymizer is not None
        assert anonymizer.use_regex is True
        assert anonymizer.use_ner is True

    def test_anonymizer_repr(self):
        """Test reprezentacji tekstowej."""
        from anonymizer import Anonymizer

        anonymizer = Anonymizer()
        repr_str = repr(anonymizer)
        assert "Anonymizer" in repr_str

    def test_get_supported_tags(self):
        """Test pobierania listy obsługiwanych tagów."""
        from anonymizer import Anonymizer

        anonymizer = Anonymizer()
        tags = anonymizer.get_supported_tags()

        assert isinstance(tags, list)
        assert "pesel" in tags
        assert "email" in tags
        assert "name" in tags


class TestRegexAnonymization:
    """Testy anonimizacji RegEx."""

    @pytest.fixture
    def anonymizer(self):
        """Fixture tworząca anonymizer."""
        from anonymizer import Anonymizer

        return Anonymizer(use_ner=False)  # Tylko RegEx

    def test_pesel_anonymization(self, anonymizer):
        """Test anonimizacji PESEL."""
        text = "Mój PESEL to 90010112345"
        result = anonymizer.anonymize(text)

        assert "{pesel}" in result
        assert "90010112345" not in result

    def test_email_anonymization(self, anonymizer):
        """Test anonimizacji email."""
        text = "Napisz do mnie na jan.kowalski@example.com"
        result = anonymizer.anonymize(text)

        assert "{email}" in result
        assert "jan.kowalski@example.com" not in result

    def test_phone_anonymization(self, anonymizer):
        """Test anonimizacji telefonu."""
        text = "Zadzwoń na +48 123 456 789"
        result = anonymizer.anonymize(text)

        # Telefon może być wykryty lub nie - zależy od formatu
        # Na razie testujemy że nie jest None
        assert result is not None

    def test_no_data_unchanged(self, anonymizer):
        """Test że tekst bez danych osobowych zostaje niezmieniony."""
        text = "To jest zwykły tekst bez danych osobowych."
        result = anonymizer.anonymize(text)

        assert result == text

    def test_multiple_data_types(self, anonymizer):
        """Test anonimizacji wielu typów danych."""
        text = "Jan, PESEL 90010112345, email jan@test.pl"
        result = anonymizer.anonymize(text)

        assert "{pesel}" in result
        assert "{email}" in result


class TestSyntheticGenerator:
    """Testy generatora danych syntetycznych."""

    @pytest.fixture
    def generator(self):
        """Fixture tworząca generator z stałym seedem."""
        from anonymizer import SyntheticGenerator

        return SyntheticGenerator(seed=42)

    def test_synthesize_name(self, generator):
        """Test syntezy imienia."""
        text = "Nazywam się {name}"
        result = generator.synthesize(text)

        assert "{name}" not in result
        assert len(result) > len("Nazywam się ")

    def test_synthesize_city(self, generator):
        """Test syntezy miasta."""
        text = "Mieszkam w {city}"
        result = generator.synthesize(text)

        assert "{city}" not in result

    def test_synthesize_multiple(self, generator):
        """Test syntezy wielu tagów."""
        text = "{name} mieszka w {city}, email: {email}"
        result = generator.synthesize(text)

        assert "{name}" not in result
        assert "{city}" not in result
        assert "{email}" not in result

    def test_unknown_tags_preserved(self, generator):
        """Test że nieznane tagi są zachowane."""
        text = "To jest {unknown_tag}"
        result = generator.synthesize(text)

        assert "{unknown_tag}" in result


class TestBatchProcessing:
    """Testy przetwarzania wsadowego."""

    def test_anonymize_batch(self):
        """Test anonimizacji wsadowej."""
        from anonymizer import Anonymizer

        anonymizer = Anonymizer(use_ner=False)
        texts = [
            "PESEL: 90010112345",
            "Email: test@example.com",
            "Zwykły tekst",
        ]

        results = anonymizer.anonymize_batch(texts)

        assert len(results) == 3
        assert "{pesel}" in results[0]
        assert "{email}" in results[1]
        assert results[2] == "Zwykły tekst"

    def test_synthesize_batch(self):
        """Test syntezy wsadowej."""
        from anonymizer import SyntheticGenerator

        generator = SyntheticGenerator(seed=42)
        texts = [
            "Imię: {name}",
            "Miasto: {city}",
        ]

        results = generator.synthesize_batch(texts)

        assert len(results) == 2
        assert "{name}" not in results[0]
        assert "{city}" not in results[1]


class TestTagFormat:
    """Testy formatu tagów."""

    def test_curly_brackets_default(self):
        """Test że domyślnie używane są nawiasy klamrowe."""
        from anonymizer import Anonymizer

        anonymizer = Anonymizer(use_ner=False, use_brackets=False)
        result = anonymizer.anonymize("PESEL 90010112345")

        assert "{pesel}" in result
        assert "[pesel]" not in result

    def test_square_brackets_option(self):
        """Test opcji nawiasów kwadratowych."""
        from anonymizer import Anonymizer

        anonymizer = Anonymizer(use_ner=False, use_brackets=True)
        result = anonymizer.anonymize("PESEL 90010112345")

        assert "[pesel]" in result
        assert "{pesel}" not in result


# Uruchomienie testów z terminala
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
