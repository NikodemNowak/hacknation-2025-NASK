import os

from anonymizer.core import Anonymizer

# Upewnij się, że ścieżka jest poprawna (względem miejsca uruchomienia skryptu)
MODEL_PATH = "models/herbert_ner_model"

def run_test():
    print(f"🔄 Ładowanie modelu z: {MODEL_PATH}...")

    # 1. Inicjalizacja z Twoim modelem
    # Jeśli prompt zadziałał poprawnie, Anonymizer powinien przyjmować parametr model_path
    try:
        anonymizer = Anonymizer(ner_model_path=MODEL_PATH)
        print("✅ Model załadowany pomyślnie!")
        # Włącz NER od razu, żeby mieć dostęp do debugowania encji
        anonymizer._init_ner_layer()
    except Exception as e:
        print(f"❌ Błąd ładowania modelu: {e}")
        print("Czy folder models/herbert_ner_model zawiera plik config.json?")
        return

    # 2. Przykładowe teksty do testów
    test_cases = [
        # Prosty test imienia (NER)
        "Spotkałem dzisiaj Jana Kowalskiego w sklepie.",

        # Test hybrydowy (Regex + NER)
        "Pani Anna Nowak (PESEL: 90010112345) mieszka w Warszawie na ulicy Złotej.",

        # Test kontekstu (czy nie usunie 'Odry' jako rzeki)
        "Mój kolega Marek pojechał nad rzekę Odrę.",
    ]

    print("\n--- ROZPOCZYNAM TESTY ANONIMIZACJI ---\n")

    for text in test_cases:
        print(f"📝 ORYGINAŁ: {text}")

        # Diagnostyka warstwy NER
        if anonymizer.use_ner and anonymizer._ner_layer:
            entities = anonymizer._ner_layer.extract_entities(text, debug=True)
            if entities:
                print("🔍 Encje NER:")
                for ent in entities:
                    print(
                        f"  - {ent.label} ({ent.start}-{ent.end}): '{ent.text}' -> {ent.tag}"
                    )
            else:
                print("ℹ️  Brak encji zwróconych przez model NER.")

        # Uruchomienie anonimizacji (zwróć uwagę czy wyniki są poprawne)
        result = anonymizer.anonymize(text)

        print(f"🔒 WYNIK:    {result}")
        print("-" * 50)

if __name__ == "__main__":
    run_test()
