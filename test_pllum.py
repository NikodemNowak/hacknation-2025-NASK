import os
import random

from anonymizer.core import Anonymizer

MODEL_PATH = "models/herbert_ner_v2"
ORIGINAL_PATH = "nask_train/original.txt"
ANON_PATH = "nask_train/anon.txt"
SAMPLES = 10
SEED = 42


def load_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def main():
    random.seed(SEED)

    print(f"🔄 Ładowanie modelu z: {MODEL_PATH}...")
    anonymizer = Anonymizer(
        ner_model_path=MODEL_PATH,
        use_brackets=True,
        use_synthetic=False,  # przebieg 1: bez PLLuM
    )
    anonymizer._init_ner_layer()

    print(f"📂 Wczytywanie danych: {ORIGINAL_PATH} i {ANON_PATH}...")
    original = load_lines(ORIGINAL_PATH)
    anon_ref = load_lines(ANON_PATH)

    if len(original) != len(anon_ref):
        raise ValueError(
            f"Pliki mają różną liczbę linii: original={len(original)}, anon={len(anon_ref)}"
        )

    total = len(original)
    indices = random.sample(range(total), min(SAMPLES, total))

    print(f"✅ Wczytano {total} linii, losuję {len(indices)} przykładów\n")

    for idx in indices:
        src = original[idx]
        ref = anon_ref[idx]

        # 1) RegEx + NER (bez PLLuM)
        anonymized = anonymizer.anonymize(src, with_synthetic=False)
        # 2) RegEx + NER + PLLuM walidacja/uzupełnienie + synteza
        anonymized_llm = anonymizer.anonymize(src, with_synthetic=True)

        print("=" * 100)
        print(f"PRZYKŁAD {idx + 1}/{total}")
        print("=" * 100)

        # Oryginał
        print("\n🔵 ORYGINAŁ:\n" + (src if len(src) <= 800 else src[:800] + "..."))
        # Wersja referencyjna NASK
        print("\n🟣 REFERENCJA NASK (anon):\n" + (ref if len(ref) <= 800 else ref[:800] + "..."))
        # Nasza anonimizacja (regex + NER)
        print("\n🟢 NASZE (Regex + NER):\n" + (anonymized if len(anonymized) <= 800 else anonymized[:800] + "..."))
        # Nasza synteza (regex + NER + PLLuM/lokalny fallback)
        print(
            "\n🟡 NASZE (Regex + NER + PLLuM/fallback):\n"
            + (anonymized_llm if len(anonymized_llm) <= 800 else anonymized_llm[:800] + "...")
        )
        print()

    print("\n" + "=" * 100)
    print("KONIEC TESTÓW PLLuM")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
